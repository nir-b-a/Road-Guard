const { spawn } = require('child_process');
const path = require('path');
const Drive = require('../models/Drive');
const Notification = require('../models/Notification');
const r2 = require('../services/r2');

const FPS = 30;

// Video size gate (declared at init, re-checked from the real object at complete).
const VIDEO_MIN_BYTES = parseInt(process.env.VIDEO_MIN_BYTES || String(5 * 1024 * 1024), 10);            // 5 MB (same floor as the Android app)
const VIDEO_MAX_BYTES = parseInt(process.env.VIDEO_MAX_BYTES || String(4 * 1024 * 1024 * 1024), 10);      // 4 GB
// Admission guard: reject new uploads once the bucket is this full (R2 free tier ~10 GB).
const R2_MAX_BYTES    = parseInt(process.env.R2_MAX_BYTES || String(9.5 * 1024 * 1024 * 1024), 10);

// The "7-file" contract: the video (arbitrary name) + these 6 fixed-name artifacts.
const SENSOR_FILES = ['frames.csv', 'gps.csv', 'gravity.csv', 'gyro.csv', 'linacc.csv', 'intrinsics.json'];
const VIDEO_EXTS   = ['.mp4', '.mov', '.mpeg', '.m4v'];

// Drive.files keys, mapped from the on-storage file name.
const FILE_KEYS = { 'frames.csv': 'frames', 'gps.csv': 'gps', 'gravity.csv': 'gravity', 'gyro.csv': 'gyro', 'linacc.csv': 'linacc', 'intrinsics.json': 'intrinsics' };

// Strip anything that could escape the session prefix; both sessionId and the video
// name become parts of an object key.
const safeName = (raw) => String(raw || '').trim().replace(/[^a-zA-Z0-9_.-]/g, '_');

const parseSpeedSamples = (speed_json) => {
    try {
        const raw = speed_json ? (typeof speed_json === 'string' ? JSON.parse(speed_json) : speed_json) : [];
        return raw.map(s => ({
            frame:       Math.round((s.timestampMs / 1000) * FPS),
            timestampMs: s.timestampMs,
            speedKmh:    s.speedKmh,
            lat:         s.lat,
            lon:         s.lon
        }));
    } catch { return []; }
};

/**
 * STEP 1 — POST /api/driver/upload/init
 * Validates the request (all 7 files declared, video size in range, bucket not full),
 * creates the Drive in 'created' state, and returns one presigned PUT URL per file so the
 * app can upload every artifact DIRECTLY to R2. No file bytes touch this server.
 *
 * Body: { sessionId, videoName, videoSize, speed_json? }
 * Returns: { driveId, sessionId, uploads: { <fileName>: <presignedPutUrl> } }
 */
const initUpload = async (req, res) => {
    const { sessionId, videoName, videoSize, speed_json } = req.body;
    if (!sessionId)  return res.status(400).json({ success: false, message: 'sessionId is required', data: null });
    if (!videoName)  return res.status(400).json({ success: false, message: 'videoName is required', data: null });

    const size = Number(videoSize);
    if (!Number.isFinite(size) || size < VIDEO_MIN_BYTES || size > VIDEO_MAX_BYTES) {
        return res.status(400).json({ success: false, message: `Video size must be between ${VIDEO_MIN_BYTES} and ${VIDEO_MAX_BYTES} bytes`, data: null });
    }
    const safeVideo = safeName(videoName);
    if (!VIDEO_EXTS.some(ext => safeVideo.toLowerCase().endsWith(ext))) {
        return res.status(400).json({ success: false, message: 'videoName must be one of ' + VIDEO_EXTS.join(', '), data: null });
    }

    const safeSession = safeName(sessionId);
    const existing = await Drive.findOne({ sessionId: safeSession });
    if (existing) {
        if (existing.status !== 'created') {
            return res.status(409).json({ success: false, message: 'sessionId already used', data: null });
        }
        // Stale incomplete upload (init was called but complete never was) — clean up so
        // the retry can start fresh without a sessionId conflict.
        if (r2.isConfigured()) await r2.deletePrefix(safeSession + '/');
        await existing.deleteOne();
    }

    // Admission guard: don't let a burst of big uploads overflow the free-tier bucket.
    if (r2.isConfigured()) {
        const used = await r2.bucketUsageBytes();
        if (used + size > R2_MAX_BYTES) {
            return res.status(507).json({ success: false, message: 'Storage is full — try again after pending drives are processed', data: null });
        }
    }

    // Each session is its own top-level prefix in the bucket: <sessionId>/<file>
    // (the sessionId already carries a "session_" prefix, e.g. session_1719_/video.mp4).
    const prefix  = safeSession + '/';
    const videoKey = prefix + safeVideo;

    // One presigned PUT URL per file. Declared size was range-checked above; the real
    // size is re-validated from the uploaded object at /upload/complete.
    const uploads = {};
    uploads[safeVideo] = await r2.presignPut(videoKey);
    const files = { video: videoKey };
    for (const name of SENSOR_FILES) {
        const key = prefix + name;
        uploads[name] = await r2.presignPut(key);
        files[FILE_KEYS[name]] = key;
    }

    const drive = await Drive.create({
        driverId:     req.user._id,
        sessionId:    safeSession,
        videoPath:    videoKey,
        files,
        speedSamples: parseSpeedSamples(speed_json),
        fps:          FPS,
        status:       'created'
    });

    res.status(201).json({ success: true, message: 'Upload initialized', data: {
        driveId: drive._id, sessionId: drive.sessionId, uploads
    }});
};

/**
 * STEP 2 — POST /api/driver/upload/complete
 * The app calls this once it has PUT all 7 files to R2. We HEAD every object to confirm
 * it really arrived and that the video size is in range, then queue the drive for the GPU
 * worker. This is the authoritative validation gate (the app can't be trusted).
 *
 * Body: { sessionId }
 */
const completeUpload = async (req, res) => {
    const { sessionId } = req.body;
    if (!sessionId) return res.status(400).json({ success: false, message: 'sessionId is required', data: null });

    const drive = await Drive.findOne({ sessionId: safeName(sessionId), driverId: req.user._id });
    if (!drive) return res.status(404).json({ success: false, message: 'Drive not found', data: null });

    const keys = [drive.files.video, ...SENSOR_FILES.map(n => drive.files[FILE_KEYS[n]])];

    if (r2.isConfigured()) {
        // Confirm all 7 objects exist and the video size is in range.
        let videoSize = null;
        for (const key of keys) {
            const head = await r2.headObject(key);
            if (!head) {
                drive.status = 'rejected';
                drive.error  = 'Missing uploaded file: ' + key;
                await drive.save();
                await r2.deletePrefix(drive.sessionId + '/');
                return res.status(400).json({ success: false, message: 'Missing uploaded file: ' + key, data: null });
            }
            if (key === drive.files.video) videoSize = head.size;
        }
        if (videoSize < VIDEO_MIN_BYTES || videoSize > VIDEO_MAX_BYTES) {
            drive.status = 'rejected';
            drive.error  = 'Uploaded video size out of range: ' + videoSize;
            await drive.save();
            await r2.deletePrefix(drive.sessionId + '/');
            return res.status(400).json({ success: false, message: 'Uploaded video size out of range', data: null });
        }
    } else {
        // No R2 configured (local dev / tests): we can't HEAD, so trust the client.
        console.warn('[upload/complete] R2 not configured — skipping object validation');
    }

    drive.status = 'queued';
    drive.error  = null;
    await drive.save();
    res.status(202).json({ success: true, message: 'Drive queued for processing', data: { driveId: drive._id, status: drive.status } });
};

const getNotifications = async (req, res) => {
    const notifications = await Notification.find({ driverId: req.params.driverId }).sort({ createdAt: -1 });
    res.json({ success: true, message: 'Notifications fetched', data: notifications });
};

module.exports = { initUpload, completeUpload, getNotifications };
