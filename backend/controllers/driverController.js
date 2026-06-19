const Drive = require('../models/Drive');
const Notification = require('../models/Notification');

const FPS = 30;

// Upload field name (== on-device file name) -> Drive.files key.
const FILE_KEYS = {
    'video':           'video',
    'frames.csv':      'frames',
    'gps.csv':         'gps',
    'gravity.csv':     'gravity',
    'gyro.csv':        'gyro',
    'linacc.csv':      'linacc',
    'intrinsics.json': 'intrinsics',
    'tags.json':       'tags'
};

const uploadDrive = async (req, res) => {
    // upload.fields() -> req.files is { fieldName: [file], ... }; the video is required.
    const files = req.files || {};
    const videoFile = files.video && files.video[0];
    if (!videoFile) return res.status(400).json({ success: false, message: 'No video file uploaded', data: null });

    const { sessionId, tags, speed_json } = req.body;
    if (!sessionId) return res.status(400).json({ success: false, message: 'sessionId is required', data: null });

    // Map every received artifact to the relative path it is served from (/uploads/...).
    const storedFiles = {};
    for (const [field, list] of Object.entries(files)) {
        const key = FILE_KEYS[field];
        const f = list && list[0];
        if (key && f) storedFiles[key] = 'uploads/sessions/' + req._sessionId + '/' + f.filename;
    }

    let parsedTags = [];
    try { parsedTags = tags ? JSON.parse(tags) : []; } catch { parsedTags = []; }

    let parsedSpeedSamples = [];
    try {
        const raw = speed_json ? JSON.parse(speed_json) : [];
        parsedSpeedSamples = raw.map(s => ({
            frame:       Math.round((s.timestampMs / 1000) * FPS),
            timestampMs: s.timestampMs,
            speedKmh:    s.speedKmh,
            lat:         s.lat,
            lon:         s.lon
        }));
    } catch { parsedSpeedSamples = []; }

    const drive = await Drive.create({
        driverId:     req.user._id,
        sessionId,
        videoPath:    storedFiles.video,   // back-compat alias of files.video
        files:        storedFiles,
        tags:         parsedTags,
        speedSamples: parsedSpeedSamples,
        fps:          FPS,
        status:       'pending'
    });

    res.status(201).json({ success: true, message: 'Drive uploaded successfully', data: { sessionId: drive.sessionId, driveId: drive._id, files: Object.keys(storedFiles), speedSamplesCount: parsedSpeedSamples.length } });
};

const getNotifications = async (req, res) => {
    const notifications = await Notification.find({ driverId: req.params.driverId }).sort({ createdAt: -1 });
    res.json({ success: true, message: 'Notifications fetched', data: notifications });
};

module.exports = { uploadDrive, getNotifications };
