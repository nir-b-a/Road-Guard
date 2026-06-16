const { spawn } = require('child_process');
const path = require('path');
const Drive = require('../models/Drive');
const Notification = require('../models/Notification');

const FPS = 30;

const uploadDrive = async (req, res) => {
    if (!req.file) return res.status(400).json({ success: false, message: 'No video file uploaded', data: null });

    const { sessionId, tags, speed_json } = req.body;
    if (!sessionId) return res.status(400).json({ success: false, message: 'sessionId is required', data: null });

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
        videoPath:    'uploads/videos/' + req.file.filename,
        tags:         parsedTags,
        speedSamples: parsedSpeedSamples,
        fps:          FPS,
        status:       'pending'
    });

    res.status(201).json({ success: true, message: 'Drive uploaded successfully', data: { sessionId: drive.sessionId, driveId: drive._id, speedSamplesCount: parsedSpeedSamples.length } });

    const mainPy = path.join(__dirname, '..', '..', 'main.py');
    const videoFullPath = path.join(__dirname, '..', 'uploads', 'videos', req.file.filename);
    const pythonBin = process.env.PYTHON_BIN || 'python';
    const brain = spawn(pythonBin, [mainPy, videoFullPath, drive._id.toString()], {
        stdio: ['ignore', 'pipe', 'pipe']
    });
    brain.stdout.on('data', d => console.log(`[brain ${drive._id}] ${d.toString().trim()}`));
    brain.stderr.on('data', d => console.error(`[brain ${drive._id}] ${d.toString().trim()}`));
    brain.on('close', code => console.log(`[brain ${drive._id}] finished (code ${code})`));
    brain.unref();
};

const getNotifications = async (req, res) => {
    const notifications = await Notification.find({ driverId: req.params.driverId }).sort({ createdAt: -1 });
    res.json({ success: true, message: 'Notifications fetched', data: notifications });
};

module.exports = { uploadDrive, getNotifications };
