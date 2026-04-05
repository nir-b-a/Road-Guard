const Drive = require('../models/Drive');
const Notification = require('../models/Notification');
const uploadDrive = async (req, res) => {
    if (!req.file) return res.status(400).json({ success: false, message: 'No video file uploaded', data: null });
    const { sessionId, tags } = req.body;
    if (!sessionId) return res.status(400).json({ success: false, message: 'sessionId is required', data: null });
    let parsedTags = [];
    try { parsedTags = tags ? JSON.parse(tags) : []; } catch { parsedTags = []; }
    const drive = await Drive.create({ driverId: req.user._id, sessionId, videoPath: 'uploads/videos/' + req.file.filename, tags: parsedTags, status: 'pending' });
    res.status(201).json({ success: true, message: 'Drive uploaded successfully', data: { sessionId: drive.sessionId, driveId: drive._id } });
};
const getNotifications = async (req, res) => {
    const notifications = await Notification.find({ driverId: req.params.driverId }).sort({ createdAt: -1 });
    res.json({ success: true, message: 'Notifications fetched', data: notifications });
};
module.exports = { uploadDrive, getNotifications };