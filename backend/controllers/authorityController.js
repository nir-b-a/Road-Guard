const Violation = require('../models/Violation');
const Notification = require('../models/Notification');
const Drive = require('../models/Drive');

const getViolations = async (req, res) => {
    const { status, date, license_plate } = req.query;
    const filter = {};
    if (status) filter.status = status;
    if (license_plate) filter.carId = { $regex: license_plate, $options: 'i' };
    if (date) {
        const start = new Date(date);
        const end = new Date(date);
        end.setDate(end.getDate() + 1);
        filter.detectedAt = { $gte: start, $lt: end };
    }
    const violations = await Violation.find(filter).populate('driverId', 'name email').sort({ detectedAt: -1 });
    res.json({ success: true, message: 'Violations fetched', data: violations });
};

const verifyViolation = async (req, res) => {
    const violation = await Violation.findById(req.params.id);
    if (!violation) return res.status(404).json({ success: false, message: 'Violation not found', data: null });
    violation.status = 'verified';
    violation.reviewedBy = req.user._id;
    await violation.save();
    await Notification.create({ driverId: violation.driverId, type: 'violation_verified', message: 'Your report has been verified' });
    res.json({ success: true, message: 'Violation verified and driver notified', data: { status: violation.status } });
};

const dismissViolation = async (req, res) => {
    const violation = await Violation.findById(req.params.id);
    if (!violation) return res.status(404).json({ success: false, message: 'Violation not found', data: null });
    violation.status = 'dismissed';
    violation.reviewedBy = req.user._id;
    await violation.save();
    res.json({ success: true, message: 'Violation dismissed', data: { status: violation.status } });
};

const getEvidence = async (req, res) => {
    const violation = await Violation.findById(req.params.id);
    if (!violation) return res.status(404).json({ success: false, message: 'Violation not found', data: null });
    res.json({ success: true, message: 'Evidence retrieved', data: {
        video_clip_url: req.protocol + '://' + req.get('host') + '/' + violation.videoClipPath,
        car_id_recognition: violation.carId,
        calculated_speed: violation.calculatedSpeed,
        location: violation.location,
        detected_at: violation.detectedAt
    }});
};

const getDrives = async (req, res) => {
    const drives = await Drive.find({})
        .populate('driverId', 'name email')
        .sort({ createdAt: -1 });
    res.json({ success: true, data: drives });
};

const searchViolations = async (req, res) => {
    const { license_plate, status } = req.query;
    const filter = {};
    if (license_plate) filter.carId = { $regex: license_plate, $options: 'i' };
    if (status) filter.status = status;
    const results = await Violation.find(filter).populate('driverId', 'name email').sort({ detectedAt: -1 });
    res.json({ success: true, message: 'Found ' + results.length + ' result(s)', data: { results } });
};

module.exports = { getViolations, verifyViolation, dismissViolation, getEvidence, searchViolations, getDrives };
