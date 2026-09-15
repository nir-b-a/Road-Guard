const Violation = require('../models/Violation');
const Notification = require('../models/Notification');
const Drive = require('../models/Drive');
const r2 = require('../services/r2');

// Where the dashboard loads a stored R2 object from. R2 keeps the bucket private, so hand back a
// short-lived presigned GET URL (the bytes never pass through this server). Falls back to a
// plain path when R2 isn't configured (local dev / tests).
const objectUrl = async (req, key) => r2.isConfigured()
    ? r2.presignGet(key)
    : req.protocol + '://' + req.get('host') + '/' + key;

const getViolations = async (req, res) => {
    const { status, date, license_plate, reviewed_by } = req.query;
    const filter = {};
    if (status) filter.status = status;
    // reviewed_by=me -> only the violations this authority verified or dismissed. Every review
    // stamps reviewedBy, so this is each authority's own list; add status to split it.
    if (reviewed_by) {
        if (reviewed_by !== 'me') return res.status(400).json({ success: false, message: "reviewed_by must be 'me'", data: null });
        filter.reviewedBy = req.user._id;
    }
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
        video_clip_url: await objectUrl(req, violation.videoClipPath),
        plate_image_url: violation.plateImagePath ? await objectUrl(req, violation.plateImagePath) : null,
        car_id_recognition: violation.carId,
        calculated_speed: violation.calculatedSpeed,
        location: violation.location,
        detected_at: violation.detectedAt,
        // The detail page shows Verify/Dismiss only while this is 'pending'.
        status: violation.status
    }});
};

// A fresh URL for the plate picture alone. The dashboard asks for it when the button is
// clicked, so a page left open longer than the presign expiry still shows the picture.
const getPlateImage = async (req, res) => {
    const violation = await Violation.findById(req.params.id);
    if (!violation) return res.status(404).json({ success: false, message: 'Violation not found', data: null });
    if (!violation.plateImagePath) return res.status(404).json({ success: false, message: 'No plate picture for this violation', data: null });
    res.json({ success: true, message: 'Plate picture retrieved', data: {
        plate_image_url: await objectUrl(req, violation.plateImagePath),
        car_id_recognition: violation.carId
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

module.exports = { getViolations, verifyViolation, dismissViolation, getEvidence, getPlateImage, searchViolations, getDrives };
