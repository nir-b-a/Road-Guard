const Violation = require('../models/Violation');
const Notification = require('../models/Notification');
const Drive = require('../models/Drive');
const r2 = require('../services/r2');

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
    // Mirror the brain's lexicographic priority: tier asc (0=solid first), confidence desc within tier.
    const violations = await Violation.find(filter)
        .populate('driverId', 'name email')
        .sort({ tier: 1, confidence: -1, detectedAt: -1 });
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
    // Remove the violation's R2 evidence folder (clip + crops + plate) so dismissed
    // violations don't accumulate against the free-tier quota.
    if (r2.isConfigured() && violation.videoClipPath) {
        const folder = violation.videoClipPath.replace(/\/[^/]+$/, '/');
        r2.deletePrefix(folder).catch(e => console.warn('[r2] dismiss cleanup failed:', e.message));
    }
    res.json({ success: true, message: 'Violation dismissed', data: { status: violation.status } });
};

const getEvidence = async (req, res) => {
    const violation = await Violation.findById(req.params.id);
    if (!violation) return res.status(404).json({ success: false, message: 'Violation not found', data: null });

    // Mint short-lived presigned GET URLs so the browser can stream video/images directly
    // from R2 without needing an Authorization header (which <video>/<img> can't send).
    // URLs expire in 15 minutes — sufficient for a review session.
    const video_clip_url = r2.isConfigured() && violation.videoClipPath
        ? await r2.presignGet(violation.videoClipPath, { expiresIn: 900 })
        : null;
    const plate_url = r2.isConfigured() && violation.plateClipPath
        ? await r2.presignGet(violation.plateClipPath, { expiresIn: 900 })
        : null;

    // plate_text: only expose the carId string when it is a real plate (FastALPR read),
    // not the tracker-id fallback that starts with 'vehicle_'.
    const plate_text = (violation.carId && !violation.carId.startsWith('vehicle_'))
        ? violation.carId : null;

    res.json({ success: true, message: 'Evidence retrieved', data: {
        video_clip_url,
        plate_url,
        plate_text,
        car_id_recognition: violation.carId,
        calculated_speed:   violation.calculatedSpeed,
        location:           violation.location,
        detected_at:        violation.detectedAt,
        recorded_at:        violation.recordedAt,   // GPS epoch from clip (null for stubs)
        violation_type:     violation.violationType,
        tier:               violation.tier,
        confidence:         violation.confidence,
    }});
};

/** Stream a violation's evidence clip from R2 through the server. Supports HTTP Range so
 *  the dashboard's <video> element can seek without downloading the whole file first. */
const streamVideo = async (req, res) => {
    const violation = await Violation.findById(req.params.id);
    if (!violation) return res.status(404).json({ success: false, message: 'Violation not found', data: null });
    if (!r2.isConfigured()) return res.status(503).json({ success: false, message: 'R2 not configured', data: null });

    const rangeHeader = req.headers['range'] || null;
    let stream;
    try {
        stream = await r2.getObjectStream(violation.videoClipPath, rangeHeader);
    } catch (err) {
        console.error('[streamVideo] R2 error:', err.message);
        return res.status(502).json({ success: false, message: 'Storage error', data: null });
    }
    if (!stream) return res.status(404).json({ success: false, message: 'Clip not found in storage', data: null });

    res.setHeader('Content-Type', stream.contentType);
    res.setHeader('Accept-Ranges', 'bytes');
    if (stream.contentRange)  res.setHeader('Content-Range', stream.contentRange);
    if (stream.contentLength != null) res.setHeader('Content-Length', stream.contentLength);
    res.status(stream.statusCode);
    stream.body.pipe(res);
};

/** Stream a violation's plate crop from R2. Same Range-forwarding logic as streamVideo. */
const streamPlate = async (req, res) => {
    const violation = await Violation.findById(req.params.id);
    if (!violation || !violation.plateClipPath)
        return res.status(404).json({ success: false, message: 'Plate not found', data: null });
    if (!r2.isConfigured()) return res.status(503).json({ success: false, message: 'R2 not configured', data: null });

    let stream;
    try {
        stream = await r2.getObjectStream(violation.plateClipPath, null);
    } catch (err) {
        return res.status(502).json({ success: false, message: 'Storage error', data: null });
    }
    if (!stream) return res.status(404).json({ success: false, message: 'Plate not found in storage', data: null });

    res.setHeader('Content-Type', stream.contentType || 'image/png');
    if (stream.contentLength != null) res.setHeader('Content-Length', stream.contentLength);
    stream.body.pipe(res);
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

module.exports = { getViolations, verifyViolation, dismissViolation, getEvidence, streamVideo, streamPlate, searchViolations, getDrives };
