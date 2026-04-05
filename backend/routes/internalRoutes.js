const express = require('express');
const router = express.Router();
const Drive = require('../models/Drive');
const Violation = require('../models/Violation');
router.post('/violation', async (req, res) => {
    const { driveId, videoClipPath, carId, calculatedSpeed, lat, lon } = req.body;
    if (!driveId || !videoClipPath || !carId || !calculatedSpeed || !lat || !lon) {
        return res.status(400).json({ success: false, message: 'Missing required fields', data: null });
    }
    const drive = await Drive.findById(driveId);
    if (!drive) return res.status(404).json({ success: false, message: 'Drive not found', data: null });
    const violation = await Violation.create({ driveId, driverId: drive.driverId, videoClipPath, carId, calculatedSpeed, location: { lat, lon } });
    drive.status = 'processed';
    await drive.save();
    res.status(201).json({ success: true, message: 'Violation recorded', data: { violationId: violation._id } });
});
module.exports = router;