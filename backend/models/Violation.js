const mongoose = require('mongoose');
const ViolationSchema = new mongoose.Schema({
    driveId: { type: mongoose.Schema.Types.ObjectId, ref: 'Drive', required: true },
    driverId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    videoClipPath: { type: String, required: true },
    plateClipPath: { type: String, default: null },   // output/<session>/<vid>/plate.png, when present
    carId: { type: String, required: true },
    calculatedSpeed: { type: Number, default: 0 },
    location: { lat: { type: Number, default: 0 }, lon: { type: Number, default: 0 } },
    // Brain priority fields (see violations/export.py priority_model):
    violationType: { type: String, default: null },   // e.g. SOLID_LINE_CROSSING, SPEEDING
    tier: { type: Number, default: 2 },               // 0=solid > 1=speeding > 2=other > 3=yellow
    confidence: { type: Number, default: null },      // detector_confidence in [0,1]
    detectedAt: { type: Date, default: Date.now },
    // GPS-derived timestamp from the recording device (ISO-8601 from gps.csv).
    // null when the clip used synthetic/stub sensor data without real epoch timestamps.
    // Populated once friends wire real Android GPS into the upload flow.
    recordedAt: { type: Date, default: null },
    status: { type: String, enum: ['pending', 'verified', 'dismissed'], default: 'pending' },
    reviewedBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User', default: null }
});
module.exports = mongoose.model('Violation', ViolationSchema);