const mongoose = require('mongoose');
const ViolationSchema = new mongoose.Schema({
    driveId: { type: mongoose.Schema.Types.ObjectId, ref: 'Drive', required: true },
    driverId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    videoClipPath: { type: String, required: true },
    // R2 key of the clearest picture of the offending car's plate; null when no plate was found.
    plateImagePath: { type: String, default: null },
    carId: { type: String, required: true },
    calculatedSpeed: { type: Number, required: true },
    location: { lat: { type: Number, required: true }, lon: { type: Number, required: true } },
    detectedAt: { type: Date, default: Date.now },
    violationType: { type: String, enum: ['speeding', 'lane_crossing'], default: 'speeding' },
    status: { type: String, enum: ['pending', 'verified', 'dismissed'], default: 'pending' },
    reviewedBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User', default: null }
});
module.exports = mongoose.model('Violation', ViolationSchema);