const mongoose = require('mongoose');
const ViolationSchema = new mongoose.Schema({
    driveId: { type: mongoose.Schema.Types.ObjectId, ref: 'Drive', required: true },
    driverId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    videoClipPath: { type: String, required: true },
    carId: { type: String, required: true },
    calculatedSpeed: { type: Number, required: true },
    location: { lat: { type: Number, required: true }, lon: { type: Number, required: true } },
    detectedAt: { type: Date, default: Date.now },
    status: { type: String, enum: ['pending', 'verified', 'dismissed'], default: 'pending' },
    reviewedBy: { type: mongoose.Schema.Types.ObjectId, ref: 'User', default: null }
});
module.exports = mongoose.model('Violation', ViolationSchema);