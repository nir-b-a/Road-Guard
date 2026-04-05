const mongoose = require('mongoose');
const TagSchema = new mongoose.Schema({
    timestamp: { type: Number, required: true },
    lat: { type: Number, required: true },
    lon: { type: Number, required: true }
}, { _id: false });
const DriveSchema = new mongoose.Schema({
    driverId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    sessionId: { type: String, required: true, unique: true },
    videoPath: { type: String, required: true },
    tags: [TagSchema],
    uploadedAt: { type: Date, default: Date.now },
    status: { type: String, enum: ['pending', 'processed'], default: 'pending' }
});
module.exports = mongoose.model('Drive', DriveSchema);