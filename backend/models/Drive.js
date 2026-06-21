const mongoose = require('mongoose');

const TagSchema = new mongoose.Schema({
    timestamp: { type: Number, required: true },
    lat:       { type: Number, required: true },
    lon:       { type: Number, required: true }
}, { _id: false });

const SpeedSampleSchema = new mongoose.Schema({
    frame:       { type: Number, required: true },
    timestampMs: { type: Number, required: true },
    speedKmh:    { type: Number, required: true },
    lat:         { type: Number, required: true },
    lon:         { type: Number, required: true }
}, { _id: false });

// Stored relative path (served under /uploads) of every artifact captured for one
// session. All optional except the video: a drive can be uploaded before some
// sensor stream exists, and intrinsics.json is only written when calibration is known.
const SessionFilesSchema = new mongoose.Schema({
    video:      { type: String },
    frames:     { type: String },
    gps:        { type: String },
    gravity:    { type: String },
    gyro:       { type: String },
    linacc:     { type: String },
    intrinsics: { type: String },
    tags:       { type: String }
}, { _id: false });

const DriveSchema = new mongoose.Schema({
    driverId:     { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    sessionId:    { type: String, required: true, unique: true },
    videoPath:    { type: String, required: true },   // kept for back-compat; == files.video
    files:        { type: SessionFilesSchema, default: {} },
    tags:         [TagSchema],
    speedSamples: [SpeedSampleSchema],
    fps:          { type: Number, default: 30 },
    uploadedAt:   { type: Date, default: Date.now },
    status:       { type: String, enum: ['pending', 'processed'], default: 'pending' }
});

module.exports = mongoose.model('Drive', DriveSchema);
