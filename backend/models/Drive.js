const mongoose = require('mongoose');

const SpeedSampleSchema = new mongoose.Schema({
    frame:       { type: Number, required: true },
    timestampMs: { type: Number, required: true },
    speedKmh:    { type: Number, required: true },
    lat:         { type: Number, required: true },
    lon:         { type: Number, required: true }
}, { _id: false });

// R2 object key of every artifact captured for one session. The video is required;
// the 5 sensor CSVs + intrinsics.json complete the "7-file" contract enforced at upload.
const SessionFilesSchema = new mongoose.Schema({
    video:      { type: String },
    frames:     { type: String },
    gps:        { type: String },
    gravity:    { type: String },
    gyro:       { type: String },
    linacc:     { type: String },
    intrinsics: { type: String }
}, { _id: false });

// Status lifecycle:
//   created    -> /upload/init ran, presigned URLs handed out, app is PUTting to R2
//   queued     -> /upload/complete validated all 7 objects; waiting for a GPU worker
//   processing -> a worker claimed it (see claimedAt/workerId)
//   processed  -> pipeline finished, violations recorded, raw video purged from R2
//   failed     -> pipeline threw (see error); watchdog may requeue stuck 'processing'
//   rejected   -> validation failed (missing files / size out of range / over quota)
const DriveSchema = new mongoose.Schema({
    driverId:     { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    sessionId:    { type: String, required: true, unique: true },
    videoPath:    { type: String, required: true },   // kept for back-compat; == files.video (R2 key)
    files:        { type: SessionFilesSchema, default: {} },
    speedSamples: [SpeedSampleSchema],
    fps:          { type: Number, default: 30 },
    uploadedAt:   { type: Date, default: Date.now },
    status:       { type: String, enum: ['created', 'queued', 'processing', 'processed', 'failed', 'rejected'], default: 'created' },
    claimedAt:    { type: Date, default: null },   // when a worker claimed it (for the stuck-job watchdog)
    workerId:     { type: String, default: null }, // which worker is/was processing it
    error:        { type: String, default: null }  // failure / rejection reason
});

module.exports = mongoose.model('Drive', DriveSchema);
