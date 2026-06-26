const express = require('express');
const router = express.Router();
const Drive = require('../models/Drive');
const Violation = require('../models/Violation');
const r2 = require('../services/r2');
const { internalAuth } = require('../middleware/internalAuth');

// A 'processing' job older than this is assumed dead (worker crashed) and requeued.
const JOB_STALE_MS = parseInt(process.env.JOB_STALE_MINUTES || '30', 10) * 60 * 1000;

router.use(internalAuth);

/**
 * GET /api/internal/next-job
 * The GPU worker's poll. First requeues any stuck 'processing' jobs (watchdog), then
 * atomically claims the oldest 'queued' drive so two workers can't grab the same one.
 * Returns the drive's R2 keys (the worker downloads them with its own R2 credentials),
 * or data:null when the queue is empty.
 */
router.get('/next-job', async (req, res) => {
    const workerId = req.headers['x-worker-id'] || 'worker';

    // Watchdog: reclaim jobs whose worker died mid-processing.
    await Drive.updateMany(
        { status: 'processing', claimedAt: { $lt: new Date(Date.now() - JOB_STALE_MS) } },
        { $set: { status: 'queued', claimedAt: null, workerId: null } }
    );

    const drive = await Drive.findOneAndUpdate(
        { status: 'queued' },
        { $set: { status: 'processing', claimedAt: new Date(), workerId } },
        { sort: { uploadedAt: 1 }, new: true }
    );

    if (!drive) return res.json({ success: true, message: 'No jobs', data: null });
    res.json({ success: true, message: 'Job claimed', data: {
        driveId: drive._id, sessionId: drive.sessionId, fps: drive.fps, files: drive.files
    }});
});

/**
 * POST /api/internal/drive/:id/complete   Body: { status: 'processed'|'failed', error? }
 * The worker marks a drive done. On success we purge the raw inputs (video + sensor CSVs)
 * from R2 to stay under the free-tier quota — the evidence clips under .../out/ are kept.
 */
router.post('/drive/:id/complete', async (req, res) => {
    const { status, error } = req.body;
    if (!['processed', 'failed'].includes(status)) {
        return res.status(400).json({ success: false, message: "status must be 'processed' or 'failed'", data: null });
    }
    const drive = await Drive.findById(req.params.id);
    if (!drive) return res.status(404).json({ success: false, message: 'Drive not found', data: null });

    drive.status = status;
    drive.error  = status === 'failed' ? (error || 'unknown error') : null;
    await drive.save();

    if (status === 'processed') {
        // Retention: delete the big raw inputs, keep the small evidence outputs.
        const rawKeys = [drive.files.video, drive.files.frames, drive.files.gps, drive.files.gravity,
                         drive.files.gyro, drive.files.linacc, drive.files.intrinsics].filter(Boolean);
        for (const key of rawKeys) {
            try { await r2.deleteObject(key); } catch (e) { console.warn('[retention] failed to delete ' + key + ': ' + e.message); }
        }
    }
    res.json({ success: true, message: 'Drive ' + status, data: { driveId: drive._id, status } });
});

/**
 * POST /api/internal/violation
 * One detected violation. videoClipPath is now the R2 key of the evidence clip the worker
 * uploaded (under <sessionId>/out/). Recording a violation does NOT flip the drive to
 * processed — the worker does that via /drive/:id/complete after all violations are in.
 */
router.post('/violation', async (req, res) => {
    const { driveId, videoClipPath, carId, calculatedSpeed, lat, lon } = req.body;
    if (!driveId || !videoClipPath || !carId || calculatedSpeed == null || lat == null || lon == null) {
        return res.status(400).json({ success: false, message: 'Missing required fields', data: null });
    }
    const drive = await Drive.findById(driveId);
    if (!drive) return res.status(404).json({ success: false, message: 'Drive not found', data: null });
    const violation = await Violation.create({ driveId, driverId: drive.driverId, videoClipPath, carId, calculatedSpeed, location: { lat, lon } });
    res.status(201).json({ success: true, message: 'Violation recorded', data: { violationId: violation._id } });
});

module.exports = router;
