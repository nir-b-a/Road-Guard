const request = require('supertest');
const app = require('../app');
const { connect, disconnect, clearCollections } = require('./setup');
const { getDriverToken, getAuthorityToken } = require('./helpers');
const Drive = require('../models/Drive');
const Violation = require('../models/Violation');
beforeAll(async () => await connect());
afterAll(async () => await disconnect());
beforeEach(async () => await clearCollections());

const createDrive = async (driverId, status = 'processing') =>
    Drive.create({ driverId, sessionId: 'session_' + Date.now() + '_' + Math.random(), videoPath: 'sessions/s/video.mp4', status });

describe('CV UNIT - POST /api/internal/violation', () => {
    it('creates a violation without flipping the drive status', async () => {
        const { userId: driverId } = await getDriverToken();
        const drive = await createDrive(driverId);
        const res = await request(app).post('/api/internal/violation').send({ driveId: drive._id, videoClipPath: 'sessions/s/out/clip.mp4', carId: 'XYZ-9999', calculatedSpeed: 110, lat: 32.08, lon: 34.78 });
        expect(res.statusCode).toBe(201);
        expect(res.body.data.violationId).toBeDefined();
        const updated = await Drive.findById(drive._id);
        expect(updated.status).toBe('processing');   // /violation does not complete the drive
    });
    it('stores the plate picture key sent with the violation', async () => {
        const { userId: driverId } = await getDriverToken();
        const drive = await createDrive(driverId);
        const res = await request(app).post('/api/internal/violation').send({ driveId: drive._id, videoClipPath: 'sessions/s/out/clip.mp4', plateImagePath: 'sessions/s/out/clip_plate.png', carId: 'XYZ-9999', calculatedSpeed: 110, lat: 32.08, lon: 34.78 });
        expect(res.statusCode).toBe(201);
        expect((await Violation.findById(res.body.data.violationId)).plateImagePath).toBe('sessions/s/out/clip_plate.png');
    });
    it('stores no plate picture when the worker found none', async () => {
        const { userId: driverId } = await getDriverToken();
        const drive = await createDrive(driverId);
        const res = await request(app).post('/api/internal/violation').send({ driveId: drive._id, videoClipPath: 'sessions/s/out/clip.mp4', plateImagePath: null, carId: 'vehicle-7', calculatedSpeed: 0, lat: 0, lon: 0 });
        expect(res.statusCode).toBe(201);
        expect((await Violation.findById(res.body.data.violationId)).plateImagePath).toBeNull();
    });
    it('fails with missing carId', async () => {
        const { userId: driverId } = await getDriverToken();
        const drive = await createDrive(driverId);
        const res = await request(app).post('/api/internal/violation').send({ driveId: drive._id, videoClipPath: 'sessions/s/out/clip.mp4', calculatedSpeed: 110, lat: 32.08, lon: 34.78 });
        expect(res.statusCode).toBe(400);
        expect(res.body.message).toMatch(/missing required fields/i);
    });
    it('fails with invalid driveId', async () => {
        const res = await request(app).post('/api/internal/violation').send({ driveId: '64b2f1a2e4b0a1b2c3d4e5f6', videoClipPath: 'sessions/s/out/clip.mp4', carId: 'XYZ-9999', calculatedSpeed: 110, lat: 32.08, lon: 34.78 });
        expect(res.statusCode).toBe(404);
        expect(res.body.message).toMatch(/drive not found/i);
    });
    it('evidence package is complete after violation created', async () => {
        const { userId: driverId } = await getDriverToken();
        const { token: authToken } = await getAuthorityToken();
        const drive = await createDrive(driverId);
        const cvRes = await request(app).post('/api/internal/violation').send({ driveId: drive._id, videoClipPath: 'sessions/s/out/clip.mp4', carId: 'US9-TEST', calculatedSpeed: 130, lat: 32.08, lon: 34.78 });
        const evidenceRes = await request(app).get('/api/authority/evidence/' + cvRes.body.data.violationId).set('Authorization', 'Bearer ' + authToken);
        expect(evidenceRes.statusCode).toBe(200);
        expect(evidenceRes.body.data.video_clip_url).toBeDefined();
        expect(evidenceRes.body.data.car_id_recognition).toBe('US9-TEST');
        expect(evidenceRes.body.data.calculated_speed).toBe(130);
    });
});

describe('CV UNIT - GET /api/internal/next-job', () => {
    it('atomically claims the oldest queued drive', async () => {
        const { userId: driverId } = await getDriverToken();
        const drive = await createDrive(driverId, 'queued');
        const res = await request(app).get('/api/internal/next-job').set('x-worker-id', 'w1');
        expect(res.statusCode).toBe(200);
        expect(String(res.body.data.driveId)).toBe(String(drive._id));
        const updated = await Drive.findById(drive._id);
        expect(updated.status).toBe('processing');
        expect(updated.workerId).toBe('w1');
        // A second poll finds nothing left to claim.
        const res2 = await request(app).get('/api/internal/next-job');
        expect(res2.body.data).toBeNull();
    });
});

describe('CV UNIT - POST /api/internal/drive/:id/complete', () => {
    it('marks a drive processed', async () => {
        const { userId: driverId } = await getDriverToken();
        const drive = await createDrive(driverId);
        const res = await request(app).post('/api/internal/drive/' + drive._id + '/complete').send({ status: 'processed' });
        expect(res.statusCode).toBe(200);
        expect((await Drive.findById(drive._id)).status).toBe('processed');
    });
    it('marks a drive failed with an error message', async () => {
        const { userId: driverId } = await getDriverToken();
        const drive = await createDrive(driverId);
        const res = await request(app).post('/api/internal/drive/' + drive._id + '/complete').send({ status: 'failed', error: 'boom' });
        expect(res.statusCode).toBe(200);
        const updated = await Drive.findById(drive._id);
        expect(updated.status).toBe('failed');
        expect(updated.error).toBe('boom');
    });
    it('rejects an invalid status', async () => {
        const { userId: driverId } = await getDriverToken();
        const drive = await createDrive(driverId);
        const res = await request(app).post('/api/internal/drive/' + drive._id + '/complete').send({ status: 'banana' });
        expect(res.statusCode).toBe(400);
    });
});

describe('CV UNIT - POST /api/internal/drive/:id/release', () => {
    const claimedBy = async (driverId, workerId) => {
        const drive = await createDrive(driverId, 'queued');
        await request(app).get('/api/internal/next-job').set('x-worker-id', workerId);
        return Drive.findById(drive._id);
    };
    it('hands a drive back to the queue so the next poll can claim it again', async () => {
        const { userId: driverId } = await getDriverToken();
        const drive = await claimedBy(driverId, 'w1');
        expect(drive.status).toBe('processing');
        const res = await request(app).post('/api/internal/drive/' + drive._id + '/release').set('x-worker-id', 'w1').send({ reason: 'stop now' });
        expect(res.statusCode).toBe(200);
        const updated = await Drive.findById(drive._id);
        expect(updated.status).toBe('queued');
        expect(updated.workerId).toBeNull();
        expect(updated.claimedAt).toBeNull();
        const again = await request(app).get('/api/internal/next-job').set('x-worker-id', 'w2');
        expect(String(again.body.data.driveId)).toBe(String(drive._id));
    });
    it('refuses to release a drive another worker holds', async () => {
        const { userId: driverId } = await getDriverToken();
        const drive = await claimedBy(driverId, 'w1');
        const res = await request(app).post('/api/internal/drive/' + drive._id + '/release').set('x-worker-id', 'w2').send({});
        expect(res.statusCode).toBe(409);
        expect((await Drive.findById(drive._id)).status).toBe('processing');
    });
    it('refuses to release a drive that is no longer processing', async () => {
        const { userId: driverId } = await getDriverToken();
        const drive = await createDrive(driverId, 'processed');
        const res = await request(app).post('/api/internal/drive/' + drive._id + '/release').set('x-worker-id', 'worker').send({});
        expect(res.statusCode).toBe(409);
        expect((await Drive.findById(drive._id)).status).toBe('processed');
    });
    it('returns 404 for an unknown drive', async () => {
        const res = await request(app).post('/api/internal/drive/64b2f1a2e4b0a1b2c3d4e5f6/release').send({});
        expect(res.statusCode).toBe(404);
    });
});
