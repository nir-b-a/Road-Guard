const request = require('supertest');
const app = require('../app');
const { connect, disconnect, clearCollections } = require('./setup');
const { getDriverToken, getAuthorityToken } = require('./helpers');
const Drive = require('../models/Drive');
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
