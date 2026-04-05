const request = require('supertest');
const app = require('../app');
const { connect, disconnect, clearCollections } = require('./setup');
const { getDriverToken, getAuthorityToken } = require('./helpers');
const Drive = require('../models/Drive');
beforeAll(async () => await connect());
afterAll(async () => await disconnect());
beforeEach(async () => await clearCollections());
const createDrive = async (driverId) => Drive.create({ driverId, sessionId: 'session_' + Date.now(), videoPath: 'uploads/videos/test.mp4', tags: [], status: 'pending' });
describe('CV UNIT - POST /api/internal/violation', () => {
    it('creates violation and marks drive as processed', async () => {
        const { userId: driverId } = await getDriverToken();
        const drive = await createDrive(driverId);
        const res = await request(app).post('/api/internal/violation').send({ driveId: drive._id, videoClipPath: 'uploads/videos/clip.mp4', carId: 'XYZ-9999', calculatedSpeed: 110, lat: 32.08, lon: 34.78 });
        expect(res.statusCode).toBe(201);
        expect(res.body.data.violationId).toBeDefined();
        const updated = await Drive.findById(drive._id);
        expect(updated.status).toBe('processed');
    });
    it('fails with missing carId', async () => {
        const { userId: driverId } = await getDriverToken();
        const drive = await createDrive(driverId);
        const res = await request(app).post('/api/internal/violation').send({ driveId: drive._id, videoClipPath: 'uploads/videos/clip.mp4', calculatedSpeed: 110, lat: 32.08, lon: 34.78 });
        expect(res.statusCode).toBe(400);
        expect(res.body.message).toMatch(/missing required fields/i);
    });
    it('fails with invalid driveId', async () => {
        const res = await request(app).post('/api/internal/violation').send({ driveId: '64b2f1a2e4b0a1b2c3d4e5f6', videoClipPath: 'uploads/videos/clip.mp4', carId: 'XYZ-9999', calculatedSpeed: 110, lat: 32.08, lon: 34.78 });
        expect(res.statusCode).toBe(404);
        expect(res.body.message).toMatch(/drive not found/i);
    });
    it('evidence package is complete after violation created', async () => {
        const { userId: driverId } = await getDriverToken();
        const { token: authToken } = await getAuthorityToken();
        const drive = await createDrive(driverId);
        const cvRes = await request(app).post('/api/internal/violation').send({ driveId: drive._id, videoClipPath: 'uploads/videos/clip.mp4', carId: 'US9-TEST', calculatedSpeed: 130, lat: 32.08, lon: 34.78 });
        const evidenceRes = await request(app).get('/api/authority/evidence/' + cvRes.body.data.violationId).set('Authorization', 'Bearer ' + authToken);
        expect(evidenceRes.statusCode).toBe(200);
        expect(evidenceRes.body.data.video_clip_url).toBeDefined();
        expect(evidenceRes.body.data.car_id_recognition).toBe('US9-TEST');
        expect(evidenceRes.body.data.calculated_speed).toBe(130);
    });
});