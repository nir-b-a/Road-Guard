const request = require('supertest');
const app = require('../app');
const { connect, disconnect, clearCollections } = require('./setup');
const { getDriverToken, getAuthorityToken } = require('./helpers');
const Notification = require('../models/Notification');
const Drive = require('../models/Drive');
beforeAll(async () => await connect());
afterAll(async () => await disconnect());
beforeEach(async () => await clearCollections());

const VALID_SIZE = 200 * 1024 * 1024;   // 200 MB, inside [100 MB, 4 GB]

describe('DRIVER - POST /api/driver/upload/init', () => {
    it('initializes upload and returns 7 presigned URLs', async () => {
        const { token } = await getDriverToken();
        const res = await request(app).post('/api/driver/upload/init').set('Authorization', 'Bearer ' + token)
            .send({ sessionId: 'session_' + Date.now(), videoName: 'drive.mp4', videoSize: VALID_SIZE });
        expect(res.statusCode).toBe(201);
        expect(res.body.data.driveId).toBeDefined();
        expect(Object.keys(res.body.data.uploads).sort()).toEqual(
            ['drive.mp4', 'frames.csv', 'gps.csv', 'gravity.csv', 'gyro.csv', 'intrinsics.json', 'linacc.csv'].sort());
    });
    it('fails with no sessionId', async () => {
        const { token } = await getDriverToken();
        const res = await request(app).post('/api/driver/upload/init').set('Authorization', 'Bearer ' + token)
            .send({ videoName: 'drive.mp4', videoSize: VALID_SIZE });
        expect(res.statusCode).toBe(400);
        expect(res.body.message).toMatch(/sessionId is required/i);
    });
    it('fails with no videoName', async () => {
        const { token } = await getDriverToken();
        const res = await request(app).post('/api/driver/upload/init').set('Authorization', 'Bearer ' + token)
            .send({ sessionId: 'session_x', videoSize: VALID_SIZE });
        expect(res.statusCode).toBe(400);
        expect(res.body.message).toMatch(/videoName is required/i);
    });
    it('rejects a video below the size floor', async () => {
        const { token } = await getDriverToken();
        const res = await request(app).post('/api/driver/upload/init').set('Authorization', 'Bearer ' + token)
            .send({ sessionId: 'session_small', videoName: 'drive.mp4', videoSize: 1024 });
        expect(res.statusCode).toBe(400);
        expect(res.body.message).toMatch(/size must be between/i);
    });
    it('rejects a non-video extension', async () => {
        const { token } = await getDriverToken();
        const res = await request(app).post('/api/driver/upload/init').set('Authorization', 'Bearer ' + token)
            .send({ sessionId: 'session_ext', videoName: 'drive.txt', videoSize: VALID_SIZE });
        expect(res.statusCode).toBe(400);
    });
    it('fails when authority tries to init', async () => {
        const { token } = await getAuthorityToken();
        const res = await request(app).post('/api/driver/upload/init').set('Authorization', 'Bearer ' + token)
            .send({ sessionId: 'session_auth', videoName: 'drive.mp4', videoSize: VALID_SIZE });
        expect(res.statusCode).toBe(403);
    });
});

describe('DRIVER - POST /api/driver/upload/complete', () => {
    it('queues the drive after init (R2 validation skipped in test)', async () => {
        const { token } = await getDriverToken();
        const sessionId = 'session_' + Date.now();
        await request(app).post('/api/driver/upload/init').set('Authorization', 'Bearer ' + token)
            .send({ sessionId, videoName: 'drive.mp4', videoSize: VALID_SIZE });
        const res = await request(app).post('/api/driver/upload/complete').set('Authorization', 'Bearer ' + token)
            .send({ sessionId });
        expect(res.statusCode).toBe(202);
        expect(res.body.data.status).toBe('queued');
        const drive = await Drive.findOne({ sessionId });
        expect(drive.status).toBe('queued');
    });
    it('404 when completing an unknown session', async () => {
        const { token } = await getDriverToken();
        const res = await request(app).post('/api/driver/upload/complete').set('Authorization', 'Bearer ' + token)
            .send({ sessionId: 'nope_' + Date.now() });
        expect(res.statusCode).toBe(404);
    });
});

describe('DRIVER - GET /api/notifications/inbox/:driverId', () => {
    it('returns empty array when no notifications', async () => {
        const { token, userId } = await getDriverToken();
        const res = await request(app).get('/api/notifications/inbox/' + userId).set('Authorization', 'Bearer ' + token);
        expect(res.statusCode).toBe(200);
        expect(res.body.data).toEqual([]);
    });
    it('returns notifications after one is seeded', async () => {
        const { token, userId } = await getDriverToken();
        await Notification.create({ driverId: userId, type: 'violation_verified', message: 'Your report has been verified' });
        const res = await request(app).get('/api/notifications/inbox/' + userId).set('Authorization', 'Bearer ' + token);
        expect(res.statusCode).toBe(200);
        expect(res.body.data.length).toBe(1);
        expect(res.body.data[0].type).toBe('violation_verified');
    });
});
