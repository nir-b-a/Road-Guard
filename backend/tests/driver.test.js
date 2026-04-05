const request = require('supertest');
const app = require('../app');
const { connect, disconnect, clearCollections } = require('./setup');
const { getDriverToken, getAuthorityToken } = require('./helpers');
const Notification = require('../models/Notification');
beforeAll(async () => await connect());
afterAll(async () => await disconnect());
beforeEach(async () => await clearCollections());
const fakeVideoBuffer = Buffer.from('fake-video-content');
describe('DRIVER - POST /api/driver/upload', () => {
    it('uploads drive successfully', async () => {
        const { token } = await getDriverToken();
        const res = await request(app).post('/api/driver/upload').set('Authorization', 'Bearer ' + token)
            .attach('video', fakeVideoBuffer, 'test.mp4').field('sessionId', 'session_' + Date.now()).field('tags', JSON.stringify([{ timestamp: Date.now(), lat: 32.08, lon: 34.78 }]));
        expect(res.statusCode).toBe(201);
        expect(res.body.data.sessionId).toBeDefined();
        expect(res.body.data.driveId).toBeDefined();
    });
    it('fails with no video file', async () => {
        const { token } = await getDriverToken();
        const res = await request(app).post('/api/driver/upload').set('Authorization', 'Bearer ' + token).field('sessionId', 'session_test_1');
        expect(res.statusCode).toBe(400);
        expect(res.body.message).toMatch(/no video file/i);
    });
    it('fails with no sessionId', async () => {
        const { token } = await getDriverToken();
        const res = await request(app).post('/api/driver/upload').set('Authorization', 'Bearer ' + token).attach('video', fakeVideoBuffer, 'test.mp4');
        expect(res.statusCode).toBe(400);
        expect(res.body.message).toMatch(/sessionId is required/i);
    });
    it('fails when authority tries to upload', async () => {
        const { token } = await getAuthorityToken();
        const res = await request(app).post('/api/driver/upload').set('Authorization', 'Bearer ' + token).attach('video', fakeVideoBuffer, 'test.mp4').field('sessionId', 'session_auth_test');
        expect(res.statusCode).toBe(403);
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