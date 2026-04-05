const request = require('supertest');
const app = require('../app');
const { connect, disconnect, clearCollections } = require('./setup');
const { getAuthorityToken, getDriverToken, seedViolation } = require('./helpers');
const Notification = require('../models/Notification');
beforeAll(async () => await connect());
afterAll(async () => await disconnect());
beforeEach(async () => await clearCollections());
describe('AUTHORITY - GET /api/authority/violations', () => {
    it('returns all violations', async () => {
        const { token } = await getAuthorityToken();
        const { userId: driverId } = await getDriverToken();
        await seedViolation(driverId);
        const res = await request(app).get('/api/authority/violations').set('Authorization', 'Bearer ' + token);
        expect(res.statusCode).toBe(200);
        expect(res.body.data.length).toBe(1);
    });
    it('filters by status=pending', async () => {
        const { token } = await getAuthorityToken();
        const { userId: driverId } = await getDriverToken();
        await seedViolation(driverId);
        const res = await request(app).get('/api/authority/violations?status=pending').set('Authorization', 'Bearer ' + token);
        expect(res.body.data.every(v => v.status === 'pending')).toBe(true);
    });
    it('filters by license_plate', async () => {
        const { token } = await getAuthorityToken();
        const { userId: driverId } = await getDriverToken();
        await seedViolation(driverId);
        const res = await request(app).get('/api/authority/violations?license_plate=ABC').set('Authorization', 'Bearer ' + token);
        expect(res.body.data.length).toBe(1);
        expect(res.body.data[0].carId).toMatch(/ABC/);
    });
    it('rejects request with no token', async () => {
        const res = await request(app).get('/api/authority/violations');
        expect(res.statusCode).toBe(401);
    });
});
describe('AUTHORITY - verify / dismiss', () => {
    it('verifies violation and creates driver notification', async () => {
        const { token } = await getAuthorityToken();
        const { userId: driverId } = await getDriverToken();
        const { violation } = await seedViolation(driverId);
        const res = await request(app).post('/api/authority/violation/' + violation._id + '/verify').set('Authorization', 'Bearer ' + token);
        expect(res.statusCode).toBe(200);
        expect(res.body.data.status).toBe('verified');
        const notif = await Notification.findOne({ driverId });
        expect(notif).not.toBeNull();
        expect(notif.type).toBe('violation_verified');
    });
    it('dismisses violation', async () => {
        const { token } = await getAuthorityToken();
        const { userId: driverId } = await getDriverToken();
        const { violation } = await seedViolation(driverId);
        const res = await request(app).post('/api/authority/violation/' + violation._id + '/dismiss').set('Authorization', 'Bearer ' + token);
        expect(res.statusCode).toBe(200);
        expect(res.body.data.status).toBe('dismissed');
    });
    it('returns 404 for non-existent violation', async () => {
        const { token } = await getAuthorityToken();
        const res = await request(app).post('/api/authority/violation/64b2f1a2e4b0a1b2c3d4e5f6/verify').set('Authorization', 'Bearer ' + token);
        expect(res.statusCode).toBe(404);
        expect(res.body.message).toMatch(/not found/i);
    });
});
describe('AUTHORITY - evidence & search', () => {
    it('returns complete evidence package', async () => {
        const { token } = await getAuthorityToken();
        const { userId: driverId } = await getDriverToken();
        const { violation } = await seedViolation(driverId);
        const res = await request(app).get('/api/authority/evidence/' + violation._id).set('Authorization', 'Bearer ' + token);
        expect(res.statusCode).toBe(200);
        expect(res.body.data.video_clip_url).toBeDefined();
        expect(res.body.data.car_id_recognition).toBeDefined();
        expect(res.body.data.calculated_speed).toBeDefined();
    });
    it('searches by license_plate and status after verify', async () => {
        const { token } = await getAuthorityToken();
        const { userId: driverId } = await getDriverToken();
        const { violation } = await seedViolation(driverId);
        await request(app).post('/api/authority/violation/' + violation._id + '/verify').set('Authorization', 'Bearer ' + token);
        const res = await request(app).get('/api/authority/search?license_plate=ABC&status=verified').set('Authorization', 'Bearer ' + token);
        expect(res.statusCode).toBe(200);
        expect(res.body.data.results.length).toBeGreaterThan(0);
        expect(res.body.data.results[0].status).toBe('verified');
    });
});