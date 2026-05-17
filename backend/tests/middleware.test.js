const request = require('supertest');
const app = require('../app');
const { connect, disconnect, clearCollections } = require('./setup');
const { getDriverToken, getAuthorityToken } = require('./helpers');
beforeAll(async () => await connect());
afterAll(async () => await disconnect());
beforeEach(async () => await clearCollections());
describe('MIDDLEWARE - Auth Guard', () => {
    it('rejects request with no token', async () => {
        const res = await request(app).get('/api/authority/violations');
        expect(res.statusCode).toBe(401);
        expect(res.body.message).toMatch(/no token/i);
    });
    it('rejects request with fake token', async () => {
        const res = await request(app).get('/api/authority/violations').set('Authorization', 'Bearer this.is.fake');
        expect(res.statusCode).toBe(401);
        expect(res.body.message).toMatch(/invalid or expired/i);
    });
});
describe('MIDDLEWARE - Role Guard', () => {
    it('driver token cannot access authority route', async () => {
        const { token } = await getDriverToken();
        const res = await request(app).get('/api/authority/violations').set('Authorization', 'Bearer ' + token);
        expect(res.statusCode).toBe(403);
    });
    it('authority token cannot access driver upload route', async () => {
        const { token } = await getAuthorityToken();
        const res = await request(app).post('/api/driver/upload').set('Authorization', 'Bearer ' + token);
        expect(res.statusCode).toBe(403);
    });
    it('driver token passes driver route guard', async () => {
        const { token, userId } = await getDriverToken();
        const res = await request(app).get('/api/notifications/inbox/' + userId).set('Authorization', 'Bearer ' + token);
        expect(res.statusCode).toBe(200);
    });
});