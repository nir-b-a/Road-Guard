const request = require('supertest');
const app = require('../app');
const { connect, disconnect, clearCollections } = require('./setup');
const { INVITE_CODE } = require('./helpers');
beforeAll(async () => await connect());
afterAll(async () => await disconnect());
beforeEach(async () => await clearCollections());
describe('AUTH - POST /api/auth/register', () => {
    it('registers a driver successfully', async () => {
        const res = await request(app).post('/api/auth/register').send({ name: 'Tal', email: 'tal@test.com', password: 'password123', role: 'driver' });
        expect(res.statusCode).toBe(201);
        expect(res.body.success).toBe(true);
        expect(res.body.data.token).toBeDefined();
        expect(res.body.data.role).toBe('driver');
    });
    it('registers an authority user successfully', async () => {
        const res = await request(app).post('/api/auth/register').send({ name: 'Officer', email: 'officer@test.com', password: 'password123', role: 'authority', inviteCode: INVITE_CODE });
        expect(res.statusCode).toBe(201);
        expect(res.body.data.role).toBe('authority');
    });
    it('fails with missing fields', async () => {
        const res = await request(app).post('/api/auth/register').send({ email: 'noname@test.com' });
        expect(res.statusCode).toBe(400);
        expect(res.body.success).toBe(false);
    });
    it('fails with duplicate email', async () => {
        await request(app).post('/api/auth/register').send({ name: 'Tal', email: 'dup@test.com', password: 'password123' });
        const res = await request(app).post('/api/auth/register').send({ name: 'Tal2', email: 'dup@test.com', password: 'password123' });
        expect(res.statusCode).toBe(400);
        expect(res.body.message).toMatch(/already registered/i);
    });
});
describe('AUTH - POST /api/auth/login', () => {
    beforeEach(async () => {
        await request(app).post('/api/auth/register').send({ name: 'Tal', email: 'login@test.com', password: 'password123' });
    });
    it('logs in with correct credentials', async () => {
        const res = await request(app).post('/api/auth/login').send({ email: 'login@test.com', password: 'password123' });
        expect(res.statusCode).toBe(200);
        expect(res.body.data.token).toBeDefined();
    });
    it('fails with wrong password', async () => {
        const res = await request(app).post('/api/auth/login').send({ email: 'login@test.com', password: 'wrongpass' });
        expect(res.statusCode).toBe(401);
        expect(res.body.message).toMatch(/invalid credentials/i);
    });
    it('fails with unknown email', async () => {
        const res = await request(app).post('/api/auth/login').send({ email: 'nobody@test.com', password: 'password123' });
        expect(res.statusCode).toBe(401);
    });
});