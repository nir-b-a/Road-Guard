// GET /api/health backs the Docker healthcheck in docker-compose.yml, which the
// worker services gate on via `depends_on: condition: service_healthy`. If this
// endpoint stops reporting the real Mongo state, the worker starts claiming jobs
// against a database that is not connected yet.
const request = require('supertest');
const mongoose = require('mongoose');
const app = require('../app');
const { connect, disconnect } = require('./setup');

describe('HEALTH - GET /api/health', () => {
    describe('with the database connected', () => {
        beforeAll(async () => await connect());
        afterAll(async () => await disconnect());

        it('returns 200 and reports mongo connected', async () => {
            const res = await request(app).get('/api/health');
            expect(res.statusCode).toBe(200);
            expect(res.body.success).toBe(true);
            expect(res.body.data.mongo).toBe('connected');
        });

        it('needs no authentication', async () => {
            // The healthcheck runs inside the container with no token to offer.
            const res = await request(app).get('/api/health').set('Authorization', '');
            expect(res.statusCode).toBe(200);
        });

        it('answers quickly enough for a 5s healthcheck timeout', async () => {
            const started = Date.now();
            await request(app).get('/api/health');
            expect(Date.now() - started).toBeLessThan(1000);
        });
    });

    describe('with the database disconnected', () => {
        // The important half: liveness alone would report healthy here, the worker
        // would start, and every job would fail on the first query.
        it('returns 503 rather than 200', async () => {
            expect(mongoose.connection.readyState).toBe(0);
            const res = await request(app).get('/api/health');
            expect(res.statusCode).toBe(503);
            expect(res.body.success).toBe(false);
            expect(res.body.data.mongo).toBe('disconnected');
        });
    });
});
