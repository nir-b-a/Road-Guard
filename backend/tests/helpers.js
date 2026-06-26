const request = require('supertest');
const app = require('../app');
const Drive = require('../models/Drive');
const Violation = require('../models/Violation');

const getDriverToken = async () => {
    const res = await request(app).post('/api/auth/register')
        .send({ name: 'Test Driver', email: 'driver_' + Date.now() + '@test.com', password: 'password123', role: 'driver' });
    return { token: res.body.data.token, userId: res.body.data.id };
};

const getAuthorityToken = async () => {
    const res = await request(app).post('/api/auth/register')
        .send({ name: 'Test Authority', email: 'authority_' + Date.now() + '@test.com', password: 'password123', role: 'authority' });
    return { token: res.body.data.token, userId: res.body.data.id };
};

const seedViolation = async (driverId) => {
    const drive = await Drive.create({ driverId, sessionId: 'session_' + Date.now(), videoPath: 'sessions/s/video.mp4', status: 'processed' });
    const violation = await Violation.create({ driveId: drive._id, driverId, videoClipPath: 'sessions/s/out/clip.mp4', carId: 'ABC-1234', calculatedSpeed: 95, location: { lat: 32.08, lon: 34.78 }, status: 'pending' });
    return { drive, violation };
};

module.exports = { getDriverToken, getAuthorityToken, seedViolation };
