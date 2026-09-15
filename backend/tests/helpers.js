const request = require('supertest');
const app = require('../app');
const Drive = require('../models/Drive');
const Violation = require('../models/Violation');

const getDriverToken = async () => {
    const res = await request(app).post('/api/auth/register')
        .send({ name: 'Test Driver', email: 'driver_' + Date.now() + '@test.com', password: 'password123', role: 'driver' });
    return { token: res.body.data.token, userId: res.body.data.id };
};

// Registering an authority requires the invite code (authController.register).
const INVITE_CODE = process.env.INVITE_CODE || 'ROADGUARD-2026';

const getAuthorityToken = async () => {
    const res = await request(app).post('/api/auth/register')
        .send({ name: 'Test Authority', email: 'authority_' + Date.now() + '@test.com', password: 'password123', role: 'authority', inviteCode: INVITE_CODE });
    return { token: res.body.data.token, userId: res.body.data.id };
};

const seedViolation = async (driverId, fields = {}) => {
    const drive = await Drive.create({ driverId, sessionId: 'session_' + Date.now() + '_' + Math.random(), videoPath: 'sessions/s/video.mp4', status: 'processed' });
    const violation = await Violation.create({ driveId: drive._id, driverId, videoClipPath: 'sessions/s/out/clip.mp4', carId: 'ABC-1234', calculatedSpeed: 95, location: { lat: 32.08, lon: 34.78 }, status: 'pending', ...fields });
    return { drive, violation };
};

module.exports = { getDriverToken, getAuthorityToken, seedViolation, INVITE_CODE };
