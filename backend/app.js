require('dotenv').config();
require('express-async-errors');
const express = require('express');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const authRoutes = require('./routes/authRoutes');
const driverRoutes = require('./routes/driverRoutes');
const authorityRoutes = require('./routes/authorityRoutes');

const app = express();

app.set('trust proxy', 1);
app.use(cors({ origin: '*', credentials: false }));
// Endpoints now carry only small JSON (presign requests, completion, results) — never the
// big files — so a tight body cap is safe and stops oversized payloads.
app.use(express.json({ limit: process.env.JSON_BODY_LIMIT || '10mb' }));
app.use(express.urlencoded({ extended: true, limit: process.env.JSON_BODY_LIMIT || '10mb' }));

// Backpressure: the server should absorb a high request rate, but cap per-IP abuse.
// Disabled under test so the suite isn't throttled; internal worker polling is excluded.
if (process.env.NODE_ENV !== 'test') {
    const limiter = rateLimit({
        windowMs: 60 * 1000,
        limit: parseInt(process.env.RATE_LIMIT_MAX || '600', 10),
        standardHeaders: true,
        legacyHeaders: false,
        message: { success: false, message: 'Too many requests — slow down', data: null }
    });
    app.use('/api', (req, res, next) => req.originalUrl.startsWith('/api/internal') ? next() : limiter(req, res, next));
}

// Request/response logger: prints the final status and, on 4xx/5xx, the error message,
// so rejections are visible in the server console (clients only get a truncated toast).
app.use((req, res, next) => {
    const startedAt = Date.now();
    const origJson = res.json.bind(res);
    res.json = (body) => {
        const ms = Date.now() - startedAt;
        const reason = (res.statusCode >= 400 && body && body.message) ? ' - ' + body.message : '';
        const tag = res.statusCode >= 400 ? 'WARN' : 'OK';
        console.log(`[${tag}] ${req.method} ${req.originalUrl} -> ${res.statusCode} (${ms}ms)${reason}`);
        return origJson(body);
    };
    next();
});

// Registered before the routers so it stays cheap - no auth, no DB query, no
// router stack to walk. It does sit under the /api rate limiter mounted above,
// which is fine: the healthcheck polls once every 10s against a 600/min cap.
// Docker's healthcheck polls this, and the worker waits for it to go healthy
// before it starts claiming jobs: a worker that polls an API whose Mongo
// connection is still opening just burns retries and logs noise.
// Liveness alone is not enough - server.js exits(1) on a failed initial connect,
// but a mid-flight drop leaves the process up with an unusable database, so the
// mongoose connection state is what decides the status code.
app.get('/api/health', (req, res) => {
    // mongoose readyState: 0 disconnected, 1 connected, 2 connecting, 3 disconnecting.
    const state = require('mongoose').connection.readyState;
    const ok = state === 1;
    res.status(ok ? 200 : 503).json({
        success: ok,
        message: ok ? 'ok' : 'database not connected',
        data: { mongo: ['disconnected', 'connected', 'connecting', 'disconnecting'][state] ?? 'unknown' }
    });
});

app.use('/api/auth', authRoutes);
app.use('/api/authority', authorityRoutes);
app.use('/api/internal', require('./routes/internalRoutes'));
app.use('/api', driverRoutes);

app.get('/', (req, res) => {
    res.json({ success: true, message: 'RoadGuard API is running' });
});

app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(err.status || 500).json({ success: false, message: err.message || 'Internal Server Error', data: null });
});

module.exports = app;