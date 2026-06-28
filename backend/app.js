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