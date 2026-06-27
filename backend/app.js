require('dotenv').config();
require('express-async-errors');
const express = require('express');
const cors = require('cors');
const path = require('path');
const authRoutes = require('./routes/authRoutes');
const driverRoutes = require('./routes/driverRoutes');
const authorityRoutes = require('./routes/authorityRoutes');

const app = express();

app.use(cors({ origin: '*', credentials: false }));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// app.use((req, res, next) => { console.log(req.method, req.originalUrl); next(); });

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