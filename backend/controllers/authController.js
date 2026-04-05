const jwt = require('jsonwebtoken');
const User = require('../models/User');

const INVITE_CODE = process.env.INVITE_CODE || 'ROADGUARD-2026';
const generateToken = (id) => jwt.sign({ id }, process.env.JWT_SECRET, { expiresIn: '24h' });

const register = async (req, res) => {
    const { name, email, password, role, inviteCode } = req.body;
    if (inviteCode !== INVITE_CODE)
        return res.status(403).json({ success: false, message: 'Invalid invite code', data: null });
    if (!name || !email || !password)
        return res.status(400).json({ success: false, message: 'Please provide name, email and password', data: null });
    const existingUser = await User.findOne({ email });
    if (existingUser)
        return res.status(400).json({ success: false, message: 'Email already registered', data: null });
    const user = await User.create({ name, email, password, role: role || 'driver' });
    res.status(201).json({ success: true, message: 'Registered successfully', data: { id: user._id, name: user.name, email: user.email, role: user.role, token: generateToken(user._id) } });
};

const login = async (req, res) => {
    const { email, password } = req.body;
    if (!email || !password)
        return res.status(400).json({ success: false, message: 'Please provide email and password', data: null });
    const user = await User.findOne({ email });
    if (!user || !(await user.matchPassword(password)))
        return res.status(401).json({ success: false, message: 'Invalid credentials', data: null });
    res.json({ success: true, message: 'Login successful', data: { id: user._id, name: user.name, email: user.email, role: user.role, token: generateToken(user._id) } });
};

module.exports = { register, login };
