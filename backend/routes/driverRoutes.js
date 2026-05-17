const express = require('express');
const router = express.Router();
const { protect } = require('../middleware/auth');
const { authorize } = require('../middleware/roleGuard');
const upload = require('../middleware/uploadMiddleware');
const { uploadDrive, getNotifications } = require('../controllers/driverController');
router.post('/driver/upload', protect, authorize('driver'), upload.single('video'), uploadDrive);
router.get('/notifications/inbox/:driverId', protect, authorize('driver'), getNotifications);
module.exports = router;