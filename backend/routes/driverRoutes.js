const express = require('express');
const router = express.Router();
const { protect } = require('../middleware/auth');
const { authorize } = require('../middleware/roleGuard');
const upload = require('../middleware/uploadMiddleware');
const { uploadDrive, getNotifications } = require('../controllers/driverController');
router.post('/driver/upload', protect, authorize('driver'), upload.fields(upload.UPLOAD_FIELDS), uploadDrive);
router.get('/notifications/inbox/:driverId', protect, authorize('driver'), getNotifications);
module.exports = router;