const express = require('express');
const router = express.Router();
const { protect } = require('../middleware/auth');
const { authorize } = require('../middleware/roleGuard');
const { initUpload, completeUpload, getNotifications } = require('../controllers/driverController');

// Two-step direct-to-R2 upload: init hands out presigned PUT URLs, complete validates &
// queues. The big video bytes go app -> R2 directly, never through this route.
router.post('/driver/upload/init', protect, authorize('driver'), initUpload);
router.post('/driver/upload/complete', protect, authorize('driver'), completeUpload);
router.get('/notifications/inbox/:driverId', protect, authorize('driver'), getNotifications);

module.exports = router;
