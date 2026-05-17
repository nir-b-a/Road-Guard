const express = require('express');
const router = express.Router();
const { protect } = require('../middleware/auth');
const { authorize } = require('../middleware/roleGuard');
const { getViolations, verifyViolation, dismissViolation, getEvidence, searchViolations, getDrives } = require('../controllers/authorityController');

router.use(protect, authorize('authority'));
router.get('/violations', getViolations);
router.post('/violation/:id/verify', verifyViolation);
router.post('/violation/:id/dismiss', dismissViolation);
router.get('/evidence/:id', getEvidence);
router.get('/search', searchViolations);
router.get('/drives', getDrives);

module.exports = router;
