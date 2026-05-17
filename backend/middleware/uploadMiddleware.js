const multer = require('multer');
const path = require('path');
const fs = require('fs');
const uploadDir = path.join(__dirname, '../uploads/videos');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, uploadDir),
    filename: (req, file, cb) => cb(null, Date.now() + '-' + file.originalname)
});
const fileFilter = (req, file, cb) => {
    const allowed = ['video/mp4', 'video/mpeg', 'video/quicktime'];
    allowed.includes(file.mimetype) ? cb(null, true) : cb(new Error('Only video files allowed'), false);
};
const upload = multer({ storage, fileFilter, limits: { fileSize: 500 * 1024 * 1024 } });
module.exports = upload;