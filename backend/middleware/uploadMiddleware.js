const multer = require('multer');
const path = require('path');
const fs = require('fs');

// Each session's artifacts live together under uploads/sessions/<sessionId>/.
const sessionsRoot = path.join(__dirname, '../uploads/sessions');
if (!fs.existsSync(sessionsRoot)) fs.mkdirSync(sessionsRoot, { recursive: true });

// Non-video artifact fields. The Android client names each multipart field exactly
// after the on-device file, so the field name IS the stored file name.
const DATA_FIELDS = ['frames.csv', 'gps.csv', 'gravity.csv', 'gyro.csv', 'linacc.csv', 'intrinsics.json', 'tags.json'];

// Field list for upload.fields(): the video plus every data file, one of each.
const UPLOAD_FIELDS = [{ name: 'video', maxCount: 1 }, ...DATA_FIELDS.map(name => ({ name, maxCount: 1 }))];

const VIDEO_MIMES = ['video/mp4', 'video/mpeg', 'video/quicktime'];

// sessionId becomes a directory name, so strip anything that could escape uploads/.
const safeSessionId = (raw) => {
    const cleaned = String(raw || '').trim().replace(/[^a-zA-Z0-9_-]/g, '_');
    return cleaned || ('session_' + Date.now());
};

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        // The client sends sessionId as the FIRST multipart field, so it is already
        // parsed into req.body by the time any file part reaches us.
        if (!req._sessionId) req._sessionId = safeSessionId(req.body.sessionId);
        const dir = path.join(sessionsRoot, req._sessionId);
        fs.mkdir(dir, { recursive: true }, (err) => cb(err, dir));
    },
    filename: (req, file, cb) => {
        // Data fields are named after their file; keep the video's original name.
        const name = file.fieldname === 'video' ? path.basename(file.originalname) : file.fieldname;
        cb(null, name);
    }
});

const fileFilter = (req, file, cb) => {
    if (file.fieldname === 'video') {
        return VIDEO_MIMES.includes(file.mimetype)
            ? cb(null, true)
            : cb(new Error('Only video files allowed for the video field'), false);
    }
    if (DATA_FIELDS.includes(file.fieldname)) return cb(null, true);   // csv / json text
    return cb(new Error('Unexpected upload field: ' + file.fieldname), false);
};

const upload = multer({ storage, fileFilter, limits: { fileSize: 500 * 1024 * 1024 } });

// Expose the field list so the route can wire upload.fields(...) without re-listing.
upload.UPLOAD_FIELDS = UPLOAD_FIELDS;
upload.DATA_FIELDS = DATA_FIELDS;

module.exports = upload;
