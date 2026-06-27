const connectDB = require('./config/db');
const app = require('./app');

connectDB();

// Note: large files are no longer served from local disk — evidence clips are streamed
// directly from R2 via short-lived presigned GET URLs (see authorityController.getEvidence).

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log('🚀 Server running on port ' + PORT));
