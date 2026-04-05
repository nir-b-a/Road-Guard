const connectDB = require('./config/db');
const app = require('./app');

connectDB();

app.use('/uploads', require('express').static('uploads'));

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log('🚀 Server running on port ' + PORT));
