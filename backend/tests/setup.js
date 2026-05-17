const { MongoMemoryServer } = require('mongodb-memory-server');
const mongoose = require('mongoose');
let mongoServer;
const connect = async () => {
    mongoServer = await MongoMemoryServer.create();
    const uri = mongoServer.getUri();
    await mongoose.connect(uri);
};
const disconnect = async () => {
    await mongoose.connection.dropDatabase();
    await mongoose.connection.close();
    await mongoServer.stop();
};
const clearCollections = async () => {
    const collections = mongoose.connection.collections;
    for (const key in collections) { await collections[key].deleteMany({}); }
};
module.exports = { connect, disconnect, clearCollections };