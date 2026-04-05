import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/axios';
import Navbar from '../components/Navbar';

const DrivesPage = () => {
  const [drives, setDrives] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    api.get('/authority/drives')
      .then(res => setDrives(res.data.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="min-h-screen bg-gray-900">
      <Navbar />
      <div className="max-w-6xl mx-auto px-6 py-8">
        <h1 className="text-2xl font-bold text-white mb-6">Driver Uploads</h1>
        {loading && <p className="text-gray-400">Loading...</p>}
        <div className="space-y-4">
          {drives.map(drive => (
            <div key={drive._id} className="bg-gray-800 border border-gray-700 rounded-xl p-6">
              <div className="flex justify-between items-start mb-4">
                <div>
                  <p className="text-white font-semibold">{drive.driverId?.name || 'Unknown Driver'}</p>
                  <p className="text-gray-400 text-sm">{drive.driverId?.email}</p>
                  <p className="text-gray-500 text-xs mt-1">{new Date(drive.createdAt).toLocaleString()}</p>
                </div>
                <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
                  drive.status === 'pending' ? 'bg-yellow-500/20 text-yellow-400' : 'bg-green-500/20 text-green-400'
                }`}>
                  {drive.status}
                </span>
              </div>
              <video
                controls
                className="w-full rounded-lg bg-black max-h-64"
                src={`http://localhost:5000/${drive.videoPath}`}
              >
                Your browser does not support video.
              </video>
            </div>
          ))}
          {!loading && drives.length === 0 && (
            <p className="text-gray-500 text-center py-12">No drives uploaded yet</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default DrivesPage;
