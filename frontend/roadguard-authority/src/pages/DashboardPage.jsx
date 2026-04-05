import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/axios';
import Navbar from '../components/Navbar';
import StatusBadge from '../components/StatusBadge';
const StatCard = ({ label, count, color, icon }) => (
  <div className={"bg-gray-800 border " + color + " rounded-xl p-6 flex items-center gap-4"}>
    <span className="text-3xl">{icon}</span>
    <div><p className="text-gray-400 text-sm">{label}</p><p className="text-3xl font-bold text-white">{count}</p></div>
  </div>
);
const DashboardPage = () => {
  const [violations, setViolations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  useEffect(() => {
    api.get('/authority/violations').then(res => setViolations(res.data.data)).catch(() => setError('Failed to load.')).finally(() => setLoading(false));
  }, []);
  const total = violations.length;
  const pending = violations.filter(v => v.status === 'pending').length;
  const verified = violations.filter(v => v.status === 'verified').length;
  const dismissed = violations.filter(v => v.status === 'dismissed').length;
  return (
    <div className="min-h-screen bg-gray-900">
      <Navbar />
      <div className="max-w-6xl mx-auto px-6 py-8">
        <h1 className="text-2xl font-bold text-white mb-6">Dashboard</h1>
        {loading && <p className="text-gray-400">Loading...</p>}
        {error && <p className="text-red-400">{error}</p>}
        {!loading && <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <StatCard label="Total" count={total} color="border-blue-500/30" icon="📋" />
            <StatCard label="Pending" count={pending} color="border-yellow-500/30" icon="⏳" />
            <StatCard label="Verified" count={verified} color="border-green-500/30" icon="✅" />
            <StatCard label="Dismissed" count={dismissed} color="border-red-500/30" icon="❌" />
          </div>
          <div className="bg-gray-800 border border-gray-700 rounded-xl">
            <div className="flex justify-between items-center px-6 py-4 border-b border-gray-700">
              <h2 className="font-semibold text-white">Recent Violations</h2>
              <button onClick={() => navigate('/violations')} className="text-sm text-red-500 hover:text-red-400">View All →</button>
            </div>
            <table className="w-full text-sm">
              <thead><tr className="text-gray-400 text-left border-b border-gray-700">
                <th className="px-6 py-3">License Plate</th><th className="px-6 py-3">Speed</th><th className="px-6 py-3">Date</th><th className="px-6 py-3">Status</th>
              </tr></thead>
              <tbody>
                {violations.slice(0,5).map(v => (
                  <tr key={v._id} onClick={() => navigate('/violations/' + v._id)} className="border-b border-gray-700/50 hover:bg-gray-700/30 cursor-pointer transition">
                    <td className="px-6 py-3 font-mono text-white">{v.carId}</td>
                    <td className="px-6 py-3 text-white">{v.calculatedSpeed} km/h</td>
                    <td className="px-6 py-3 text-gray-400">{new Date(v.detectedAt).toLocaleDateString()}</td>
                    <td className="px-6 py-3"><StatusBadge status={v.status} /></td>
                  </tr>
                ))}
                {violations.length === 0 && <tr><td colSpan={4} className="px-6 py-6 text-center text-gray-500">No violations yet</td></tr>}
              </tbody>
            </table>
          </div>
        </>}
      </div>
    </div>
  );
};
export default DashboardPage;