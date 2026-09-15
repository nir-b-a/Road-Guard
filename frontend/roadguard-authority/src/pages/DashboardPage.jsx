import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/axios';
import PageLayout from '../components/PageLayout';
import StatusBadge from '../components/StatusBadge';
import PlateImageButton from '../components/PlateImageButton';
import { PriorityBadge, TypeBadge } from '../components/ViolationBadges';
import { getPriority, PRIORITY_ORDER } from '../utils/violations';
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
  const pendingViolations = violations
    .filter(v => v.status === 'pending')
    .sort((a, b) => PRIORITY_ORDER[getPriority(a).label] - PRIORITY_ORDER[getPriority(b).label]);
  const pending = pendingViolations.length;
  const verified = violations.filter(v => v.status === 'verified').length;
  const dismissed = violations.filter(v => v.status === 'dismissed').length;
  return (
    <PageLayout>
        <div className="max-w-6xl mx-auto px-6 py-8">
          <h1 className="text-2xl font-bold text-white mb-6">Dashboard</h1>
          {error && <p className="text-red-400">{error}</p>}
        {loading && (
          <div className="animate-pulse">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="bg-gray-800 border border-gray-700/30 rounded-xl p-6 h-[88px]" />
              ))}
            </div>
            <div className="bg-gray-800 border border-gray-700 rounded-xl">
              <div className="px-6 py-4 border-b border-gray-700">
                <div className="h-5 w-40 bg-gray-700 rounded" />
              </div>
              <div className="p-6 space-y-3">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div key={i} className="h-5 bg-gray-700/50 rounded" />
                ))}
              </div>
            </div>
          </div>
        )}
        {!loading && <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <StatCard label="Total" count={total} color="border-cyan-500/30" icon="📋" />
            <StatCard label="Pending" count={pending} color="border-yellow-500/30" icon="⏳" />
            <StatCard label="Verified" count={verified} color="border-green-500/30" icon="✅" />
            <StatCard label="Dismissed" count={dismissed} color="border-red-500/30" icon="❌" />
          </div>
          <div className="bg-gray-800 border border-gray-700 rounded-xl">
            <div className="flex justify-between items-center px-6 py-4 border-b border-gray-700">
              <h2 className="font-semibold text-white">Pending Review</h2>
              <button onClick={() => navigate('/violations')} className="text-sm text-cyan-400 hover:text-cyan-300">View All →</button>
            </div>
            <table className="w-full text-sm">
              <thead><tr className="text-gray-400 text-left border-b border-gray-700">
                <th className="px-6 py-3">License Plate</th><th className="px-6 py-3">Priority</th><th className="px-6 py-3">Type</th><th className="px-6 py-3">Date</th><th className="px-6 py-3">Status</th>
              </tr></thead>
              <tbody>
                {pendingViolations.slice(0,5).map(v => (
                  <tr key={v._id} onClick={() => navigate('/violations/' + v._id)} className="border-b border-gray-700/50 hover:bg-gray-700/30 cursor-pointer transition">
                    <td className="px-6 py-3 font-mono text-white">
                      <div className="flex items-center gap-2">{v.carId}{v.plateImagePath && <PlateImageButton violationId={v._id} plate={v.carId} />}</div>
                    </td>
                    <td className="px-6 py-3"><PriorityBadge violation={v} /></td>
                    <td className="px-6 py-3"><TypeBadge type={v.violationType} /></td>
                    <td className="px-6 py-3 text-gray-400">{new Date(v.detectedAt).toLocaleDateString()}</td>
                    <td className="px-6 py-3"><StatusBadge status={v.status} /></td>
                  </tr>
                ))}
                {pendingViolations.length === 0 && <tr><td colSpan={5} className="px-6 py-6 text-center text-gray-500">No pending violations</td></tr>}
              </tbody>
            </table>
          </div>
        </>}
        </div>
    </PageLayout>
  );
};
export default DashboardPage;