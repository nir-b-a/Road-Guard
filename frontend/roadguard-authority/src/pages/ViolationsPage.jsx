import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/axios';
import PageLayout from '../components/PageLayout';
import StatusBadge from '../components/StatusBadge';
import FilterBar from '../components/FilterBar';
const PER_PAGE = 10;
const ViolationsPage = () => {
  const [violations, setViolations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filters, setFilters] = useState({ status: '', date: '', plate: '' });
  const [page, setPage] = useState(1);
  const navigate = useNavigate();
  useEffect(() => {
    setLoading(true);
    const params = {};
    if (filters.status) params.status = filters.status;
    if (filters.date) params.date = filters.date;
    if (filters.plate) params.license_plate = filters.plate;
    api.get('/authority/violations', { params }).then(res => { setViolations(res.data.data); setPage(1); }).catch(() => setError('Failed to load.')).finally(() => setLoading(false));
  }, [filters]);
  const totalPages = Math.ceil(violations.length / PER_PAGE);
  const paginated = violations.slice((page-1)*PER_PAGE, page*PER_PAGE);
  return (
    <PageLayout>
      <div className="max-w-6xl mx-auto px-6 py-8">
        <h1 className="text-2xl font-bold text-white mb-6">All Violations</h1>
        <FilterBar filters={filters} onChange={setFilters} />
        {loading && <p className="text-gray-400">Loading...</p>}
        {error && <p className="text-red-400">{error}</p>}
        {!loading && <>
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
            <table className="w-full text-sm">
              <thead><tr className="text-gray-400 text-left border-b border-gray-700">
                <th className="px-6 py-3">License Plate</th><th className="px-6 py-3">Speed</th><th className="px-6 py-3">Location</th><th className="px-6 py-3">Date</th><th className="px-6 py-3">Status</th><th className="px-6 py-3">Action</th>
              </tr></thead>
              <tbody>
                {paginated.map(v => (
                  <tr key={v._id} onClick={() => navigate('/violations/' + v._id)} className="border-b border-gray-700/50 hover:bg-gray-700/30 cursor-pointer transition">
                    <td className="px-6 py-3 font-mono text-white font-semibold">{v.carId}</td>
                    <td className="px-6 py-3 text-white">{v.calculatedSpeed} km/h</td>
                    <td className="px-6 py-3 text-gray-400">{v.location?.lat?.toFixed(3)}, {v.location?.lon?.toFixed(3)}</td>
                    <td className="px-6 py-3 text-gray-400">{new Date(v.detectedAt).toLocaleDateString()}</td>
                    <td className="px-6 py-3"><StatusBadge status={v.status} /></td>
                    <td className="px-6 py-3"><button className="text-xs bg-gray-700 hover:bg-gray-600 text-white px-3 py-1.5 rounded transition">View Details</button></td>
                  </tr>
                ))}
                {paginated.length === 0 && <tr><td colSpan={6} className="px-6 py-8 text-center text-gray-500">No violations found</td></tr>}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="flex justify-center gap-2 mt-4">
              {Array.from({ length: totalPages }, (_, i) => i+1).map(p => (
                <button key={p} onClick={() => setPage(p)} className={"px-3 py-1 rounded text-sm " + (p===page ? 'bg-cyan-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600')}>{p}</button>
              ))}
            </div>
          )}
        </>}
      </div>
    </PageLayout>
  );
};
export default ViolationsPage;