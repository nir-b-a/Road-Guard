import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/axios';
import PageLayout from '../components/PageLayout';
import StatusBadge from '../components/StatusBadge';
import FilterBar from '../components/FilterBar';
import PlateImageButton from '../components/PlateImageButton';
import { PriorityBadge, TypeBadge } from '../components/ViolationBadges';
import { getPriority, PRIORITY_ORDER } from '../utils/violations';
const PER_PAGE = 10;
// FilterBar's Status choice -> the query it stands for. The "by me" choices narrow a status to
// the violations the logged-in authority reviewed itself.
const STATUS_QUERIES = {
  pending: { status: 'pending' },
  verified_by_me: { status: 'verified', reviewed_by: 'me' },
  dismissed_by_me: { status: 'dismissed', reviewed_by: 'me' },
  verified: { status: 'verified' },
  dismissed: { status: 'dismissed' },
};
const ViolationsPage = () => {
  const [violations, setViolations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filters, setFilters] = useState({ status: '', type: '', date: '', plate: '' });
  const [sort, setSort] = useState('');
  const [page, setPage] = useState(1);
  const navigate = useNavigate();
  // Only status/date/plate go to the API; type and sort are applied below, so changing them doesn't refetch.
  useEffect(() => {
    setLoading(true);
    const params = { ...STATUS_QUERIES[filters.status] };
    if (filters.date) params.date = filters.date;
    if (filters.plate) params.license_plate = filters.plate;
    api.get('/authority/violations', { params }).then(res => { setViolations(res.data.data); setPage(1); }).catch(() => setError('Failed to load.')).finally(() => setLoading(false));
  }, [filters.status, filters.date, filters.plate]);
  const shown = violations.filter(v => !filters.type || (v.violationType || 'speeding') === filters.type);
  // The API returns newest first and sort() is stable, so equal priorities stay newest first.
  if (sort) {
    const dir = sort === 'priority_asc' ? -1 : 1;
    shown.sort((a, b) => dir * (PRIORITY_ORDER[getPriority(a).label] - PRIORITY_ORDER[getPriority(b).label]));
  }
  const totalPages = Math.ceil(shown.length / PER_PAGE);
  const paginated = shown.slice((page-1)*PER_PAGE, page*PER_PAGE);
  return (
    <PageLayout>
      <div className="max-w-6xl mx-auto px-6 py-8">
        <h1 className="text-2xl font-bold text-white mb-6">All Violations</h1>
        <FilterBar filters={filters} onChange={(f) => { setFilters(f); setPage(1); }} sort={sort} onSortChange={(s) => { setSort(s); setPage(1); }} />
        {loading && <p className="text-gray-400">Loading...</p>}
        {error && <p className="text-red-400">{error}</p>}
        {!loading && <>
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-x-auto">
            <table className="w-full text-sm">
              <thead><tr className="text-gray-400 text-left border-b border-gray-700">
                <th className="px-6 py-3">License Plate</th><th className="px-6 py-3">Priority</th><th className="px-6 py-3">Type</th><th className="px-6 py-3">Speed</th><th className="px-6 py-3">Location</th><th className="px-6 py-3">Date</th><th className="px-6 py-3">Status</th><th className="px-6 py-3">Action</th>
              </tr></thead>
              <tbody>
                {paginated.map(v => (
                  <tr key={v._id} onClick={() => navigate('/violations/' + v._id)} className="border-b border-gray-700/50 hover:bg-gray-700/30 cursor-pointer transition">
                    <td className="px-6 py-3 font-mono text-white font-semibold">
                      <div className="flex items-center gap-2">{v.carId}{v.plateImagePath && <PlateImageButton violationId={v._id} plate={v.carId} />}</div>
                    </td>
                    <td className="px-6 py-3"><PriorityBadge violation={v} /></td>
                    <td className="px-6 py-3"><TypeBadge type={v.violationType} /></td>
                    <td className="px-6 py-3 text-white whitespace-nowrap">{v.calculatedSpeed} km/h</td>
                    <td className="px-6 py-3 text-gray-400">{v.location?.lat?.toFixed(3)}, {v.location?.lon?.toFixed(3)}</td>
                    <td className="px-6 py-3 text-gray-400">{new Date(v.detectedAt).toLocaleDateString()}</td>
                    <td className="px-6 py-3"><StatusBadge status={v.status} /></td>
                    <td className="px-6 py-3"><button className="text-xs bg-gray-700 hover:bg-gray-600 text-white px-3 py-1.5 rounded transition">View Details</button></td>
                  </tr>
                ))}
                {paginated.length === 0 && <tr><td colSpan={8} className="px-6 py-8 text-center text-gray-500">No violations found</td></tr>}
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