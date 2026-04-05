import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/axios';
import Navbar from '../components/Navbar';
import StatusBadge from '../components/StatusBadge';
const SearchPage = () => {
  const [plate, setPlate] = useState('');
  const [status, setStatus] = useState('');
  const [results, setResults] = useState([]);
  const [searched, setSearched] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const handleSearch = async (e) => {
    e.preventDefault(); setLoading(true); setError(''); setSearched(false);
    try {
      const params = {};
      if (plate) params.license_plate = plate;
      if (status) params.status = status;
      const res = await api.get('/authority/search', { params });
      setResults(res.data.data.results); setSearched(true);
    } catch { setError('Search failed.'); }
    finally { setLoading(false); }
  };
  return (
    <div className="min-h-screen bg-gray-900">
      <Navbar />
      <div className="max-w-5xl mx-auto px-6 py-8">
        <h1 className="text-2xl font-bold text-white mb-6">Search Violations</h1>
        <form onSubmit={handleSearch} className="bg-gray-800 border border-gray-700 rounded-xl p-6 flex flex-wrap gap-4 items-end mb-8">
          <div className="flex flex-col gap-1 flex-1 min-w-[180px]">
            <label className="text-xs text-gray-400 uppercase">License Plate</label>
            <input type="text" value={plate} onChange={(e) => setPlate(e.target.value)} placeholder="e.g. ABC-1234"
              className="bg-gray-900 border border-gray-700 text-white rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-red-600" />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-gray-400 uppercase">Status</label>
            <select value={status} onChange={(e) => setStatus(e.target.value)} className="bg-gray-900 border border-gray-700 text-gray-200 text-sm rounded-lg px-3 py-2.5 focus:outline-none focus:border-red-600">
              <option value="">All</option><option value="pending">Pending</option><option value="verified">Verified</option><option value="dismissed">Dismissed</option>
            </select>
          </div>
          <button type="submit" disabled={loading} className="bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white font-semibold px-6 py-2.5 rounded-lg transition">{loading ? 'Searching...' : 'Search'}</button>
        </form>
        {error && <p className="text-red-400 mb-4">{error}</p>}
        {searched && (results.length === 0 ? (
          <div className="text-center py-16 text-gray-500"><p className="text-4xl mb-3">🔍</p><p>No results found</p></div>
        ) : (
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-700"><p className="text-gray-400 text-sm">{results.length} result(s) found</p></div>
            <table className="w-full text-sm">
              <thead><tr className="text-gray-400 text-left border-b border-gray-700">
                <th className="px-6 py-3">License Plate</th><th className="px-6 py-3">Speed</th><th className="px-6 py-3">Date</th><th className="px-6 py-3">Status</th><th className="px-6 py-3">Action</th>
              </tr></thead>
              <tbody>
                {results.map(v => (
                  <tr key={v._id} className="border-b border-gray-700/50 hover:bg-gray-700/30 transition">
                    <td className="px-6 py-3 font-mono text-white font-semibold">{v.carId}</td>
                    <td className="px-6 py-3 text-white">{v.calculatedSpeed} km/h</td>
                    <td className="px-6 py-3 text-gray-400">{new Date(v.detectedAt).toLocaleDateString()}</td>
                    <td className="px-6 py-3"><StatusBadge status={v.status} /></td>
                    <td className="px-6 py-3"><button onClick={() => navigate('/violations/' + v._id)} className="text-xs bg-gray-700 hover:bg-gray-600 text-white px-3 py-1.5 rounded transition">View Details</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  );
};
export default SearchPage;