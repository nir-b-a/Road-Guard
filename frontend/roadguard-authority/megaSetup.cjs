const fs = require('fs');
const path = require('path');

const files = {

'tailwind.config.js': `/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: { extend: {} },
  plugins: [],
}`,

'src/index.css': `@tailwind base;
@tailwind components;
@tailwind utilities;
body { background-color: #111827; color: #f9fafb; }`,

'src/main.jsx': `import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode><App /></React.StrictMode>
);`,

'src/App.jsx': `import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import ViolationsPage from './pages/ViolationsPage';
import ViolationDetailPage from './pages/ViolationDetailPage';
import SearchPage from './pages/SearchPage';
const App = () => (
  <BrowserRouter>
    <AuthProvider>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/dashboard" element={<ProtectedRoute><DashboardPage /></ProtectedRoute>} />
        <Route path="/violations" element={<ProtectedRoute><ViolationsPage /></ProtectedRoute>} />
        <Route path="/violations/:id" element={<ProtectedRoute><ViolationDetailPage /></ProtectedRoute>} />
        <Route path="/search" element={<ProtectedRoute><SearchPage /></ProtectedRoute>} />
      </Routes>
    </AuthProvider>
  </BrowserRouter>
);
export default App;`,

'src/api/axios.js': `import axios from 'axios';
const api = axios.create({ baseURL: 'http://localhost:5000/api' });
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) config.headers.Authorization = 'Bearer ' + token;
  return config;
});
export default api;`,

'src/context/AuthContext.jsx': `import { createContext, useContext, useState } from 'react';
const AuthContext = createContext(null);
export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(() => localStorage.getItem('token') || null);
  const [user, setUser] = useState(() => { const u = localStorage.getItem('user'); return u ? JSON.parse(u) : null; });
  const login = (newToken, newUser) => {
    localStorage.setItem('token', newToken);
    localStorage.setItem('user', JSON.stringify(newUser));
    setToken(newToken); setUser(newUser);
  };
  const logout = () => {
    localStorage.removeItem('token'); localStorage.removeItem('user');
    setToken(null); setUser(null);
  };
  return <AuthContext.Provider value={{ token, user, login, logout }}>{children}</AuthContext.Provider>;
};
export const useAuth = () => useContext(AuthContext);`,

'src/components/ProtectedRoute.jsx': `import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
const ProtectedRoute = ({ children }) => {
  const { token } = useAuth();
  return token ? children : <Navigate to="/login" replace />;
};
export default ProtectedRoute;`,

'src/components/Navbar.jsx': `import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
const Navbar = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const handleLogout = () => { logout(); navigate('/login'); };
  return (
    <nav className="bg-gray-800 border-b border-gray-700 px-6 py-4 flex items-center justify-between">
      <div className="flex items-center gap-8">
        <span className="text-red-600 font-bold text-xl">🛡️ RoadGuard</span>
        <div className="flex gap-6 text-sm text-gray-300">
          <Link to="/dashboard" className="hover:text-white transition">Dashboard</Link>
          <Link to="/violations" className="hover:text-white transition">Violations</Link>
          <Link to="/search" className="hover:text-white transition">Search</Link>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <span className="text-gray-400 text-sm">{user?.name || 'Officer'}</span>
        <button onClick={handleLogout} className="bg-red-600 hover:bg-red-700 text-white text-sm px-4 py-1.5 rounded transition">Logout</button>
      </div>
    </nav>
  );
};
export default Navbar;`,

'src/components/StatusBadge.jsx': `const StatusBadge = ({ status }) => {
  const styles = {
    pending:   'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30',
    verified:  'bg-green-500/20 text-green-400 border border-green-500/30',
    dismissed: 'bg-red-500/20 text-red-400 border border-red-500/30',
  };
  return <span className={\`px-2.5 py-1 rounded-full text-xs font-semibold uppercase \${styles[status] || styles.pending}\`}>{status}</span>;
};
export default StatusBadge;`,

'src/components/FilterBar.jsx': `const FilterBar = ({ filters, onChange }) => (
  <div className="flex flex-wrap gap-4 mb-6">
    <div className="flex flex-col gap-1">
      <label className="text-xs text-gray-400 uppercase">Status</label>
      <select value={filters.status} onChange={(e) => onChange({ ...filters, status: e.target.value })}
        className="bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded px-3 py-2 focus:outline-none focus:border-red-600">
        <option value="">All</option>
        <option value="pending">Pending</option>
        <option value="verified">Verified</option>
        <option value="dismissed">Dismissed</option>
      </select>
    </div>
    <div className="flex flex-col gap-1">
      <label className="text-xs text-gray-400 uppercase">Date</label>
      <input type="date" value={filters.date} onChange={(e) => onChange({ ...filters, date: e.target.value })}
        className="bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded px-3 py-2 focus:outline-none focus:border-red-600" />
    </div>
    {(filters.status || filters.date) && (
      <div className="flex flex-col justify-end">
        <button onClick={() => onChange({ status: '', date: '' })} className="text-xs text-gray-400 hover:text-red-400 transition py-2">✕ Clear</button>
      </div>
    )}
  </div>
);
export default FilterBar;`,

'src/components/ViolationCard.jsx': `import { useNavigate } from 'react-router-dom';
import StatusBadge from './StatusBadge';
const ViolationCard = ({ violation }) => {
  const navigate = useNavigate();
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4 hover:border-red-600 transition cursor-pointer" onClick={() => navigate('/violations/' + violation._id)}>
      <div className="flex justify-between items-start mb-2">
        <span className="font-bold text-white text-lg">{violation.carId}</span>
        <StatusBadge status={violation.status} />
      </div>
      <div className="text-sm text-gray-400 space-y-1">
        <p>🚗 Speed: <span className="text-white">{violation.calculatedSpeed} km/h</span></p>
        <p>📍 {violation.location?.lat?.toFixed(4)}, {violation.location?.lon?.toFixed(4)}</p>
        <p>🕒 {new Date(violation.detectedAt).toLocaleString()}</p>
      </div>
    </div>
  );
};
export default ViolationCard;`,

'src/pages/LoginPage.jsx': `import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../api/axios';
const LoginPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();
  const handleSubmit = async (e) => {
    e.preventDefault(); setError(''); setLoading(true);
    try {
      const res = await api.post('/auth/login', { email, password });
      if (res.data.success) {
        const { token, ...user } = res.data.data;
        if (user.role !== 'authority') { setError('Access denied. This portal is for authority officers only.'); setLoading(false); return; }
        login(token, user); navigate('/dashboard');
      }
    } catch (err) { setError(err.response?.data?.message || 'Login failed.'); }
    finally { setLoading(false); }
  };
  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-red-600">🛡️ RoadGuard</h1>
          <p className="text-gray-400 mt-2">Authority Officer Portal</p>
        </div>
        <form onSubmit={handleSubmit} className="bg-gray-800 border border-gray-700 rounded-xl p-8 space-y-5">
          <h2 className="text-xl font-semibold text-white">Sign In</h2>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Email</label>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required placeholder="officer@authority.gov"
              className="w-full bg-gray-900 border border-gray-700 text-white rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-red-600" />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-1">Password</label>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required placeholder="••••••••"
              className="w-full bg-gray-900 border border-gray-700 text-white rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-red-600" />
          </div>
          {error && <p className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg px-4 py-2">{error}</p>}
          <button type="submit" disabled={loading} className="w-full bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white font-semibold py-2.5 rounded-lg transition">
            {loading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>
      </div>
    </div>
  );
};
export default LoginPage;`,

'src/pages/DashboardPage.jsx': `import { useEffect, useState } from 'react';
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
export default DashboardPage;`,

'src/pages/ViolationsPage.jsx': `import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/axios';
import Navbar from '../components/Navbar';
import StatusBadge from '../components/StatusBadge';
import FilterBar from '../components/FilterBar';
const PER_PAGE = 10;
const ViolationsPage = () => {
  const [violations, setViolations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filters, setFilters] = useState({ status: '', date: '' });
  const [page, setPage] = useState(1);
  const navigate = useNavigate();
  useEffect(() => {
    setLoading(true);
    const params = {};
    if (filters.status) params.status = filters.status;
    if (filters.date) params.date = filters.date;
    api.get('/authority/violations', { params }).then(res => { setViolations(res.data.data); setPage(1); }).catch(() => setError('Failed to load.')).finally(() => setLoading(false));
  }, [filters]);
  const totalPages = Math.ceil(violations.length / PER_PAGE);
  const paginated = violations.slice((page-1)*PER_PAGE, page*PER_PAGE);
  return (
    <div className="min-h-screen bg-gray-900">
      <Navbar />
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
                  <tr key={v._id} className="border-b border-gray-700/50 hover:bg-gray-700/30 transition">
                    <td className="px-6 py-3 font-mono text-white font-semibold">{v.carId}</td>
                    <td className="px-6 py-3 text-white">{v.calculatedSpeed} km/h</td>
                    <td className="px-6 py-3 text-gray-400">{v.location?.lat?.toFixed(3)}, {v.location?.lon?.toFixed(3)}</td>
                    <td className="px-6 py-3 text-gray-400">{new Date(v.detectedAt).toLocaleDateString()}</td>
                    <td className="px-6 py-3"><StatusBadge status={v.status} /></td>
                    <td className="px-6 py-3"><button onClick={() => navigate('/violations/' + v._id)} className="text-xs bg-gray-700 hover:bg-gray-600 text-white px-3 py-1.5 rounded transition">View Details</button></td>
                  </tr>
                ))}
                {paginated.length === 0 && <tr><td colSpan={6} className="px-6 py-8 text-center text-gray-500">No violations found</td></tr>}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="flex justify-center gap-2 mt-4">
              {Array.from({ length: totalPages }, (_, i) => i+1).map(p => (
                <button key={p} onClick={() => setPage(p)} className={"px-3 py-1 rounded text-sm " + (p===page ? 'bg-red-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600')}>{p}</button>
              ))}
            </div>
          )}
        </>}
      </div>
    </div>
  );
};
export default ViolationsPage;`,

'src/pages/ViolationDetailPage.jsx': `import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../api/axios';
import Navbar from '../components/Navbar';
import StatusBadge from '../components/StatusBadge';
const ViolationDetailPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [evidence, setEvidence] = useState(null);
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(true);
  const [actionMsg, setActionMsg] = useState('');
  const [error, setError] = useState('');
  useEffect(() => {
    api.get('/authority/evidence/' + id).then(res => { setEvidence(res.data.data); setStatus(res.data.data.status || 'pending'); }).catch(() => setError('Failed to load evidence.')).finally(() => setLoading(false));
  }, [id]);
  const handleAction = async (action) => {
    try {
      await api.post('/authority/violation/' + id + '/' + action);
      setStatus(action === 'verify' ? 'verified' : 'dismissed');
      setActionMsg(action === 'verify' ? '✅ Violation verified. Driver has been notified.' : '❌ Violation dismissed.');
    } catch { setError('Action failed.'); }
  };
  return (
    <div className="min-h-screen bg-gray-900">
      <Navbar />
      <div className="max-w-4xl mx-auto px-6 py-8">
        <button onClick={() => navigate('/violations')} className="text-gray-400 hover:text-white text-sm mb-6 flex items-center gap-1 transition">← Back to Violations</button>
        <h1 className="text-2xl font-bold text-white mb-6">Violation Details</h1>
        {loading && <p className="text-gray-400">Loading evidence...</p>}
        {error && <p className="text-red-400">{error}</p>}
        {!loading && evidence && <div className="space-y-6">
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-700"><h2 className="font-semibold text-white">Evidence Video</h2></div>
            <div className="p-4"><video controls src={evidence.video_clip_url} className="w-full rounded-lg bg-black max-h-80">Your browser does not support video.</video></div>
          </div>
          <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
            <div className="flex justify-between items-start mb-4"><h2 className="font-semibold text-white">Evidence Details</h2><StatusBadge status={status} /></div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div><p className="text-gray-400">License Plate</p><p className="text-white font-mono font-bold text-lg">{evidence.car_id_recognition}</p></div>
              <div><p className="text-gray-400">Speed</p><p className="text-white font-bold text-lg">{evidence.calculated_speed} km/h</p></div>
              <div><p className="text-gray-400">GPS</p><p className="text-white">{evidence.location?.lat?.toFixed(5)}, {evidence.location?.lon?.toFixed(5)}</p></div>
              <div><p className="text-gray-400">Detected At</p><p className="text-white">{new Date(evidence.detected_at).toLocaleString()}</p></div>
            </div>
          </div>
          {actionMsg ? (
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 text-center"><p className="text-white text-lg">{actionMsg}</p></div>
          ) : status === 'pending' ? (
            <div className="flex gap-4">
              <button onClick={() => handleAction('verify')} className="flex-1 bg-green-600 hover:bg-green-700 text-white font-semibold py-3 rounded-xl transition">✅ Verify Violation</button>
              <button onClick={() => handleAction('dismiss')} className="flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-3 rounded-xl transition">❌ Dismiss</button>
            </div>
          ) : (
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 text-center"><p className="text-gray-400">This violation has already been <strong className="text-white">{status}</strong>.</p></div>
          )}
        </div>}
      </div>
    </div>
  );
};
export default ViolationDetailPage;`,

'src/pages/SearchPage.jsx': `import { useState } from 'react';
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
export default SearchPage;`,

'README.md': `# RoadGuard Authority Web App
## Setup
\`\`\`bash
npm install
npm run dev
\`\`\`
Make sure the backend is running on http://localhost:5000 first.
Register an authority account via the backend, then log in here.`

};

for (const [filePath, content] of Object.entries(files)) {
  const dir = path.dirname(filePath);
  if (dir !== '.') fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(filePath, content, 'utf8');
  console.log('✅ ' + filePath);
}
console.log('\n🎉 Done! Run: npm run dev');
