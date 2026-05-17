import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../api/axios';

const LoginPage = () => {
  const [tab, setTab] = useState('login');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [inviteCode, setInviteCode] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const endpoint = tab === 'login' ? '/auth/login' : '/auth/register';
      const body = tab === 'login'
        ? { email, password }
        : { name, email, password, role: 'authority', inviteCode };
      const res = await api.post(endpoint, body);
      if (res.data.success) {
        const { token, ...user } = res.data.data;
        if (user.role !== 'authority') {
          setError('Access denied. This portal is for authority officers only.');
          setLoading(false);
          return;
        }
        login(token, user);
        navigate('/dashboard');
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center px-4">
      <div className="w-full max-w-md bg-gray-900 rounded-2xl p-8 shadow-xl">
        <h1 className="text-3xl font-bold text-white text-center mb-2">🛡️ RoadGuard</h1>
        <p className="text-gray-400 text-center mb-8">Authority Officer Portal</p>

        <div className="flex mb-6 bg-gray-800 rounded-lg p-1">
          <button onClick={() => setTab('login')}
            className={`flex-1 py-2 rounded-md text-sm font-medium transition ${tab === 'login' ? 'bg-red-600 text-white' : 'text-gray-400 hover:text-white'}`}>
            Login
          </button>
          <button onClick={() => setTab('register')}
            className={`flex-1 py-2 rounded-md text-sm font-medium transition ${tab === 'register' ? 'bg-red-600 text-white' : 'text-gray-400 hover:text-white'}`}>
            Register
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {tab === 'register' && (
            <>
              <input
                type="text"
                placeholder="Full Name"
                value={name}
                onChange={e => setName(e.target.value)}
                className="w-full bg-gray-800 text-white rounded-lg px-4 py-3 outline-none focus:ring-2 focus:ring-red-500"
                required
              />
              <input
                type="text"
                placeholder="Invite Code"
                value={inviteCode}
                onChange={e => setInviteCode(e.target.value)}
                className="w-full bg-gray-800 text-white rounded-lg px-4 py-3 outline-none focus:ring-2 focus:ring-red-500"
                required
              />
            </>
          )}
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            className="w-full bg-gray-800 text-white rounded-lg px-4 py-3 outline-none focus:ring-2 focus:ring-red-500"
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            className="w-full bg-gray-800 text-white rounded-lg px-4 py-3 outline-none focus:ring-2 focus:ring-red-500"
            required
          />
          {error && <p className="text-red-400 text-sm">{error}</p>}
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-red-600 hover:bg-red-700 text-white font-semibold py-3 rounded-lg transition disabled:opacity-50">
            {loading ? 'Please wait...' : tab === 'login' ? 'Login' : 'Create Account'}
          </button>
        </form>
      </div>
    </div>
  );
};

export default LoginPage;