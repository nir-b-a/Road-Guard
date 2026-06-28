import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../api/axios';
import hero from '../assets/hero.png';
import { ShieldIcon } from '../components/icons';

const MailIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M21.75 6.75v10.5a2.25 2.25 0 0 1-2.25 2.25h-15a2.25 2.25 0 0 1-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0 0 19.5 4.5h-15a2.25 2.25 0 0 0-2.25 2.25m19.5 0v.243a2.25 2.25 0 0 1-1.07 1.916l-7.5 4.615a2.25 2.25 0 0 1-2.36 0L3.32 8.91a2.25 2.25 0 0 1-1.07-1.916V6.75" />
  </svg>
);

const LockIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 1 0-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 0 0 2.25-2.25v-6.75a2.25 2.25 0 0 0-2.25-2.25H6.75a2.25 2.25 0 0 0-2.25 2.25v6.75a2.25 2.25 0 0 0 2.25 2.25Z" />
  </svg>
);

const UserIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M17.982 18.725A7.488 7.488 0 0 0 12 15.75a7.488 7.488 0 0 0-5.982 2.975m11.963 0a9 9 0 1 0-11.963 0m11.963 0A8.966 8.966 0 0 1 12 21a8.966 8.966 0 0 1-5.982-2.275M15 9.75a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
  </svg>
);

const KeyIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 5.25a3 3 0 0 1 3 3m3 0a6 6 0 0 1-7.029 5.912c-.563-.097-1.159.026-1.563.43L10.5 17.25H8.25v2.25H6v2.25H2.25v-2.818c0-.597.237-1.17.659-1.591l6.499-6.499c.404-.404.527-1 .43-1.563A6 6 0 1 1 21.75 8.25Z" />
  </svg>
);

const EyeIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
  </svg>
);

const EyeOffIcon = (props) => (
  <svg {...props} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.98 8.223A10.477 10.477 0 0 0 1.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0 1 12 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 0 1-4.293 5.774M6.228 6.228 3 3m3.228 3.228 3.65 3.65m7.894 7.894L21 21m-3.228-3.228-3.65-3.65m0 0a3 3 0 1 0-4.243-4.243m4.242 4.242L9.88 9.88" />
  </svg>
);

const InputField = ({ icon: Icon, ...props }) => (
  <div className="relative">
    <Icon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500 pointer-events-none" />
    <input
      {...props}
      className="w-full bg-gray-800 text-white rounded-lg pl-10 pr-4 py-3 outline-none focus:ring-2 focus:ring-cyan-500"
    />
  </div>
);

const LoginPage = () => {
  const [tab, setTab] = useState('login');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [inviteCode, setInviteCode] = useState('');
  const [showPassword, setShowPassword] = useState(false);
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
    <div className="min-h-screen flex bg-gray-950">
      <div className="hidden lg:flex lg:w-1/2 relative items-center justify-center overflow-hidden bg-gradient-to-br from-cyan-950 via-gray-950 to-gray-950">
        <img src={hero} alt="" className="absolute inset-0 w-full h-full object-cover opacity-20" />
        <div className="relative z-10 text-center px-12">
          <h1 className="flex items-center justify-center gap-3 text-5xl font-bold text-white mb-4">
            <ShieldIcon className="w-12 h-12 text-cyan-400" />
            RoadGuard
          </h1>
          <p className="text-gray-300 text-lg">Keeping our roads safe, together.</p>
        </div>
      </div>

      <div className="flex-1 flex items-center justify-center px-4">
        <div className="w-full max-w-md bg-gray-900/90 backdrop-blur-xl rounded-2xl shadow-2xl shadow-cyan-950/40 ring-1 ring-white/10 overflow-hidden">
          <div className="h-1.5 w-full bg-gradient-to-r from-cyan-700 via-cyan-500 to-cyan-700" />
          <div className="p-8">
            <h1 className="flex items-center justify-center gap-2 text-3xl font-bold text-white mb-2 lg:hidden">
              <ShieldIcon className="w-8 h-8 text-cyan-400" />
              RoadGuard
            </h1>
            <p className="text-gray-400 text-center mb-8">Authority Officer Portal</p>

            <div className="mb-6 bg-gray-800 rounded-lg p-1">
              <div className="relative flex gap-1">
                <div
                  className={`absolute inset-y-0 left-0 w-1/2 bg-cyan-600 rounded-md transition-transform duration-300 ease-out ${tab === 'register' ? 'translate-x-full' : 'translate-x-0'}`}
                />
                <button onClick={() => setTab('login')}
                  className={`relative z-10 flex-1 py-2 rounded-md text-sm font-medium transition ${tab === 'login' ? 'text-white' : 'text-gray-400 hover:text-white'}`}>
                  Login
                </button>
                <button onClick={() => setTab('register')}
                  className={`relative z-10 flex-1 py-2 rounded-md text-sm font-medium transition ${tab === 'register' ? 'text-white' : 'text-gray-400 hover:text-white'}`}>
                  Register
                </button>
              </div>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {tab === 'register' && (
                <>
                  <InputField
                    icon={UserIcon}
                    type="text"
                    placeholder="Full Name"
                    value={name}
                    onChange={e => setName(e.target.value)}
                    required
                  />
                  <InputField
                    icon={KeyIcon}
                    type="text"
                    placeholder="Invite Code"
                    value={inviteCode}
                    onChange={e => setInviteCode(e.target.value)}
                    required
                  />
                </>
              )}
              <InputField
                icon={MailIcon}
                type="email"
                placeholder="Email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
              />
              <div className="relative">
                <LockIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-500 pointer-events-none" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  placeholder="Password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  className="w-full bg-gray-800 text-white rounded-lg pl-10 pr-10 py-3 outline-none focus:ring-2 focus:ring-cyan-500"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(s => !s)}
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300">
                  {showPassword ? <EyeOffIcon className="w-5 h-5" /> : <EyeIcon className="w-5 h-5" />}
                </button>
              </div>
              {error && <p className="text-red-400 text-sm">{error}</p>}
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-semibold py-3 rounded-lg transition disabled:opacity-50">
                {loading ? 'Please wait...' : tab === 'login' ? 'Login' : 'Create Account'}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
