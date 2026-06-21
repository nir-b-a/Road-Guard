import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { ShieldIcon } from './icons';

const Navbar = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const handleLogout = () => { logout(); navigate('/login'); };

  return (
    <nav className="bg-gray-800 border-b border-gray-700 px-6 py-4 flex items-center justify-between">
      <div className="flex items-center gap-8">
        <span className="flex items-center gap-2 text-cyan-400 font-bold text-xl">
          <ShieldIcon className="w-6 h-6" />
          RoadGuard
        </span>
        <div className="flex gap-6 text-sm text-gray-300">
          <Link to="/dashboard" className="hover:text-white transition">Dashboard</Link>
          <Link to="/violations" className="hover:text-white transition">Violations</Link>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <span className="text-gray-400 text-sm">{user?.name || 'Officer'}</span>
        <button onClick={handleLogout} className="bg-cyan-600 hover:bg-cyan-700 text-white text-sm px-4 py-1.5 rounded transition">Logout</button>
      </div>
    </nav>
  );
};

export default Navbar;
