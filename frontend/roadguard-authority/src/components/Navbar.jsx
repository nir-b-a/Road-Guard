import { Link, useNavigate } from 'react-router-dom';
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
          <Link to="/drives" className="hover:text-white transition">Drives</Link>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <span className="text-gray-400 text-sm">{user?.name || 'Officer'}</span>
        <button onClick={handleLogout} className="bg-red-600 hover:bg-red-700 text-white text-sm px-4 py-1.5 rounded transition">Logout</button>
      </div>
    </nav>
  );
};

export default Navbar;
