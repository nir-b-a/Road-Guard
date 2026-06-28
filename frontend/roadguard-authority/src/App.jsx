import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import ViolationsPage from './pages/ViolationsPage';
import ViolationDetailPage from './pages/ViolationDetailPage';

const App = () => (
  <BrowserRouter>
    <AuthProvider>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/dashboard" element={<ProtectedRoute><DashboardPage /></ProtectedRoute>} />
        <Route path="/violations" element={<ProtectedRoute><ViolationsPage /></ProtectedRoute>} />
        <Route path="/violations/:id" element={<ProtectedRoute><ViolationDetailPage /></ProtectedRoute>} />
      </Routes>
    </AuthProvider>
  </BrowserRouter>
);

export default App;
