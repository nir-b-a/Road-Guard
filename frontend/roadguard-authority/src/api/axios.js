import axios from 'axios';
// VITE_API_BASE lets the Docker build point the dashboard at a different API host.
// The fallback keeps `npm run dev` working exactly as before with no .env file.
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || 'http://localhost:5000/api',
});
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) config.headers.Authorization = 'Bearer ' + token;
  return config;
});
export default api;