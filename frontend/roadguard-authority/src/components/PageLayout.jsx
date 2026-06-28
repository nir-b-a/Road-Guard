import Navbar from './Navbar';
import hero from '../assets/hero.png';

const PageLayout = ({ children }) => (
  <div className="min-h-screen relative bg-gradient-to-br from-cyan-900/60 via-gray-900 to-gray-950">
    <img src={hero} alt="" className="fixed inset-0 w-full h-full object-cover opacity-20 pointer-events-none" />
    <div className="relative z-10">
      <Navbar />
      <div className="h-1 w-full bg-gradient-to-r from-cyan-700 via-cyan-500 to-cyan-700" />
      {children}
    </div>
  </div>
);
export default PageLayout;
