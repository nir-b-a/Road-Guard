import { useNavigate } from 'react-router-dom';
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
export default ViolationCard;