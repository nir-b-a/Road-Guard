import { useEffect, useState } from 'react';
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
    api.get('/authority/evidence/' + id)
      .then(res => {
        setEvidence(res.data.data);
        setStatus(res.data.data.status || 'pending');
      })
      .catch(() => setError('Failed to load evidence.'))
      .finally(() => setLoading(false));
  }, [id]);

  const handleAction = async (action) => {
    try {
      await api.post('/authority/violation/' + id + '/' + action);
      setStatus(action === 'verify' ? 'verified' : 'dismissed');
      setActionMsg(action === 'verify'
        ? 'Violation verified. Driver has been notified.'
        : 'Violation dismissed. Evidence removed from storage.');
    } catch {
      setError('Action failed.');
    }
  };

  const displayPlate = evidence
    ? (evidence.plate_text || 'Unknown')
    : null;

  const displaySpeed = evidence?.calculated_speed
    ? `${evidence.calculated_speed} km/h`
    : '—';

  const displayDate = evidence?.recorded_at
    ? new Date(evidence.recorded_at).toLocaleString()
    : evidence?.detected_at
      ? new Date(evidence.detected_at).toLocaleString()
      : '—';

  return (
    <div className="min-h-screen bg-gray-900">
      <Navbar />
      <div className="max-w-4xl mx-auto px-6 py-8">
        <button
          onClick={() => navigate('/violations')}
          className="text-gray-400 hover:text-white text-sm mb-6 flex items-center gap-1 transition"
        >
          &larr; Back to Violations
        </button>
        <h1 className="text-2xl font-bold text-white mb-6">Violation Details</h1>

        {loading && <p className="text-gray-400">Loading evidence...</p>}
        {error && <p className="text-red-400">{error}</p>}

        {!loading && evidence && (
          <div className="space-y-6">

            {/* Evidence video — presigned R2 URL, playable directly */}
            <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-700">
                <h2 className="font-semibold text-white">Evidence Video</h2>
              </div>
              <div className="p-4">
                {evidence.video_clip_url ? (
                  <video
                    controls
                    src={evidence.video_clip_url}
                    className="w-full rounded-lg bg-black max-h-80"
                  >
                    Your browser does not support video.
                  </video>
                ) : (
                  <p className="text-gray-500 text-sm">No video available.</p>
                )}
              </div>
            </div>

            {/* License plate crop — shown when FastALPR collected a plate image */}
            {evidence.plate_url && (
              <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
                <div className="px-6 py-4 border-b border-gray-700">
                  <h2 className="font-semibold text-white">License Plate Evidence</h2>
                </div>
                <div className="p-4 flex items-center gap-6">
                  <img
                    src={evidence.plate_url}
                    alt="License plate crop"
                    className="max-h-20 rounded border border-gray-600 bg-black"
                  />
                  <div>
                    <p className="text-gray-400 text-xs mb-1">Best captured plate</p>
                    <p className="text-white font-mono text-2xl font-bold tracking-widest">
                      {displayPlate}
                    </p>
                    {!evidence.plate_text && (
                      <p className="text-yellow-400 text-xs mt-1">
                        OCR could not read — manual review required
                      </p>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Evidence metadata grid */}
            <div className="bg-gray-800 border border-gray-700 rounded-xl p-6">
              <div className="flex justify-between items-start mb-4">
                <h2 className="font-semibold text-white">Evidence Details</h2>
                <StatusBadge status={status} />
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-400">License Plate</p>
                  <p className="text-white font-mono font-bold text-lg">{displayPlate}</p>
                </div>
                <div>
                  <p className="text-gray-400">Speed</p>
                  <p className="text-white font-bold text-lg">{displaySpeed}</p>
                </div>
                <div>
                  <p className="text-gray-400">Violation Type</p>
                  <p className="text-white">
                    {evidence.violation_type ? evidence.violation_type.replace(/_/g, ' ') : '—'}
                  </p>
                </div>
                <div>
                  <p className="text-gray-400">Confidence</p>
                  <p className="text-white">
                    {evidence.confidence != null
                      ? (evidence.confidence * 100).toFixed(1) + '%'
                      : '—'}
                  </p>
                </div>
                <div>
                  <p className="text-gray-400">GPS Location</p>
                  <p className="text-white">
                    {evidence.location?.lat
                      ? `${evidence.location.lat.toFixed(5)}, ${evidence.location.lon.toFixed(5)}`
                      : '—'}
                  </p>
                </div>
                <div>
                  <p className="text-gray-400">
                    {evidence.recorded_at ? 'Recorded At (GPS)' : 'Detected At'}
                  </p>
                  <p className="text-white">{displayDate}</p>
                </div>
              </div>
            </div>

            {/* Action buttons */}
            {actionMsg ? (
              <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 text-center">
                <p className="text-white text-lg">{actionMsg}</p>
              </div>
            ) : status === 'pending' ? (
              <div className="flex gap-4">
                <button
                  onClick={() => handleAction('verify')}
                  className="flex-1 bg-green-600 hover:bg-green-700 text-white font-semibold py-3 rounded-xl transition"
                >
                  Verify Violation
                </button>
                <button
                  onClick={() => handleAction('dismiss')}
                  className="flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-3 rounded-xl transition"
                >
                  Dismiss
                </button>
              </div>
            ) : (
              <div className="bg-gray-800 border border-gray-700 rounded-xl p-6 text-center">
                <p className="text-gray-400">
                  This violation has already been <strong className="text-white">{status}</strong>.
                </p>
              </div>
            )}

          </div>
        )}
      </div>
    </div>
  );
};

export default ViolationDetailPage;
