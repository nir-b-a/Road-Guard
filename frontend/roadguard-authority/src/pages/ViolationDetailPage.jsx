import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../api/axios';
import PageLayout from '../components/PageLayout';
import StatusBadge from '../components/StatusBadge';
const ViolationDetailPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [evidence, setEvidence] = useState(null);
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(true);
  const [actionMsg, setActionMsg] = useState('');
  const [error, setError] = useState('');
  // A <video> whose codec the browser cannot decode fails silently: black box, live
  // controls, nothing in the console. Surface it, otherwise an unplayable clip is
  // indistinguishable from a missing one.
  const [videoError, setVideoError] = useState(false);
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
    <PageLayout>
      <div className="max-w-4xl mx-auto px-6 py-8">
        <button onClick={() => navigate('/violations')} className="text-gray-400 hover:text-white text-sm mb-6 flex items-center gap-1 transition">← Back to Violations</button>
        <h1 className="text-2xl font-bold text-white mb-6">Violation Details</h1>
        {loading && <p className="text-gray-400">Loading evidence...</p>}
        {error && <p className="text-red-400">{error}</p>}
        {!loading && evidence && <div className="space-y-6">
          <div className="bg-gray-800 border border-gray-700 rounded-xl overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-700"><h2 className="font-semibold text-white">Evidence Video</h2></div>
            <div className="p-4">
              <video controls src={evidence.video_clip_url} onError={() => setVideoError(true)} className="w-full rounded-lg bg-black max-h-80">Your browser does not support video.</video>
              {videoError && <div className="mt-3 rounded-lg border border-amber-600/40 bg-amber-950/40 px-4 py-3 text-sm text-amber-200">
                <p className="font-semibold">This clip could not be played in the browser.</p>
                <p className="mt-1 text-amber-300/80">The evidence exists but is in a format this browser cannot decode — it was most likely processed by a worker without ffmpeg, so it was never transcoded to H.264. <a href={evidence.video_clip_url} className="underline hover:text-amber-100">Download the clip</a> to review it in a desktop player.</p>
              </div>}
            </div>
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
    </PageLayout>
  );
};
export default ViolationDetailPage;