import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import api from '../api/axios';
import { PhotoIcon } from './icons';

// The picture of the violating car's plate, next to the plate number. The link is asked for on
// click rather than loaded with the list: it is a short-lived presigned R2 URL, and a page can
// stay open longer than that.
const PlateImageButton = ({ violationId, plate }) => {
  const [open, setOpen] = useState(false);
  const [url, setUrl] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  const show = async (e) => {
    e.stopPropagation();   // the button sits inside table rows that navigate on click
    setOpen(true);
    setUrl(null);
    setError('');
    try {
      const res = await api.get('/authority/violation/' + violationId + '/plate-image');
      setUrl(res.data.data.plate_image_url);
    } catch {
      setError('Could not load the plate picture.');
    }
  };

  // Clicks inside a portal still bubble through the React tree, up to that same row.
  const close = (e) => { e.stopPropagation(); setOpen(false); };

  return (
    <>
      <button type="button" onClick={show} title="Show the license plate picture"
        className="inline-flex items-center gap-1 font-sans font-normal text-xs bg-gray-700 hover:bg-gray-600 text-gray-200 px-2 py-1 rounded transition">
        <PhotoIcon className="w-3.5 h-3.5" />Photo
      </button>
      {open && createPortal(
        <div onClick={close} className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4">
          <div onClick={(e) => e.stopPropagation()} role="dialog" aria-modal="true" aria-label="License plate picture"
            className="w-full max-w-lg bg-gray-800 border border-gray-700 rounded-xl">
            <div className="flex items-center justify-between px-5 py-3 border-b border-gray-700">
              <h2 className="font-semibold text-white">License plate <span className="font-mono ml-1">{plate}</span></h2>
              <button type="button" onClick={close} aria-label="Close" className="text-gray-400 hover:text-white transition">✕</button>
            </div>
            <div className="p-5">
              {error && <p className="text-red-400 text-sm">{error}</p>}
              {!error && !url && <p className="text-gray-400 text-sm">Loading picture...</p>}
              {!error && url && <>
                <img src={url} alt={'License plate ' + plate} onError={() => setError('The plate picture could not be displayed.')}
                  className="w-full rounded-lg bg-black" />
                <a href={url} target="_blank" rel="noreferrer" className="inline-block mt-3 text-sm text-cyan-400 hover:text-cyan-300">Open original ↗</a>
              </>}
            </div>
          </div>
        </div>,
        document.body
      )}
    </>
  );
};
export default PlateImageButton;
