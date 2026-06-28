const FilterBar = ({ filters, onChange }) => (
  <div className="flex flex-wrap gap-4 mb-6">
    <div className="flex flex-col gap-1">
      <label className="text-xs text-gray-400 uppercase">Status</label>
      <select value={filters.status} onChange={(e) => onChange({ ...filters, status: e.target.value })}
        className="bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded px-3 py-2 focus:outline-none focus:border-cyan-500">
        <option value="">All</option>
        <option value="pending">Pending</option>
        <option value="verified">Verified</option>
        <option value="dismissed">Dismissed</option>
      </select>
    </div>
    <div className="flex flex-col gap-1">
      <label className="text-xs text-gray-400 uppercase">Date</label>
      <input type="date" value={filters.date} onChange={(e) => onChange({ ...filters, date: e.target.value })}
        className="bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded px-3 py-2 focus:outline-none focus:border-cyan-500" />
    </div>
    <div className="flex flex-col gap-1">
      <label className="text-xs text-gray-400 uppercase">License Plate</label>
      <input type="text" value={filters.plate} onChange={(e) => onChange({ ...filters, plate: e.target.value })} placeholder="e.g. ABC-1234"
        className="bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded px-3 py-2 focus:outline-none focus:border-cyan-500" />
    </div>
    {(filters.status || filters.date || filters.plate) && (
      <div className="flex flex-col justify-end">
        <button onClick={() => onChange({ status: '', date: '', plate: '' })} className="text-xs text-gray-400 hover:text-cyan-400 transition py-2">✕ Clear</button>
      </div>
    )}
  </div>
);
export default FilterBar;