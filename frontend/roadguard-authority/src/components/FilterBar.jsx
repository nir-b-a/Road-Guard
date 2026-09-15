import { VIOLATION_TYPES } from '../utils/violations';
// The Status values are turned into the API query by ViolationsPage (STATUS_QUERIES).
// Type and Sort are applied in the browser by ViolationsPage.
const SELECT_CLASS = "bg-gray-800 border border-gray-700 text-gray-200 text-sm rounded px-3 py-2 focus:outline-none focus:border-cyan-500";
const FilterBar = ({ filters, onChange, sort, onSortChange }) => (
  <div className="flex flex-wrap gap-4 mb-6">
    <div className="flex flex-col gap-1">
      <label className="text-xs text-gray-400 uppercase">Status</label>
      <select value={filters.status} onChange={(e) => onChange({ ...filters, status: e.target.value })} className={SELECT_CLASS}>
        <option value="">All</option>
        <option value="pending">Pending (not reviewed yet)</option>
        <optgroup label="My reviews">
          <option value="verified_by_me">Verified by me</option>
          <option value="dismissed_by_me">Dismissed (rejected) by me</option>
        </optgroup>
        <optgroup label="All authorities">
          <option value="verified">Verified</option>
          <option value="dismissed">Dismissed</option>
        </optgroup>
      </select>
    </div>
    <div className="flex flex-col gap-1">
      <label className="text-xs text-gray-400 uppercase">Type</label>
      <select value={filters.type} onChange={(e) => onChange({ ...filters, type: e.target.value })} className={SELECT_CLASS}>
        <option value="">All types</option>
        {Object.entries(VIOLATION_TYPES).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
      </select>
    </div>
    <div className="flex flex-col gap-1">
      <label className="text-xs text-gray-400 uppercase">Date</label>
      <input type="date" value={filters.date} onChange={(e) => onChange({ ...filters, date: e.target.value })} className={SELECT_CLASS} />
    </div>
    <div className="flex flex-col gap-1">
      <label className="text-xs text-gray-400 uppercase">License Plate</label>
      <input type="text" value={filters.plate} onChange={(e) => onChange({ ...filters, plate: e.target.value })} placeholder="e.g. ABC-1234" className={SELECT_CLASS} />
    </div>
    <div className="flex flex-col gap-1">
      <label className="text-xs text-gray-400 uppercase">Sort By</label>
      <select value={sort} onChange={(e) => onSortChange(e.target.value)} className={SELECT_CLASS}>
        <option value="">Newest first</option>
        <option value="priority_desc">Priority: High → Low</option>
        <option value="priority_asc">Priority: Low → High</option>
      </select>
    </div>
    {(filters.status || filters.type || filters.date || filters.plate) && (
      <div className="flex flex-col justify-end">
        <button onClick={() => onChange({ status: '', type: '', date: '', plate: '' })} className="text-xs text-gray-400 hover:text-cyan-400 transition py-2">✕ Clear</button>
      </div>
    )}
  </div>
);
export default FilterBar;
