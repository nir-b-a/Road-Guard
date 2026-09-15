// Priority is not stored on the violation; it is derived here so the Dashboard and the
// All Violations list rank violations the same way.
export const getPriority = (v) => {
  if (v.calculatedSpeed >= 120) return { label: 'High', style: 'bg-red-500/20 text-red-400 border-red-500/30' };
  if (v.violationType === 'lane_crossing') return { label: 'Medium', style: 'bg-orange-500/20 text-orange-400 border-orange-500/30' };
  return { label: 'Low', style: 'bg-gray-500/20 text-gray-400 border-gray-500/30' };
};
export const PRIORITY_ORDER = { High: 0, Medium: 1, Low: 2 };
// Matches the violationType enum in backend/models/Violation.js.
export const VIOLATION_TYPES = { speeding: 'Speeding', lane_crossing: 'Lane Crossing' };
