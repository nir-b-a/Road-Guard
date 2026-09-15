import { getPriority, VIOLATION_TYPES } from '../utils/violations';
export const PriorityBadge = ({ violation }) => {
  const priority = getPriority(violation);
  return <span className={"px-2.5 py-1 rounded-full text-xs font-semibold uppercase border " + priority.style}>{priority.label}</span>;
};
export const TypeBadge = ({ type }) => (
  <span className="px-2.5 py-1 rounded-full text-xs font-semibold uppercase whitespace-nowrap bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">{VIOLATION_TYPES[type] || VIOLATION_TYPES.speeding}</span>
);
