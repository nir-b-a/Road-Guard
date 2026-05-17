const StatusBadge = ({ status }) => {
  const styles = {
    pending:   'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30',
    verified:  'bg-green-500/20 text-green-400 border border-green-500/30',
    dismissed: 'bg-red-500/20 text-red-400 border border-red-500/30',
  };
  return <span className={`px-2.5 py-1 rounded-full text-xs font-semibold uppercase ${styles[status] || styles.pending}`}>{status}</span>;
};
export default StatusBadge;