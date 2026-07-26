export function SkeletonCard() {
  return (
    <div className="rounded-md border-2 border-cave-950 bg-cave-800 p-6 shadow-panel">
      <div className="mb-4 h-4 w-1/3 rounded-sm bg-cave-700 mc-shimmer animate-shimmer" />
      <div className="mb-2 h-3 w-3/4 rounded-sm bg-cave-700 mc-shimmer animate-shimmer" />
      <div className="h-20 rounded-sm bg-cave-700 mc-shimmer animate-shimmer" />
    </div>
  );
}
