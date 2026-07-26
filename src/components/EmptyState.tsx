import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";

export function EmptyState() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex flex-1 flex-col items-center justify-center gap-3 px-6 py-16 text-center"
    >
      <div className="mb-2 flex h-14 w-14 items-center justify-center rounded-sm border-2 border-cave-950 bg-cave-800 text-emerald-light shadow-block-sm">
        <Sparkles className="h-6 w-6" aria-hidden="true" />
      </div>
      <h2 className="m-0 font-pixel text-lg leading-snug sm:text-xl">Forge legendary prompts</h2>
      <p className="m-0 max-w-sm text-sm text-stone-300">
        Drop a raw idea in the box below and mine it into a full CRISPE-framework prompt.
      </p>
    </motion.div>
  );
}
