import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle2 } from "lucide-react";

interface ToastProps {
  message: string | null;
}

export function Toast({ message }: ToastProps) {
  return (
    <div className="fixed bottom-6 right-6 z-[1000]" role="status" aria-live="polite">
      <AnimatePresence>
        {message && (
          <motion.div
            initial={{ opacity: 0, y: 16, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.9 }}
            transition={{ duration: 0.2 }}
            className="flex items-center gap-2 rounded-md border-2 border-emerald-dark bg-gradient-to-b from-emerald-light to-emerald px-4 py-3 font-semibold text-cave-950 shadow-panel"
          >
            <CheckCircle2 className="w-4 h-4 shrink-0" aria-hidden="true" />
            <span className="text-sm">{message}</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
