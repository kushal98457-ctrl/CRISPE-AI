import { ButtonHTMLAttributes, ReactNode } from "react";
import { motion } from "framer-motion";

interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  active?: boolean;
  label: string;
}

export function IconButton({ children, active, label, className = "", ...rest }: IconButtonProps) {
  return (
    <motion.button
      {...(rest as any)}
      whileTap={{ y: 1 }}
      title={label}
      aria-label={label}
      className={[
        "inline-flex items-center gap-1 rounded-sm px-2 py-1.5 text-xs font-semibold",
        "border-2 shadow-block-sm transition-colors duration-150",
        active
          ? "bg-emerald-dark/30 border-emerald text-emerald-light"
          : "bg-cave-800 border-cave-950 text-stone-300 hover:text-diamond hover:border-diamond/60",
        className,
      ].join(" ")}
    >
      {children}
    </motion.button>
  );
}
