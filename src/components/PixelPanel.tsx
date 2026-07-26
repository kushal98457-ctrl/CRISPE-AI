import { HTMLAttributes, ReactNode } from "react";
import { motion } from "framer-motion";

interface PixelPanelProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  tone?: "stone" | "wood" | "cave";
  hover?: boolean;
}

const toneClasses: Record<NonNullable<PixelPanelProps["tone"]>, string> = {
  stone: "bg-gradient-to-b from-cave-700 to-cave-800 border-cave-950",
  wood: "bg-gradient-to-b from-wood-700 to-wood-900 border-wood-900",
  cave: "bg-cave-900/90 border-cave-950",
};

export function PixelPanel({
  children,
  tone = "stone",
  hover = false,
  className = "",
  ...rest
}: PixelPanelProps) {
  return (
    <motion.div
      {...(rest as any)}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={hover ? { y: -3 } : undefined}
      transition={{ duration: 0.25 }}
      className={[
        "relative rounded-md border-2 shadow-panel mc-texture-stone",
        toneClasses[tone],
        className,
      ].join(" ")}
    >
      {children}
    </motion.div>
  );
}
