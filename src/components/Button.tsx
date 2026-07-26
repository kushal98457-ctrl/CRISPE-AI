import { ButtonHTMLAttributes, forwardRef } from "react";
import { motion } from "framer-motion";
import { Loader2 } from "lucide-react";

export type ButtonVariant =
  | "grass"
  | "stone"
  | "diamond"
  | "emerald"
  | "gold"
  | "redstone"
  | "ghost"
  | "outline"
  | "icon";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  isLoading?: boolean;
  variant?: ButtonVariant;
  size?: "sm" | "md" | "lg";
}

const variantClasses: Record<ButtonVariant, string> = {
  grass:
    "bg-gradient-to-b from-emerald-light to-emerald-dark text-cave-950 border-b-4 border-emerald-dark hover:brightness-110",
  stone:
    "bg-gradient-to-b from-stone-500 to-stone-900 text-iron border-b-4 border-cave-950 hover:brightness-110",
  diamond:
    "bg-gradient-to-b from-diamond-light to-diamond-dark text-cave-950 border-b-4 border-diamond-dark hover:brightness-110",
  emerald:
    "bg-gradient-to-b from-emerald-light to-emerald text-cave-950 border-b-4 border-emerald-dark hover:brightness-110",
  gold: "bg-gradient-to-b from-gold-light to-gold-dark text-cave-950 border-b-4 border-gold-dark hover:brightness-110",
  redstone:
    "bg-gradient-to-b from-redstone-light to-redstone-dark text-white border-b-4 border-redstone-dark hover:brightness-110",
  ghost: "bg-white/5 text-iron border-b-4 border-transparent hover:bg-white/10",
  outline:
    "bg-transparent text-iron border-2 border-stone-500 hover:border-diamond hover:text-diamond",
  icon: "bg-cave-800 text-iron border-b-2 border-cave-950 hover:text-diamond",
};

const sizeClasses: Record<NonNullable<ButtonProps["size"]>, string> = {
  sm: "px-3 py-1.5 text-xs gap-1.5",
  md: "px-5 py-2.5 text-sm gap-2",
  lg: "px-7 py-3.5 text-base gap-2.5",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ isLoading, children, disabled, variant = "emerald", size = "md", className = "", ...rest }, ref) => {
    const isDisabled = disabled || isLoading;

    return (
      <motion.button
        ref={ref}
        {...(rest as any)}
        disabled={isDisabled}
        whileTap={isDisabled ? undefined : { y: 2, boxShadow: "none" }}
        className={[
          "relative inline-flex items-center justify-center font-pixel uppercase tracking-tight rounded-sm",
          "shadow-block-sm transition-[filter,transform] duration-100 select-none",
          "focus-visible:outline focus-visible:outline-2 focus-visible:outline-diamond focus-visible:outline-offset-2",
          sizeClasses[size],
          isDisabled ? "opacity-50 cursor-not-allowed grayscale" : "cursor-pointer active:shadow-block-pressed",
          variantClasses[variant],
          className,
        ].join(" ")}
      >
        {isLoading && <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />}
        <span className="font-body font-semibold normal-case tracking-normal">
          {isLoading ? "Generating…" : children}
        </span>
      </motion.button>
    );
  }
);

Button.displayName = "Button";
