import { InputHTMLAttributes, forwardRef } from "react";

interface TextFieldProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
}

export const TextField = forwardRef<HTMLInputElement, TextFieldProps>(
  ({ label, id, className = "", ...rest }, ref) => {
    return (
      <label htmlFor={id} className="block mb-4">
        <span className="block mb-1.5 text-sm font-semibold text-iron/90">{label}</span>
        <input
          ref={ref}
          id={id}
          {...rest}
          className={[
            "w-full box-border rounded-sm px-3 py-2.5 text-sm",
            "bg-cave-900 text-iron placeholder:text-stone-300/60",
            "border-2 border-cave-950 shadow-block-pressed",
            "transition-shadow duration-150",
            "focus:outline-none focus:shadow-glow focus:border-diamond",
            className,
          ].join(" ")}
        />
      </label>
    );
  }
);

TextField.displayName = "TextField";
