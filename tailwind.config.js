/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        cave: {
          950: "#0b0d0f",
          900: "#121417",
          800: "#1a1d21",
          700: "#24282d",
          600: "#31363d",
        },
        stone: {
          900: "#2b2d2f",
          700: "#4a4d4f",
          500: "#6f7274",
          300: "#9a9d9f",
        },
        deepslate: {
          900: "#26282b",
          700: "#3a3d40",
        },
        wood: {
          900: "#4a3221",
          700: "#6b4a2f",
          500: "#8a6339",
          300: "#a9803f",
        },
        emerald: {
          DEFAULT: "#17c964",
          light: "#4ade80",
          dark: "#0f8a45",
        },
        diamond: {
          DEFAULT: "#4fd8e0",
          light: "#8ff0f5",
          dark: "#1f9aa3",
        },
        gold: {
          DEFAULT: "#f2c94c",
          light: "#ffe58a",
          dark: "#b8901f",
        },
        redstone: {
          DEFAULT: "#e64545",
          light: "#ff7a7a",
          dark: "#9c2323",
        },
        nether: {
          DEFAULT: "#e8752c",
          light: "#ff9d5c",
          dark: "#a3491a",
        },
        iron: {
          DEFAULT: "#d8d8d4",
          dark: "#8c8c86",
        },
      },
      fontFamily: {
        pixel: ["'Press Start 2P'", "monospace"],
        body: ["Inter", "system-ui", "sans-serif"],
      },
      boxShadow: {
        block:
          "inset 3px 3px 0 rgba(255,255,255,0.12), inset -3px -3px 0 rgba(0,0,0,0.35), 0 6px 0 rgba(0,0,0,0.45), 0 10px 16px rgba(0,0,0,0.4)",
        "block-sm":
          "inset 2px 2px 0 rgba(255,255,255,0.12), inset -2px -2px 0 rgba(0,0,0,0.35), 0 3px 0 rgba(0,0,0,0.45)",
        "block-pressed":
          "inset 3px 3px 0 rgba(0,0,0,0.35), inset -3px -3px 0 rgba(255,255,255,0.08), 0 1px 0 rgba(0,0,0,0.45)",
        panel: "0 2px 0 rgba(0,0,0,0.5), 0 12px 30px rgba(0,0,0,0.55)",
        glow: "0 0 0 3px rgba(79,216,224,0.35)",
      },
      keyframes: {
        blockPop: {
          "0%": { transform: "scale(0.85)", opacity: "0" },
          "60%": { transform: "scale(1.04)", opacity: "1" },
          "100%": { transform: "scale(1)", opacity: "1" },
        },
        floatUp: {
          "0%": { transform: "translateY(8px)", opacity: "0" },
          "100%": { transform: "translateY(0)", opacity: "1" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        blockPop: "blockPop 0.28s ease-out",
        floatUp: "floatUp 0.35s ease-out",
        shimmer: "shimmer 1.6s linear infinite",
      },
    },
  },
  plugins: [],
};
