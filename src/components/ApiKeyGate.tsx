import { useState } from "react";
import { motion } from "framer-motion";
import { Pickaxe, KeyRound, ExternalLink, AlertTriangle } from "lucide-react";

interface ApiKeyGateProps {
  onSetKey: (key: string) => void;
}

export function ApiKeyGate({ onSetKey }: ApiKeyGateProps) {
  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed) {
      setError("Please enter an API key.");
      return;
    }
    setError(null);
    onSetKey(trimmed);
  };

  return (
    <div className="mc-bg flex min-h-screen items-center justify-center p-4 text-iron">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md rounded-md border-2 border-cave-950 bg-cave-900 p-6 shadow-panel sm:p-8"
      >
        <div className="mb-6 flex flex-col items-center gap-3 text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-sm border-2 border-cave-950 bg-cave-800 text-diamond shadow-block-sm">
            <Pickaxe className="h-6 w-6" />
          </div>
          <h1 className="font-pixel text-lg leading-snug text-iron">
            CRISPE FORGE
          </h1>
          <p className="max-w-xs text-sm text-stone-300">
            Enter your Gemini API key to start forging prompts.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div>
            <label
              htmlFor="api-key"
              className="mb-1.5 block text-xs font-semibold text-iron/90"
            >
              Gemini API Key
            </label>
            <input
              id="api-key"
              type="password"
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                if (error) setError(null);
              }}
              placeholder="AIzaSy..."
              required
              className="w-full rounded-sm border-2 border-cave-950 bg-cave-800 px-3 py-2.5 text-sm text-iron placeholder:text-stone-300/60 focus:outline-none focus:shadow-glow focus:border-diamond"
            />
            {error && (
              <motion.p
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                className="mt-1.5 flex items-center gap-1.5 text-xs text-redstone-light"
              >
                <AlertTriangle className="h-3 w-3 shrink-0" />
                {error}
              </motion.p>
            )}
          </div>

          <motion.button
            type="submit"
            whileTap={{ scale: 0.96 }}
            className="flex items-center justify-center gap-2 rounded-sm border-b-4 border-diamond-dark bg-gradient-to-b from-diamond-light to-diamond px-5 py-2.5 text-sm font-bold text-cave-950 transition-[filter] hover:brightness-110"
          >
            <KeyRound className="h-4 w-4" />
            Enter the Forge
          </motion.button>
        </form>

        <div className="mt-5 text-center">
          <a
            href="https://aistudio.google.com/apikey"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-xs text-stone-300 underline underline-offset-2 hover:text-diamond transition-colors"
          >
            Get a free Gemini API key
            <ExternalLink className="h-3 w-3" />
          </a>
        </div>
      </motion.div>
    </div>
  );
}
