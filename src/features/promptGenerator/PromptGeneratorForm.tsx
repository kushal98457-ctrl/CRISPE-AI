import { forwardRef, useImperativeHandle, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, ArrowUp, Loader2 } from "lucide-react";
import { generatePrompt } from "../../api/promptApi";
import { usePromptStore } from "../../store/promptStore";

const STATUS_MESSAGES = [
  "Thinking...",
  "Understanding your request...",
  "Analyzing context...",
  "Crafting CRISPE framework...",
  "Structuring output...",
  "Polishing response...",
  "Finalizing...",
];

const VILLAGER_SOUND =
  "https://www.myinstants.com/media/sounds/yes1.mp3";

const TYPEWRITER_INTERVAL_MS = 15;
const TYPEWRITER_CHARS_PER_TICK = 2;

interface PromptGeneratorFormProps {
  showToast: (message: string) => void;
}

export interface PromptComposerHandle {
  reset: () => void;
}

export const PromptGeneratorForm = forwardRef<PromptComposerHandle, PromptGeneratorFormProps>(
  ({ showToast }, ref) => {
    const [rawPrompt, setRawPrompt] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const [streamedText, setStreamedText] = useState("");
    const [statusIndex, setStatusIndex] = useState(0);
    const [progressChars, setProgressChars] = useState(0);
    const [totalChars, setTotalChars] = useState(0);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const formRef = useRef<HTMLDivElement>(null);
    const streamContentRef = useRef<HTMLDivElement>(null);
    const apiKey = usePromptStore((s) => s.apiKey);
    const addPrompt = usePromptStore((s) => s.addPrompt);

    useImperativeHandle(ref, () => ({
      reset: () => {
        setRawPrompt("");
        setError(null);
        textareaRef.current?.focus();
        formRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      },
    }));

    useEffect(() => {
      if (!isStreaming) {
        setStatusIndex(0);
        return;
      }
      const interval = setInterval(() => {
        setStatusIndex((prev) => (prev + 1) % STATUS_MESSAGES.length);
      }, 2000);
      return () => clearInterval(interval);
    }, [isStreaming]);

    useEffect(() => {
      if (streamContentRef.current) {
        streamContentRef.current.scrollTop = streamContentRef.current.scrollHeight;
      }
    }, [streamedText]);

    const audioRef = useRef<HTMLAudioElement>(null);

    const primeAudio = () => {
      const el = audioRef.current;
      if (!el) return;
      try {
        el.volume = 0;
        el.play().then(() => {
          el.pause();
          el.currentTime = 0;
          el.volume = 0.6;
        }).catch(() => {});
      } catch {}
    };

    const playVillagerSound = () => {
      const el = audioRef.current;
      if (!el) return;
      try {
        el.currentTime = 0;
        el.play().catch(() => {});
      } catch {}
    };

    const charCount = rawPrompt.length;
    const wordCount = rawPrompt.trim() ? rawPrompt.trim().split(/\s+/).length : 0;
    const estimatedTokens = Math.ceil(wordCount * 1.3);

    const handleSubmit = async (e: React.FormEvent) => {
      e.preventDefault();
      if (!rawPrompt.trim()) return;

      primeAudio();
      setLoading(true);
      setError(null);
      setIsStreaming(true);
      setStreamedText("");
      setProgressChars(0);
      setTotalChars(0);

      try {
        const generatedText = await generatePrompt(rawPrompt, apiKey);

        setTotalChars(generatedText.length);

        let currentIndex = 0;
        const typeInterval = setInterval(() => {
          const nextIndex = Math.min(
            currentIndex + TYPEWRITER_CHARS_PER_TICK,
            generatedText.length
          );
          setStreamedText(generatedText.slice(0, nextIndex));
          setProgressChars(nextIndex);
          currentIndex = nextIndex;

          if (currentIndex >= generatedText.length) {
            clearInterval(typeInterval);

            const record = {
              id: crypto.randomUUID(),
              original_prompt: rawPrompt,
              generated_prompt: generatedText,
              created_at: new Date().toISOString(),
            };
            addPrompt(record);
            setIsStreaming(false);
            setLoading(false);
            setStreamedText("");
            setProgressChars(0);
            setTotalChars(0);
            setRawPrompt("");
            playVillagerSound();
            showToast("Prompt forged successfully");
          }
        }, TYPEWRITER_INTERVAL_MS);
      } catch (err: any) {
        setIsStreaming(false);
        setLoading(false);
        setStreamedText("");
        setProgressChars(0);
        setTotalChars(0);
        const msg =
          err.response?.data?.error?.message ||
          err.message ||
          "Network error occurred.";
        setError(`Error: ${msg}`);
      }
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        handleSubmit(e as any);
      }
    };

    const typewriterPercent =
      totalChars > 0 ? Math.round((progressChars / totalChars) * 100) : 0;

    return (
      <div ref={formRef} className="border-t-2 border-cave-950 bg-cave-900/95">
        <div className="mx-auto max-w-3xl px-4 py-4 sm:px-6">
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-2 flex items-center gap-2 rounded-sm border-2 border-redstone-dark bg-redstone-dark/20 px-3 py-2 text-sm text-redstone-light"
              >
                <AlertTriangle className="h-4 w-4 shrink-0" aria-hidden="true" />
                {error}
              </motion.div>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {isStreaming && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-3 overflow-hidden"
              >
                <div className="rounded-sm border-2 border-diamond-dark bg-cave-950/80 p-3">
                  <div className="mb-1.5 flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-3.5 w-3.5 animate-spin text-diamond" />
                      <span className="font-pixel text-[0.55rem] tracking-wide text-diamond">
                        {STATUS_MESSAGES[statusIndex]}
                      </span>
                    </div>
                    {totalChars > 0 && (
                      <span className="text-[0.6rem] text-stone-300/70">
                        {typewriterPercent}%
                      </span>
                    )}
                  </div>
                  {streamedText && (
                    <div
                      ref={streamContentRef}
                      className="max-h-64 overflow-y-auto whitespace-pre-wrap break-words rounded-sm bg-cave-900 p-2.5 text-xs leading-relaxed text-iron/90"
                    >
                      {streamedText}
                      <span className="inline-block h-3.5 w-1.5 ml-0.5 bg-diamond animate-pulse" />
                    </div>
                  )}
                  {!streamedText && (
                    <div className="flex items-center gap-1.5 rounded-sm bg-cave-900 p-2.5 text-xs text-stone-300/60">
                      <span className="animate-pulse">Awaiting response...</span>
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <form
            onSubmit={handleSubmit}
            className="rounded-md border-2 border-cave-950 bg-cave-900/80 shadow-block-pressed"
          >
            <textarea
              ref={textareaRef}
              value={rawPrompt}
              onChange={(e) => setRawPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Help me negotiate a raise with my boss..."
              required
              rows={2}
              disabled={isStreaming}
              className="w-full resize-none bg-transparent px-4 pt-3 text-[0.95rem] text-iron placeholder:text-stone-300/50 focus:outline-none disabled:opacity-50"
            />

            <div className="flex items-center justify-between gap-3 px-4 pb-3 pt-1">
              <div className="flex flex-wrap gap-x-4 gap-y-0.5 text-xs text-stone-300/70">
                <span>
                  Chars: <strong className="text-iron/90">{charCount}</strong>
                </span>
                <span>
                  Words: <strong className="text-iron/90">{wordCount}</strong>
                </span>
                <span>
                  ~Tokens: <strong className="text-iron/90">{estimatedTokens}</strong>
                </span>
                <span className="hidden sm:inline">Ctrl + Enter to forge</span>
              </div>

              <motion.button
                type="submit"
                disabled={loading || !rawPrompt.trim()}
                whileTap={loading || !rawPrompt.trim() ? undefined : { scale: 0.92 }}
                aria-label="Turn into CRISPE prompt"
                className={[
                  "flex h-9 w-9 shrink-0 items-center justify-center rounded-full border-b-4 transition-[filter]",
                  loading || !rawPrompt.trim()
                    ? "cursor-not-allowed border-cave-950 bg-cave-800 text-stone-300/50"
                    : "cursor-pointer border-emerald-dark bg-gradient-to-b from-emerald-light to-emerald text-cave-950 hover:brightness-110",
                ].join(" ")}
              >
                {loading ? (
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                ) : (
                  <ArrowUp className="h-4 w-4" aria-hidden="true" />
                )}
              </motion.button>
            </div>
          </form>
          <audio ref={audioRef} src={VILLAGER_SOUND} preload="auto" />
        </div>
      </div>
    );
  }
);

PromptGeneratorForm.displayName = "PromptGeneratorForm";
