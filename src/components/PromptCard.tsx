import { useState } from "react";
import { Copy, Check, FileText, FileDown } from "lucide-react";
import { PromptRecord } from "../types";
import { IconButton } from "./IconButton";
import { PixelPanel } from "./PixelPanel";

interface PromptCardProps {
  prompt: PromptRecord;
  showToast: (message: string) => void;
}

export function PromptCard({ prompt, showToast }: PromptCardProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(prompt.generated_prompt);
      setCopied(true);
      showToast("Prompt copied to clipboard");
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy text: ", err);
    }
  };

  const handleDownload = (format: "txt" | "md") => {
    const element = document.createElement("a");
    const file = new Blob([prompt.generated_prompt], {
      type: format === "md" ? "text/markdown" : "text/plain",
    });
    element.href = URL.createObjectURL(file);
    element.download = `crispe-prompt-${Date.now()}.${format}`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
    showToast(`Downloaded as .${format}`);
  };

  return (
    <PixelPanel
      tone="stone"
      hover
      className="p-5 sm:p-6 animate-blockPop"
    >
      <div className="mb-4 border-b-2 border-cave-950/60 pb-4">
        <span className="mb-1.5 block font-pixel text-[0.6rem] text-gold tracking-wide">
          YOUR RAW IDEA
        </span>
        <p className="m-0 text-[1.05rem] leading-snug text-stone-300">{prompt.original_prompt}</p>
      </div>

      <div className="relative rounded-sm border-2 border-cave-950 bg-cave-950/70 p-4">
          <div className="mb-2 flex items-start justify-between gap-3">
            <span className="font-pixel text-[0.6rem] tracking-wide text-diamond">CRISPE PROMPT</span>
            <div className="flex flex-wrap items-center gap-1.5">
            <IconButton label="Copy to clipboard" active={copied} onClick={handleCopy}>
              {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? "Copied" : "Copy"}
            </IconButton>
            <IconButton label="Download as .txt" onClick={() => handleDownload("txt")}>
              <FileText className="w-3.5 h-3.5" />
              .TXT
            </IconButton>
            <IconButton label="Download as .md" onClick={() => handleDownload("md")}>
              <FileDown className="w-3.5 h-3.5" />
              .MD
            </IconButton>
          </div>
        </div>
        <p className="m-0 whitespace-pre-wrap text-sm leading-relaxed text-iron/90">
          {prompt.generated_prompt}
        </p>
      </div>

      <div className="mt-3 text-right text-xs text-stone-300/70">
        {new Date(prompt.created_at).toLocaleString()}
      </div>
    </PixelPanel>
  );
}
