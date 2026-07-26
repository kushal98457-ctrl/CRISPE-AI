import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Pickaxe, Plus, Layers, Search, PanelLeftClose, PanelLeft, KeyRound, X } from "lucide-react";
import { PromptRecord } from "../types";
import { Button } from "./Button";
import { usePromptStore } from "../store/promptStore";

interface SidebarProps {
  history: PromptRecord[];
  onNewPrompt: () => void;
  searchQuery: string;
  onSearchChange: (value: string) => void;
  onJumpToPrompt: (id: string) => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

export function Sidebar({
  history,
  onNewPrompt,
  searchQuery,
  onSearchChange,
  onJumpToPrompt,
  mobileOpen,
  onCloseMobile,
}: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const setApiKey = usePromptStore((s) => s.setApiKey);

  const handleResetKey = () => {
    setApiKey("");
    onCloseMobile();
  };

  const handleNewPromptClick = () => {
    onNewPrompt();
    onCloseMobile();
  };

  const handleJumpClick = (id: string) => {
    onJumpToPrompt(id);
    onCloseMobile();
  };

  const sidebarContent = (
    <>
      <div className="flex items-center justify-between gap-2 border-b-2 border-cave-950 p-4">
        <div className="flex items-center gap-2 overflow-hidden">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-sm border-2 border-cave-950 bg-cave-800 text-diamond">
            <Pickaxe className="h-4 w-4" />
          </div>
          <span className="truncate font-pixel text-[0.6rem] text-iron">CRISPE FORGE</span>
        </div>
        <button
          onClick={() => setCollapsed(true)}
          aria-label="Collapse sidebar"
          className="hidden md:flex h-7 w-7 shrink-0 items-center justify-center rounded-sm text-stone-300 hover:text-diamond"
        >
          <PanelLeftClose className="h-4 w-4" />
        </button>
        <button
          onClick={onCloseMobile}
          aria-label="Close menu"
          className="flex md:hidden h-7 w-7 shrink-0 items-center justify-center rounded-sm text-stone-300 hover:text-diamond"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      <div className="flex flex-col gap-1 p-3">
        <Button variant="grass" size="sm" className="w-full justify-start" onClick={handleNewPromptClick}>
          <Plus className="h-3.5 w-3.5" />
          New prompt
        </Button>

        <button
          onClick={() => setSearchOpen((v) => !v)}
          className="mt-1 flex items-center gap-2.5 rounded-sm px-3 py-2 text-left text-sm text-stone-300 hover:bg-white/5 hover:text-iron"
        >
          <Search className="h-4 w-4 shrink-0" aria-hidden="true" />
          Search
        </button>
        <AnimatePresence>
          {searchOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden px-1"
            >
              <input
                autoFocus
                value={searchQuery}
                onChange={(e) => onSearchChange(e.target.value)}
                placeholder="Search history..."
                className="w-full rounded-sm border-2 border-cave-950 bg-cave-800 px-2.5 py-1.5 text-xs text-iron placeholder:text-stone-300/50 focus:outline-none focus:shadow-glow"
              />
            </motion.div>
          )}
        </AnimatePresence>

        <div className="mt-1 flex items-center gap-2.5 px-3 py-2 text-sm text-stone-300">
          <Layers className="h-4 w-4 shrink-0" aria-hidden="true" />
          History
          <span className="ml-auto rounded-sm bg-cave-800 px-1.5 py-0.5 text-[0.65rem] text-stone-300">
            {history.length}
          </span>
        </div>
      </div>

      <div className="border-t-2 border-cave-950/70 px-4 pt-3 text-[0.65rem] font-semibold uppercase tracking-wide text-stone-300/70">
        Recent
      </div>

      <div className="flex-1 overflow-y-auto px-2 py-2">
        {history.length === 0 ? (
          <p className="px-2 py-3 text-xs text-stone-300/60">Nothing forged yet.</p>
        ) : (
          <ul className="flex flex-col gap-0.5">
            {history.map((item) => (
              <li key={item.id}>
                <button
                  onClick={() => handleJumpClick(item.id)}
                  className="w-full truncate rounded-sm px-2.5 py-2 text-left text-sm text-stone-300 hover:bg-white/5 hover:text-iron"
                  title={item.original_prompt}
                >
                  {item.original_prompt || "Untitled prompt"}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="border-t-2 border-cave-950 p-3">
        <Button
          variant="diamond"
          size="sm"
          className="w-full justify-center"
          onClick={handleResetKey}
        >
          <KeyRound className="w-3.5 h-3.5" />
          Change API Key
        </Button>
      </div>
    </>
  );

  return (
    <>
      {/* Mobile overlay */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-40 bg-black/60 md:hidden"
            onClick={onCloseMobile}
          />
        )}
      </AnimatePresence>

      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ x: "-100%" }}
            animate={{ x: 0 }}
            exit={{ x: "-100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 250 }}
            className="fixed left-0 top-0 z-50 flex h-full w-72 flex-col border-r-2 border-cave-950 bg-cave-900 shadow-panel md:hidden"
          >
            {sidebarContent}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Desktop sidebar */}
      {collapsed ? (
        <div className="hidden md:flex h-full w-16 shrink-0 flex-col items-center gap-4 border-r-2 border-cave-950 bg-cave-900 py-4">
          <button
            onClick={() => setCollapsed(false)}
            aria-label="Expand sidebar"
            className="flex h-9 w-9 items-center justify-center rounded-sm border-2 border-cave-950 bg-cave-800 text-stone-300 hover:text-diamond"
          >
            <PanelLeft className="h-4 w-4" />
          </button>
          <div className="flex h-10 w-10 items-center justify-center rounded-sm border-2 border-cave-950 bg-cave-800 text-diamond">
            <Pickaxe className="h-5 w-5" />
          </div>
          <button
            onClick={onNewPrompt}
            aria-label="New prompt"
            className="flex h-10 w-10 items-center justify-center rounded-sm border-b-4 border-emerald-dark bg-gradient-to-b from-emerald-light to-emerald text-cave-950"
          >
            <Plus className="h-5 w-5" />
          </button>
        </div>
      ) : (
        <div className="hidden md:flex h-full w-64 shrink-0 flex-col border-r-2 border-cave-950 bg-cave-900">
          {sidebarContent}
        </div>
      )}
    </>
  );
}
