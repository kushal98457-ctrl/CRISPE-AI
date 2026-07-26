import { create } from "zustand";
import { persist } from "zustand/middleware";
import { PromptRecord } from "../types";

interface PromptStoreState {
  apiKey: string;
  history: PromptRecord[];
  isLoading: boolean;
  error: string | null;
  setApiKey: (apiKey: string) => void;
  setHistory: (history: PromptRecord[]) => void;
  addPrompt: (prompt: PromptRecord) => void;
  setLoading: (isLoading: boolean) => void;
  setError: (error: string | null) => void;
}

export const usePromptStore = create<PromptStoreState>()(
  persist(
    (set) => ({
      apiKey: "",
      history: [],
      isLoading: false,
      error: null,
      setApiKey: (apiKey) => set({ apiKey }),
      setHistory: (history) => set({ history }),
      addPrompt: (prompt) =>
        set((state) => ({ history: [prompt, ...state.history] })),
      setLoading: (isLoading) => set({ isLoading }),
      setError: (error) => set({ error }),
    }),
    {
      name: "crispe-prompt-store",
      partialize: (state) => ({
        apiKey: state.apiKey,
        history: state.history,
      }),
    }
  )
);
