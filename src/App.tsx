import { useRef, useState } from "react";
import { Menu } from "lucide-react";
import { PromptGeneratorForm, PromptComposerHandle } from "./features/promptGenerator/PromptGeneratorForm";
import { PromptHistory } from "./features/promptGenerator/PromptHistory";
import { usePromptStore } from "./store/promptStore";
import { ApiKeyGate } from "./components/ApiKeyGate";
import { Toast } from "./components/Toast";
import { Sidebar } from "./components/Sidebar";

const App = () => {
  const { apiKey, history, isLoading, setApiKey } = usePromptStore();
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const composerRef = useRef<PromptComposerHandle>(null);

  const showToast = (message: string) => {
    setToastMessage(message);
    setTimeout(() => {
      setToastMessage(null);
    }, 3000);
  };

  const handleNewPrompt = () => {
    setSearchQuery("");
    composerRef.current?.reset();
  };

  const handleJumpToPrompt = (id: string) => {
    document.getElementById(`prompt-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  if (!apiKey) {
    return (
      <>
        <ApiKeyGate onSetKey={setApiKey} />
        <Toast message={toastMessage} />
      </>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar
        history={history}
        onNewPrompt={handleNewPrompt}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        onJumpToPrompt={handleJumpToPrompt}
        mobileOpen={mobileMenuOpen}
        onCloseMobile={() => setMobileMenuOpen(false)}
      />

      <div className="flex flex-1 flex-col overflow-hidden transition-colors duration-300 mc-bg text-iron">
        <div className="flex flex-1 flex-col overflow-y-auto">
          <button
            onClick={() => setMobileMenuOpen(true)}
            aria-label="Open menu"
            className="fixed left-4 top-4 z-30 flex md:hidden h-10 w-10 items-center justify-center rounded-sm border-2 border-cave-950 bg-cave-800 text-stone-300 shadow-block-sm hover:text-diamond"
          >
            <Menu className="h-5 w-5" />
          </button>
          <PromptHistory
            history={history}
            isLoading={isLoading}
            showToast={showToast}
            searchQuery={searchQuery}
            className="pt-16 md:pt-0"
          />
        </div>

        <PromptGeneratorForm
          ref={composerRef}
          showToast={showToast}
        />
      </div>

      <Toast message={toastMessage} />
    </div>
  );
};

export default App;
