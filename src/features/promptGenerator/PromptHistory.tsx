import { motion } from "framer-motion";
import { SearchX } from "lucide-react";
import { PromptRecord } from "../../types";
import { PromptCard } from "../../components/PromptCard";
import { SkeletonCard } from "../../components/SkeletonCard";
import { EmptyState } from "../../components/EmptyState";

interface PromptHistoryProps {
  history: PromptRecord[];
  isLoading?: boolean;
  showToast: (message: string) => void;
  searchQuery?: string;
  className?: string;
}

export const PromptHistory: React.FC<PromptHistoryProps> = ({
  history,
  isLoading,
  showToast,
  searchQuery = "",
  className = "",
}) => {
  if (isLoading) {
    return (
      <div className="flex flex-col gap-6 px-4 py-6 sm:px-6">
        <SkeletonCard />
        <SkeletonCard />
      </div>
    );
  }

  if (!history || history.length === 0) {
    return <EmptyState />;
  }

  const query = searchQuery.trim().toLowerCase();
  const filtered = query
    ? history.filter(
        (item) =>
          item.original_prompt.toLowerCase().includes(query) ||
          item.generated_prompt.toLowerCase().includes(query)
      )
    : history;

  if (filtered.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex flex-col items-center gap-3 px-6 py-16 text-center text-stone-300"
      >
        <SearchX className="h-8 w-8 opacity-60" aria-hidden="true" />
        <p className="m-0 text-sm">No prompts match "{searchQuery}".</p>
      </motion.div>
    );
  }

  return (
    <div className={"flex flex-col gap-6 px-4 py-6 sm:px-6 " + className}>
      {filtered.map((item) => (
        <div key={item.id} id={`prompt-${item.id}`}>
          <PromptCard prompt={item} showToast={showToast} />
        </div>
      ))}
    </div>
  );
};
