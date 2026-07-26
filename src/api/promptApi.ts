import axios from "axios";

const MASTER_PROMPT = `You are an expert prompt engineer. Transform the raw idea below into a CRISPE-framework prompt. Use ONLY these headings: # C - Context, # R - Role, # I - Instruction, # S - Specification, # P - Performance, # E - Example. Output the CRISPE prompt directly. No introductions, explanations, meta-commentary, or analysis. No markdown outside the headings.`;

export async function generatePrompt(
  promptText: string,
  apiKey: string
): Promise<string> {
  const response = await axios.post(
    `https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent?key=${apiKey}`,
    {
      system_instruction: {
        parts: [{ text: MASTER_PROMPT }],
      },
      contents: [
        {
          parts: [{ text: promptText }],
        },
      ],
      generationConfig: {
        temperature: 0.4,
      },
    }
  );

  const candidate = response.data?.candidates?.[0];
  if (!candidate) {
    throw new Error("No response from Gemini API");
  }

  return candidate.content?.parts?.[0]?.text || "";
}
