import { Message, Role, Source, GroundingSupport } from "../types";

export class VortexService {
  private modelName: string = 'vortex-local';
  private apiUrl: string = 'http://localhost:8000/chat';

  constructor() {}

  private extractFileChanges(content: string): { path: string, diff: string }[] {
    const changes: { path: string, diff: string }[] = [];
    const codeBlockRegex = /```file:([^\n]+)\n([\s\S]*?)```/g;
    let match;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      changes.push({
        path: match[1].trim(),
        diff: match[2].trim()
      });
    }

    return changes;
  }

  async *generateResponseStream(history: Message[], prompt: string, useInternet: boolean = false, useThinking: boolean = true) {
    try {
      const response = await fetch(this.apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [...history, { role: Role.USER, content: prompt }],
          temperature: useThinking ? 0.7 : 0.4,
          max_tokens: 2048,
          use_internet: useInternet,
          use_thinking: useThinking
        })
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let rawBuffer = "";
      
      // Variables declared outside to be accessible after loop
      let displayText = "";
      let thoughtText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        rawBuffer += chunk;

        // Reset for recalculation based on full buffer
        displayText = rawBuffer;
        thoughtText = ""; // Reset thought to rebuild it from full buffer
        let parsedSources: Source[] = [];

        // 0. Extract Sources
        const sourcesRegex = /<vortex_sources>(.*?)<\/vortex_sources>/;
        const sourcesMatch = sourcesRegex.exec(rawBuffer);
        if (sourcesMatch) {
            try {
                parsedSources = JSON.parse(sourcesMatch[1]);
            } catch (e) {
                console.error("Failed to parse sources", e);
            }
            // Remove sources from display
            displayText = displayText.replace(sourcesRegex, "");
        }

        // 1. Extract complete blocks
        const thinkingRegex = /<thinking>([\s\S]*?)<\/thinking>/g;
        let match;
        while ((match = thinkingRegex.exec(displayText)) !== null) {
            thoughtText += match[1] + "\n";
        }
        
        // 2. Remove complete blocks from display
        displayText = displayText.replace(thinkingRegex, "");

        // 3. Handle active/open block (at the end)
        const openTagIndex = displayText.indexOf("<thinking>");
        if (openTagIndex !== -1) {
            // We have an open tag that hasn't closed yet
            const pendingThought = displayText.substring(openTagIndex + 10); // +10 for <thinking>
            thoughtText += pendingThought;
            displayText = displayText.substring(0, openTagIndex); // Remove everything starting from <thinking>
        }

        yield {
          text: displayText.trim(), // Trim strictly for display
          thought: thoughtText.trim(),
          sources: parsedSources,
          groundingSupports: [],
          fileChanges: this.extractFileChanges(displayText),
          done: false
        };
      }
      
      // Final yield
      // Re-parse sources one last time to be sure (though loop covers it)
      let finalSources: Source[] = [];
      const sourcesRegexFinal = /<vortex_sources>(.*?)<\/vortex_sources>/;
      const finalSourcesMatch = sourcesRegexFinal.exec(rawBuffer);
      if (finalSourcesMatch) { try { finalSources = JSON.parse(finalSourcesMatch[1]); } catch {} }

      yield {
          text: displayText.trim(),
          thought: thoughtText.trim(),
          sources: finalSources,
          groundingSupports: [],
          fileChanges: this.extractFileChanges(displayText),
          done: true
      };

    } catch (error) {
      console.error("Vortex Kernel Streaming Error:", error);
      throw error;
    }
  }
}

export const vortexService = new VortexService();
