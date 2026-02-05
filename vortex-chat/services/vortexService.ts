import { AppMode, GroundingSupport, Message, Role, Source } from "../types";

type StreamChunk = {
  text: string;
  thought: string;
  sources: Source[];
  groundingSupports: GroundingSupport[];
  fileChanges?: { path: string; diff: string }[];
  done: boolean;
};

const extractDomain = (rawUrl: string): string => {
  try {
    return new URL(rawUrl).hostname.replace("www.", "");
  } catch {
    return "local";
  }
};

const toSources = (refs: unknown): Source[] => {
  if (!Array.isArray(refs)) return [];
  return refs
    .map((r, index) => {
      const url = typeof r === "string" ? r : JSON.stringify(r);
      const domain = extractDomain(url);
      return {
        url,
        domain,
        title: domain === "local" ? url : domain,
        index,
      } satisfies Source;
    })
    .filter(Boolean);
};

const extractFileChanges = (content: string): { path: string; diff: string }[] => {
  const changes: { path: string; diff: string }[] = [];
  const codeBlockRegex = /```file:([^\n]+)\n([\s\S]*?)```/g;
  let match: RegExpExecArray | null;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    changes.push({
      path: match[1].trim(),
      diff: match[2].trim(),
    });
  }
  return changes;
};

const buildMessages = (history: Message[], prompt: string, mode: AppMode) => {
  const messages: Array<{ role: "system" | "user" | "assistant"; content: string }> = [];

  if (mode === "agent") {
    messages.push({
      role: "system",
      content:
        "Eres Vortex, un asistente de ingeniería. " +
        "Cuando propongas cambios de código, usa bloques con este formato:\n\n" +
        "```file:ruta/al/archivo\n- línea eliminada\n+ línea añadida\n```\n\n" +
        "No inventes rutas. Sé preciso y minimalista.",
    });
  }

  for (const msg of history) {
    const content = (msg.content ?? "").trim();
    if (!content) continue;
    if (msg.role === Role.USER) messages.push({ role: "user", content });
    else messages.push({ role: "assistant", content });
  }

  messages.push({ role: "user", content: prompt });
  return messages;
};

const parseSseLines = (rawEvent: string): string[] => {
  // Supports multi-line SSE events; we only care about `data:`.
  return rawEvent
    .split(/\r?\n/)
    .map((l) => l.trimEnd())
    .filter((l) => l.startsWith("data:"))
    .map((l) => l.slice("data:".length).trim());
};

export class VortexService {
  private model: string = "auto";

  async *generateResponseStream(
    history: Message[],
    prompt: string,
    includeSources: boolean = false,
    useThinking: boolean = true,
    mode: AppMode = "ask"
  ): AsyncGenerator<StreamChunk> {
    const abortController = new AbortController();

    try {
      const payload = {
        model: this.model,
        stream: true,
        include_sources: includeSources,
        temperature: useThinking ? 0.7 : 0.2,
        messages: buildMessages(history, prompt, mode),
      };

      const resp = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(payload),
        signal: abortController.signal,
      });

      if (!resp.ok || !resp.body) {
        const text = await resp.text().catch(() => "");
        let detail = text;
        try {
          const parsed = JSON.parse(text);
          detail = parsed?.error?.message || parsed?.detail || text;
        } catch {
          // ignore
        }
        throw new Error(detail || `HTTP ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();

      let buffer = "";
      let fullText = "";
      let sources: Source[] = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        while (true) {
          const sepIndex = buffer.indexOf("\n\n");
          if (sepIndex === -1) break;

          const rawEvent = buffer.slice(0, sepIndex);
          buffer = buffer.slice(sepIndex + 2);

          for (const data of parseSseLines(rawEvent)) {
            if (!data) continue;
            if (data === "[DONE]") {
              return;
            }

            let parsed: any;
            try {
              parsed = JSON.parse(data);
            } catch {
              continue;
            }

            if (parsed?.sources) {
              sources = toSources(parsed.sources);
            }

            const delta = parsed?.choices?.[0]?.delta?.content;
            if (typeof delta === "string" && delta.length > 0) {
              fullText += delta;
            }

            if (typeof fullText === "string") {
              yield {
                text: fullText,
                thought: "",
                sources,
                groundingSupports: [],
                fileChanges: extractFileChanges(fullText),
                done: false,
              };
            }
          }
        }
      }

      yield {
        text: fullText,
        thought: "",
        sources,
        groundingSupports: [],
        fileChanges: extractFileChanges(fullText),
        done: true,
      };
    } finally {
      abortController.abort();
    }
  }
}

export const vortexService = new VortexService();

