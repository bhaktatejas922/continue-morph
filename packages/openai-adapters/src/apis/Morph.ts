import {
    Completion,
    CompletionCreateParamsNonStreaming,
    CompletionCreateParamsStreaming,
} from "openai/resources/completions.mjs";
import {
    CreateEmbeddingResponse,
    EmbeddingCreateParams,
} from "openai/resources/embeddings.mjs";
import {
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionCreateParamsNonStreaming,
    ChatCompletionCreateParamsStreaming,
} from "openai/resources/index.mjs";
import { Model } from "openai/resources/models.mjs";
import { z } from "zod";
import { OpenAIConfigSchema } from "../types.js";
import { chatChunk, chatCompletion, customFetch } from "../util.js";
import {
    BaseLlmApi,
    CreateRerankResponse,
    FimCreateParamsStreaming,
    RerankCreateParams,
} from "./base.js";

// Morph only supports apply through a custom endpoint
export class MorphApi implements BaseLlmApi {
  private apiBase = "https://api.morphllm.com/v1/";

  constructor(private readonly config: z.infer<typeof OpenAIConfigSchema>) {
    this.apiBase = config.apiBase ?? this.apiBase;
    if (!this.apiBase.endsWith("/")) {
      this.apiBase += "/";
    }
    this.config = config;
  }

  async chatCompletionNonStream(
    body: ChatCompletionCreateParamsNonStreaming,
    signal: AbortSignal,
  ): Promise<ChatCompletion> {
    let content = "";

    // Convert the non-streaming params to streaming params
    const streamingBody: ChatCompletionCreateParamsStreaming = {
      ...body,
      stream: true,
    };

    for await (const chunk of this.chatCompletionStream(
      streamingBody,
      signal,
    )) {
      content += chunk.choices[0]?.delta?.content || "";
    }

    return chatCompletion({
      content,
      model: body.model,
    });
  }

  // We convert from what would be sent to OpenAI to Morph's format
  async *chatCompletionStream(
    body: ChatCompletionCreateParamsStreaming,
    signal: AbortSignal,
  ): AsyncGenerator<ChatCompletionChunk> {
    const fetch = customFetch(this.config.requestOptions);
    const headers = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${this.config.apiKey}`,
    };

    const prediction = body.prediction?.content ?? "";
    const originalCode =
      typeof prediction === "string"
        ? prediction
        : prediction.map((p) => p.text).join("");

    const userContent = body.messages.find((m) => m.role === "user")?.content;
    if (!userContent) {
      throw new Error("No edit snippet provided.");
    }

    const newCode =
      typeof userContent === "string"
        ? userContent
        : userContent
            .filter((p) => p.type === "text")
            .map((p) => p.text)
            .join("");

    // Morph expects a specific format with <code> and <update> tags
    // as mentioned in the docs
    const promptContent = `<code>${originalCode}</code>\n<update>${newCode}</update>`;

    const data = {
      model: body.model,
      messages: [
        {
          role: "user",
          content: promptContent,
        },
      ],
    };

    const url = this.apiBase + "chat/completions";
    const response = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(data),
      signal,
    });

    const result = (await response.json()) as any;
    const mergedCode = result.choices?.[0]?.message?.content || "";

    yield chatChunk({
      content: mergedCode,
      model: body.model,
    });
  }

  completionNonStream(
    body: CompletionCreateParamsNonStreaming,
    signal: AbortSignal,
  ): Promise<Completion> {
    throw new Error(
      "Morph provider does not support non-streaming completion.",
    );
  }

  completionStream(
    body: CompletionCreateParamsStreaming,
    signal: AbortSignal,
  ): AsyncGenerator<Completion> {
    throw new Error("Morph provider does not support streaming completion.");
  }

  fimStream(
    body: FimCreateParamsStreaming,
    signal: AbortSignal,
  ): AsyncGenerator<ChatCompletionChunk> {
    throw new Error(
      "Morph provider does not support streaming FIM completion.",
    );
  }

  embed(body: EmbeddingCreateParams): Promise<CreateEmbeddingResponse> {
    throw new Error("Morph provider does not support embeddings.");
  }

  rerank(body: RerankCreateParams): Promise<CreateRerankResponse> {
    throw new Error("Morph provider does not support reranking.");
  }

  list(): Promise<Model[]> {
    throw new Error("Morph provider does not support model listing.");
  }
}
