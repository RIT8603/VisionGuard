/**
 * OpenRouter adapter. The only file that knows about OpenRouter's API
 * (OpenAI-style chat completions over plain fetch; no SDK).
 *
 * Env: OPENROUTER_API_KEY and ASSISTANT_MODEL, both required. There is no
 * default model on purpose: price varies widely between models, so the
 * assistant stays unavailable until one is chosen explicitly.
 */

import { ProviderError, type AssistantProvider, type Env, type ReplyInput } from './provider.js';

const ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions';
/** Per attempt. A help widget should fail fast, not hang. */
const TIMEOUT_MS = 25_000;

interface CompletionResponse {
  choices?: Array<{
    finish_reason?: string | null;
    message?: { content?: unknown };
    error?: { message?: string };
  }>;
  error?: { code?: number | string; message?: string };
}

export function createOpenRouterProvider(env: Env): AssistantProvider | null {
  const apiKey = env.OPENROUTER_API_KEY?.trim();
  const model = env.ASSISTANT_MODEL?.trim();
  if (!apiKey || !model) return null;

  async function post(body: string): Promise<Response> {
    try {
      return await fetch(ENDPOINT, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
          // Attribution shown in the OpenRouter dashboard.
          'X-Title': 'VisionGuard help assistant',
        },
        body,
        signal: AbortSignal.timeout(TIMEOUT_MS),
      });
    } catch (error) {
      const reason = error instanceof Error && error.name === 'TimeoutError' ? 'timeout' : 'network';
      throw new ProviderError('upstream', `OpenRouter ${reason}`);
    }
  }

  return {
    name: 'openrouter',
    async reply({ system, messages, maxOutputTokens }: ReplyInput): Promise<string> {
      const body = JSON.stringify({
        model,
        max_tokens: maxOutputTokens,
        messages: [{ role: 'system', content: system }, ...messages],
      });

      // One retry, only for failures that are likely transient.
      let response = await post(body).catch(() => null);
      if (!response || response.status >= 500) response = await post(body);

      if (response.status === 429) throw new ProviderError('rate_limited', 'OpenRouter rate limit');
      // 403 = the input was flagged by the model's moderation.
      if (response.status === 403) throw new ProviderError('refused', 'OpenRouter moderation flagged the input');
      if (!response.ok) {
        // 401 bad key, 402 out of credits, 400 bad model id, 5xx provider down...
        throw new ProviderError('upstream', `OpenRouter HTTP ${response.status}`);
      }

      let data: CompletionResponse;
      try {
        data = (await response.json()) as CompletionResponse;
      } catch {
        throw new ProviderError('upstream', 'OpenRouter returned invalid JSON');
      }

      // Errors can also arrive inside a 200 response.
      if (data.error) throw new ProviderError('upstream', `OpenRouter error ${data.error.code ?? ''}`.trim());
      const choice = data.choices?.[0];
      if (choice?.error) throw new ProviderError('upstream', 'OpenRouter provider error');
      if (choice?.finish_reason === 'content_filter') {
        throw new ProviderError('refused', 'OpenRouter content filter');
      }

      const content = choice?.message?.content;
      const text = typeof content === 'string' ? content.trim() : '';
      if (!text) throw new ProviderError('upstream', `Empty reply (finish_reason ${choice?.finish_reason ?? 'none'})`);
      return choice?.finish_reason === 'length' ? `${text}…` : text;
    },
  };
}
