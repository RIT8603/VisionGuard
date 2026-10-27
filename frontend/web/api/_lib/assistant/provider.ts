/**
 * Provider-neutral contract for the help assistant.
 *
 * The handler only ever talks to an AssistantProvider. Adding a provider is
 * one adapter file plus one case in getProvider(); nothing else changes.
 */

export interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
}

export interface ReplyInput {
  system: string;
  /** Oldest first; always ends with a user turn. */
  messages: ChatTurn[];
  /** Upper bound on the reply length, in the provider's tokens. */
  maxOutputTokens: number;
}

export interface AssistantProvider {
  readonly name: string;
  reply(input: ReplyInput): Promise<string>;
}

/**
 * Why a provider call failed, in terms the handler can act on.
 * Adapters translate their SDK's errors into one of these.
 */
export type ProviderErrorKind = 'refused' | 'rate_limited' | 'upstream';

export class ProviderError extends Error {
  readonly kind: ProviderErrorKind;

  constructor(kind: ProviderErrorKind, message: string) {
    super(message);
    this.name = 'ProviderError';
    this.kind = kind;
  }
}

export type Env = Record<string, string | undefined>;

/**
 * The configured provider, or null when the assistant is not set up
 * (no ASSISTANT_PROVIDER, or the provider's credentials are missing).
 * Adapters are imported lazily so only the selected SDK is loaded.
 */
export async function getProvider(env: Env): Promise<AssistantProvider | null> {
  switch (env.ASSISTANT_PROVIDER?.trim().toLowerCase()) {
    case 'openrouter': {
      const { createOpenRouterProvider } = await import('./openrouter.js');
      return createOpenRouterProvider(env);
    }
    default:
      return null;
  }
}
