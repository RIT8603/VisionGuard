import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import {
  ASSISTANT_MAX_CHARS,
  AssistantError,
  askAssistant,
  getAssistantAvailability,
  type AssistantTurn,
} from '../../api/assistant';

const SUGGESTIONS = [
  'What does VisionGuard do?',
  'How do I add an RTSP camera?',
  'What is the difference between an event and an alert?',
  'Why is detection slow?',
];

type Availability = 'unknown' | 'checking' | 'available' | 'unavailable';

/**
 * Product-help chat: a small corner button that opens a side panel (a bottom
 * sheet on phones). Static product knowledge only - it never sees live data.
 * Availability is checked on first open, not on page load.
 */
export function AssistantWidget() {
  const [open, setOpen] = useState(false);
  const [availability, setAvailability] = useState<Availability>('unknown');
  const [messages, setMessages] = useState<AssistantTurn[]>([]);
  const [draft, setDraft] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggleRef = useRef<HTMLButtonElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const requestRef = useRef<AbortController | null>(null);

  // Check availability once, the first time the panel opens. Keyed on `open`
  // only: closing mid-check aborts it and the next open checks again.
  useEffect(() => {
    if (!open || availability !== 'unknown') return;
    const controller = new AbortController();
    setAvailability('checking');
    getAssistantAvailability(controller.signal)
      .then((ok) => setAvailability(ok ? 'available' : 'unavailable'))
      .catch(() => {
        if (!controller.signal.aborted) setAvailability('unavailable');
      });
    return () => {
      controller.abort();
      setAvailability((current) => (current === 'checking' ? 'unknown' : current));
    };
  }, [open]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open, availability]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [messages, pending, error]);

  // Cancel an in-flight request when the portal unmounts.
  useEffect(() => () => requestRef.current?.abort(), []);

  const close = () => {
    setOpen(false);
    toggleRef.current?.focus();
  };

  const send = async (history: AssistantTurn[]) => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setPending(true);
    setError(null);
    try {
      const reply = await askAssistant(history, controller.signal);
      setMessages([...history, { role: 'assistant', content: reply }]);
    } catch (err) {
      if (controller.signal.aborted) return;
      if (err instanceof AssistantError && err.kind === 'unavailable') {
        setAvailability('unavailable');
      }
      setError(err instanceof Error ? err.message : 'The assistant could not answer right now.');
    } finally {
      if (requestRef.current === controller) {
        requestRef.current = null;
        setPending(false);
      }
    }
  };

  const ask = (text: string) => {
    const content = text.trim();
    if (!content || pending || availability !== 'available') return;
    const history: AssistantTurn[] = [...messages, { role: 'user', content }];
    setMessages(history);
    setDraft('');
    void send(history);
  };

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    ask(draft);
  };

  const onInputKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      ask(draft);
    }
  };

  const lastIsUser = messages.length > 0 && messages[messages.length - 1].role === 'user';
  const canType = availability === 'available' && !pending;
  const remaining = ASSISTANT_MAX_CHARS - draft.length;

  return (
    <>
      <button
        ref={toggleRef}
        type="button"
        className="assistant-toggle"
        aria-expanded={open}
        aria-controls="assistant-panel"
        onClick={() => (open ? close() : setOpen(true))}
      >
        <span aria-hidden="true">?</span> Help
      </button>

      {open && (
        <section
          id="assistant-panel"
          className="assistant-panel panel"
          role="dialog"
          aria-labelledby="assistant-title"
          onKeyDown={(event) => {
            if (event.key === 'Escape') close();
          }}
        >
          <div className="panel-head">
            <h2 id="assistant-title">Help assistant</h2>
            <button type="button" className="assistant-close" onClick={close} aria-label="Close help assistant">
              ×
            </button>
          </div>

          <div className="assistant-log" ref={logRef} aria-live="polite">
            {availability === 'checking' && <p className="assistant-note">Connecting…</p>}

            {availability === 'unavailable' && (
              <div className="assistant-state">
                <p className="assistant-state-title">Assistant unavailable</p>
                <p className="muted">
                  The help assistant is not set up on this deployment. The repository README covers setup and
                  troubleshooting.
                </p>
              </div>
            )}

            {availability === 'available' && messages.length === 0 && (
              <div className="assistant-state">
                <p className="muted">
                  Ask about VisionGuard's modules, camera setup or troubleshooting. Product help only: it cannot see
                  your cameras, events or alerts.
                </p>
                <div className="assistant-suggestions">
                  {SUGGESTIONS.map((question) => (
                    <button key={question} type="button" className="assistant-suggestion" onClick={() => ask(question)}>
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((message, index) => (
              <div key={index} className={`assistant-msg assistant-msg-${message.role}`}>
                <span className="assistant-msg-who">{message.role === 'user' ? 'You' : 'Assistant'}</span>
                <p>{message.content}</p>
              </div>
            ))}

            {pending && (
              <p className="assistant-note" role="status">
                Thinking<span className="assistant-dots" aria-hidden="true" />
              </p>
            )}

            {error && (
              <div className="assistant-error" role="alert">
                <p>{error}</p>
                {lastIsUser && availability === 'available' && (
                  <button type="button" className="assistant-retry" onClick={() => void send(messages)}>
                    Retry
                  </button>
                )}
              </div>
            )}
          </div>

          <form className="assistant-form" onSubmit={onSubmit}>
            <label htmlFor="assistant-input" className="visually-hidden">
              Your question
            </label>
            <textarea
              id="assistant-input"
              ref={inputRef}
              rows={2}
              value={draft}
              maxLength={ASSISTANT_MAX_CHARS}
              placeholder={availability === 'available' ? 'Ask a question…' : 'Assistant unavailable'}
              disabled={!canType}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={onInputKey}
            />
            <div className="assistant-form-row">
              <span className="assistant-count mono">{remaining < 200 ? `${remaining} left` : ''}</span>
              <button type="submit" className="summary-btn" disabled={!canType || !draft.trim()}>
                Send
              </button>
            </div>
          </form>
        </section>
      )}
    </>
  );
}
