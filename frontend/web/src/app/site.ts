/**
 * Public-site facts in one place. The Contact page and footer render from here.
 */

export interface ContactChannel {
  label: string;
  /** Shown to the reader, e.g. an address or handle. */
  value: string;
  href: string;
}

export const SITE = {
  name: 'VisionGuard',
  repositoryUrl: 'https://github.com/RIT8603/VisionGuard',
  issuesUrl: 'https://github.com/RIT8603/VisionGuard/issues',
} as const;

/**
 * Official contact channels. Empty until the project owner publishes them;
 * the Contact page then points people to the issue tracker instead.
 *
 * To add one: { label: 'Email', value: 'team@example.org', href: 'mailto:team@example.org' }
 */
export const OFFICIAL_CONTACTS: readonly ContactChannel[] = [];
