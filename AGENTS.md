# Repository privacy and security rules

## Never record private data

- Never write school email addresses, names, student IDs, phone numbers, mailbox contents, course content, assignment details, or personal schedules into documentation, examples, tests, commit messages, issue text, or agent/session notes.
- Use synthetic placeholders such as `student@example.edu`, `COURSE-1234`, and fictional messages instead.
- Do not paste, print, summarize, or store passwords, API keys, AWS credentials, OAuth tokens, refresh tokens, cookies, browser storage state, or authentication URLs containing codes.
- Treat NTULearn session files, Microsoft Graph token caches, SQLite workspace databases, and downloaded course/email files as private runtime data. Keep them local and ignored by Git.

## Agent and documentation behavior

- Before reading or editing runtime data, verify that it is necessary; prefer metadata, schemas, and synthetic fixtures.
- Redact sensitive values in terminal output and explanations. Do not include them in `README.md`, `AGENTS.md`, tests, examples, or generated summaries.
- Never commit `.env` files, session/storage JSON, SQLite databases, downloaded documents, mailbox exports, or logs containing personal data.
- If a secret or authenticated session is exposed, stop using it, remove local copies, and tell the user to revoke or sign out and re-authenticate. Do not attempt to conceal the exposure.

## Testing

- Tests must use fake credentials, fake cookies, synthetic course records, and temporary databases.
- Test output should contain only redacted status and counts, never message bodies, tokens, cookies, or personal identifiers.