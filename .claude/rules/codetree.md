MANDATORY: Use CodeTree MCP tools instead of native Read/Grep/Glob, and prefer cheaper tools first.

Tool ladder (cheapest → most expensive — pick the cheapest that answers the question):

1. `codetree_outline` — file shape (exports, imports, size, hash). NO body.
2. `codetree_probe` — does file X mention Y? Line numbers only, NO snippets.
3. `codetree_search` — symbol search; small `limit`, use `cursor` for pagination.
4. `codetree_structure` — project tree; pass `format='text'` for compact output.
5. `codetree_read` — full content. ALWAYS pass `session_id` so the server can return
   `unchanged_since_last_read` when you've already read the same content this session.
   If you saved the previous response's `hash`, pass it as `expected_hash` to skip the body.
6. `codetree_find_refs` — cross-file references.
7. `codetree_edit` / `codetree_write` — edit/write without a prior Read.

Rules:

- DO NOT re-read a file with the same hash you already saw — refer to your prior response.
- Use `codetree_summary` at session start; it includes per-file `estimatedTokens` so you can
  rank what to read by **cost**.
- Use `codetree_memory` action `get_context` to recall prior sessions; `save_insight` for
  durable findings (architecture, bug causes, key decisions).

These tools serve cached data and save tokens. ALWAYS prefer them.
