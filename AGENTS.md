<!-- codetree-agents -->
## CodeTree — token-saving MCP tools (Cursor, Codex, Antigravity, …)

This repo registers the **CodeTree** MCP server. Use the cheapest tool that answers the question:

| Goal | Tool | Why it's cheap |
|------|------|----------------|
| What does file X export? | `codetree_outline` | Symbols + imports + size, **no body** |
| Does file X mention Y? | `codetree_probe` | Line numbers only, **no snippets** |
| Find a symbol | `codetree_search` (small `limit`, `cursor` for paging) | Index-only |
| Browse the tree | `codetree_structure` (`format='text'`) | Compact text, ~40% fewer tokens than JSON |
| Read content | `codetree_read` (`session_id` + `expected_hash`) | Skips body when unchanged since your last read |
| Find references | `codetree_find_refs` | Token postings + index |
| Project overview | `codetree_summary` | Includes per-file `estimatedTokens` |
| Memory across sessions | `codetree_memory` | `get_context`, `save_insight` |

Rules:

- **Do not re-read** a file when its `hash` matches your prior response.
- Pass `session_id` to `codetree_read` so the server can return `unchanged_since_last_read`.
- Prefer `codetree_outline` → `codetree_probe` → `codetree_search` → `codetree_read`.

Run `codetree init` from the repo root if MCP is not configured.
<!-- /codetree-agents -->
