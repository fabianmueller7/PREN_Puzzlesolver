<!-- codetree-instructions -->
## CodeTree — Cached Codebase Access

This project has CodeTree MCP tools for faster, token-efficient codebase access.
**Use these instead of native Read/Grep/Glob whenever possible:**

| Instead of | Use | Why |
|-----------|-----|-----|
| `Read` | `codetree_read` | Cached content + symbol metadata |
| `Grep` | `codetree_search` | Instant symbol index search. Set `content_search=true` for line matches |
| `Glob` | `codetree_structure` | Indexed project tree with language stats |

**Other tools:**
- `codetree_find_refs` — Find all references to a symbol
- `codetree_summary` — Project overview (call at session start)
- `codetree_memory` — Save/recall insights across sessions

**Session workflow:**
1. Start: call `codetree_summary` and `codetree_memory` with `action: "get_context"`
2. During work: use `codetree_read`/`codetree_search` instead of native tools
3. Save discoveries: `codetree_memory` with `action: "save_insight"`
<!-- /codetree-instructions -->
