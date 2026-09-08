# academic-mcp

A local **MCP service that is the whole backend of an academic research workflow**:
literature search, citation chains, per-project research memory, paper download
and PDF→Markdown conversion. Any MCP client can drive it — the first one is the
DSH `academic` agent preset, which contains no Python at all.

```
client (any MCP host)  ──JSON-RPC /mcp──▶  academic-mcp :8790
                                             ├─ Scopus + OpenAlex    (search, citation chains)
                                             ├─ project memory JSON  (notes, findings, progress)
                                             ├─ arXiv / Elsevier / browser → PDF
                                             └─ MinerU cloud API     (PDF → Markdown)
```

Everything that needs Python lives here. A client only sends tool calls.

## Tools

| Tool | Purpose |
|------|---------|
| `search_papers` | Scopus + OpenAlex search, merged, deduplicated and hybrid-reranked. `doi` looks a single paper up directly. Records the session's DOIs so `validate_doi` can authorise a download. |
| `citation_chain` | Forward / backward citation expansion from seed DOIs (OpenAlex). |
| `memory` | Per-project research memory: `working-memory`, `session-key`, `update`, `list`, `get`, `goal`, `note`, `finding`, `unresolved`, `progress`, `resume`, `delete-paper`, `dump`. |
| `telemetry` | Read a session's tool-call log. |
| `fetch_paper_text` | DOI → full text (Markdown), cached under `<data_dir>/texts/`. |
| `fetch_paper_pdf` | DOI → PDF path only (no conversion). |
| `convert_document` | Any document (local path **or URL**) → Markdown: PDF, Word, PPT, Excel, images, HTML. Returns path + outline + page→line map. |
| `validate_doi` | May this DOI be downloaded in this session? (no network) |
| `contract` | Machine-readable tool contract: every tool with its required/optional arguments. Clients validate against it instead of assuming names. |
| `health` | Config + resolver stats (secrets masked). Call this first when something fails. |

`convert_document` is the one MinerU implementation for both the paper pipeline
and general document reading.

## Configuration

Settings come from the process environment, then `<MCP_HOME>/.env` (this
directory), then `~/.shellrc`. **Copy `.env.example` to `.env` and fill it in.**
Nothing else is read — in particular, no client's directory is consulted.

The essentials:

| Variable | Meaning | Default |
|---|---|---|
| `ACADEMIC_DATA_DIR` | library root: project memory JSON, `pdfs/`, `texts/`, `search_cache/`, `telemetry/` | `<MCP_HOME>/data` |
| `MINERU_TOKEN` | MinerU cloud PDF→Markdown (required for conversion) | — |
| `ELSEVIER_API_KEY` | Scopus search + ScienceDirect download | — |
| `OPENALEX_API_KEY` | OpenAlex search + citation chains | — |
| `DOWNLOAD_PROXY` | publisher downloads **and** Scopus/OpenAlex search; empty = direct | — |
| `GFW_PROXY` | arXiv / search-engine fallback only | — |
| `ACADEMIC_MCP_HOST` / `ACADEMIC_MCP_PORT` | listen address | `127.0.0.1` / `8790` |
| `ACADEMIC_DOC_CACHE` | local conversion cache | `~/.cache/academic-mcp/doc-read` |

Full list with defaults and tuning knobs: [`.env.example`](.env.example).
`ELSEVIER_INSTTOKEN` is an optional subscription-institution token — leave it
empty unless your library requires it.

Restart the service after editing (configuration is read once at startup):

```bash
systemctl --user restart academic-mcp      # or however you run it
```

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .                 # or: pip install academic-mcp
cp .env.example .env             # then edit
academic-mcp                     # or: python -m academic_mcp
```

The browser fallback needs a one-time download and a display (Xvfb is fine):

```bash
camoufox fetch
```

`academic-mcp.service` and `academic-mcp-xvfb.service` in this directory are
**examples** — they assume Linux with `systemd --user`. Adjust the paths before
using them.

## PDF → Markdown: MinerU only

There is deliberately **no fallback converter**. The previous backend tried
local MinerU (GPU) and then PyMuPDF4LLM; both produced silently degraded text
(lost formulas, broken tables, scrambled reading order) that a reading agent
then summarised as authoritative. Failing loudly is better than fabricating
structure.

## Resolver order

Cheap and reliable first, expensive and fragile last:

1. `arxiv-direct` — the DOI *is* an arXiv ID (`10.48550/...`)
2. `elsevier-api` — `10.1016/` + ScienceDirect API key
3. `direct-pdf` — open-access publishers whose PDF needs no session cookies.
   Only entries **verified** to return `application/pdf` belong here. SciPost is
   deliberately excluded: it sits behind a proof-of-work gate, and dodging that
   with a non-browser User-Agent would evade a control someone installed on
   purpose — it goes through the browser instead.
4. `camoufox-browser` — headful Firefox, CloudFlare bypass, paywalls
5. `arxiv-title-search` — find the preprint by title (DDGS + verification)

The last one is a fallback for *content*, not access: when every publisher path
is paywalled, an arXiv preprint of the same paper is still worth reading. It
runs last so the version-of-record wins when reachable. Candidates are verified
(title similarity + author) before use — fetching the wrong paper is worse than
failing.

## Display for the browser

Headful Firefox is required for CloudFlare Turnstile (headless is detected and
challenged harder). The resolver picks, in order: an existing Xvfb on `:99`–`:96`,
the ambient `DISPLAY`, a self-started Xvfb, otherwise headless.

## Layout

```
academic-mcp/
├── academic-mcp.py          # script entrypoint (kept for compatibility)
├── academic_mcp/
│   ├── __main__.py          # python -m academic_mcp
│   ├── config.py            # env + .env settings
│   ├── server.py            # MCP tools (incl. contract / health)
│   ├── pipeline.py          # cache → resolve → convert
│   ├── mineru.py            # MinerU cloud API (the only converter)
│   ├── storage.py           # DOI→key, cache validity
│   ├── httpclient.py        # pooled client, retries, proxy policy
│   ├── validate.py          # session-scoped DOI authorization
│   ├── markdown.py          # outline + page→line map
│   ├── resolvers/           # arxiv, elsevier, publisher, camoufox
│   └── agent/               # search, snowball, memory + MCP wrappers
├── scripts/                 # manual publisher / network diagnostics
└── tests/                   # offline smoke tests
```

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q          # offline: no network, no browser, no MinerU quota
```

The tests pin the parts clients depend on as a contract: the tool surface, the
argument names, the `cwd → session_id` rule and the memory round-trip.

## Security notes

- The service binds to `127.0.0.1` by default and keeps its keys in `.env`
  (never returned over the wire; `health` masks them).
- `validate_doi` only authorises a DOI that appeared in this session's search
  results, so a stray DOI cannot trigger a download.
- The browser fallback drives a real Firefox for CloudFlare challenges and
  paywalled publishers. Use it in line with those sites' terms.

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

Copyright (C) 2026 cxxiao.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. It is distributed in the hope that it will be useful, but **without
any warranty**; without even the implied warranty of merchantability or
fitness for a particular purpose.
