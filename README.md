# academic-mcp

A local **MCP service that is the whole backend of an academic research workflow**:
literature search, citation chains, per-project research memory, paper download
and PDF→Markdown conversion. Any MCP client can drive it — the first one is the
DSH `academic` agent preset, which contains no Python at all.

```
client (any MCP host)  ──JSON-RPC /mcp──▶  academic-mcp :8790
                                             ├─ Scopus + OpenAlex    (search, citation chains)
                                             ├─ DeepSeek Search API  (arXiv title rescue)
                                             ├─ project memory JSON  (notes, findings, progress)
                                             ├─ arXiv / Elsevier / browser → PDF
                                             └─ MinerU cloud API     (PDF → Markdown)
```

Everything that needs Python lives here. A client only sends tool calls.

## Tools

| Tool | Purpose |
|------|---------|
| `search_papers` | Scopus + OpenAlex search, merged, deduplicated and hybrid-reranked. `doi` looks a single paper up directly. Records the session's DOIs so `validate_doi` can authorise a download. |
| `citation_chain` | Forward / backward citation expansion from seed DOIs (OpenAlex). Pass `session_id` so the results are authorised for download by that session. |
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
| `MINERU_ACCOUNT` / `MINERU_PASSWORD` | log in at mineru.net to renew `MINERU_TOKEN` automatically; the token itself expires (14 days, measured 2026-10-19) | — |
| `ELSEVIER_API_KEY` | Scopus search + ScienceDirect download | — |
| `OPENALEX_API_KEY` | OpenAlex search + citation chains | — |
| `DEEPSEEK_API_KEY` | DeepSeek Search API — the web engine behind the arXiv title rescue | — |
| `ARXIV_SEARCH_ENGINE` | `auto` / `deepseek` / `ddgs` / `none`: rescue engine **after** the arXiv API comes back empty | `auto` |
| `DOWNLOAD_PROXY` | publisher/API access, and the rescue route in `failover` mode; empty = direct only | — |
| `DOWNLOAD_PROXY_MODE` | `primary` / `fallback` / `failover` / `none` — governs the **browser** resolver as well as the HTTP client | `fallback` |
| `GFW_PROXY` | primary route for OpenAlex search / arXiv (failover hop applies here too) | — |
| `ACADEMIC_MCP_HOST` / `ACADEMIC_MCP_PORT` | listen address | `127.0.0.1` / `8790` |
| `ACADEMIC_DOC_CACHE` | local conversion cache | `~/.cache/academic-mcp/doc-read` |

Full list with defaults and tuning knobs: [`.env.example`](.env.example).
There are two proxy settings and one Elsevier credential: `DOWNLOAD_PROXY`
(publisher/API access, and the rescue route in `failover` mode) and
`GFW_PROXY` (arXiv/search engines/OpenAlex). `ELSEVIER_API_KEY` alone is
enough for metadata; an institutional insttoken is not supported. DeepSeek's
Search API is the exception to both: it is reached directly at
`api.deepseek.com`, the same host the model API uses, so it never goes through
`GFW_PROXY`.

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

### Token lifetime and automatic renewal

`MINERU_TOKEN` is not a permanent API key. It is the OpenXLab SSO token
(`uaa-token`) that mineru.net issues when you log in, and it is a JWT with a hard
**limited** lifetime — 14 days for the token issued on 2026-10-19 (the previous
one lasted 90 days, so the number belongs to MinerU, not to this code). When it
lapses, every conversion fails with

```
HTTP 401  {"msgCode": "A0202", "msg": "user authenticate failed"}
```

while paper downloads keep working — so the symptom is "only the conversion is
broken". That is how the outage that began on 2026-10-11 went unnoticed for eight
days.

The service now manages this itself. Give it the account you use at
<https://mineru.net/apiManage> (email, phone number or username) and its
password:

| Variable | Meaning | Default |
|---|---|---|
| `MINERU_ACCOUNT` | login account (alias: `MINERU_USERNAME`) | — |
| `MINERU_PASSWORD` | its password; leave empty to disable renewal | — |
| `MINERU_TOKEN_RENEW` | master switch | `true` |
| `MINERU_TOKEN_RENEW_DAYS` | renew while this many days remain | `7` |
| `MINERU_TOKEN_FILE` | where a renewed token is stored | `~/.local/state/academic-mcp/mineru_token.json` |
| `MINERU_LOGIN_TIMEOUT` | ceiling for one login, seconds | `180` |

With both credentials set, the token is renewed:

- **at startup**, when it is expired or within `MINERU_TOKEN_RENEW_DAYS` — in a
  background thread, because a login drives a browser and startup must not wait;
- **mid-flight on a 401** — the conversion renews once and retries, so the
  caller never sees the failure. Parallel conversions share a single login (a
  `stale`-token check under a lock), not one browser each.

Manual control, and how to inspect it:

```bash
python -m academic_mcp.mineru_auth status           # expiry, source, is renewal configured
python -m academic_mcp.mineru_auth probe            # is the stored token still accepted?
python -m academic_mcp.mineru_auth renew            # log in now
python -m academic_mcp.mineru_auth renew --visible  # if the captcha needs a human
```

The `health` tool gained a `mineru_token` block (`expires_at`, `days_left`,
`expired`, `needs_renewal`, `source`, `renew_enabled`), so the state is visible
without reading logs.

**How it logs in.** There is no usable HTTP login: the SSO
(`sso.openxlab.org.cn`) RSA-encrypts the password and demands an Aliyun captcha
before it accepts `POST /gw/uaa-be/api/v1/login/byAccount` — without the captcha
it answers `A0101 request parameter validation error` (measured). Renewal
therefore drives the real login page with Camoufox: `https://mineru.net/apiManage/token`
redirects straight to the SSO form, the account and password are filled in, 登录
is clicked, and the `确认您不是机器人` checkbox (`#aliyunCaptcha-checkbox-icon`) is
clicked whenever it appears. The token is then read from the `uaa-token` cookie
and verified against the API before it is stored (a batch id that cannot exist
answers "not found" when authenticated and 401 when not — no quota spent).

**Where the renewed token goes.** Not into `.env`: the systemd unit runs with
`ProtectSystem=strict` and only `~/.local/state/academic-mcp`, the data dir and
the cache are writable, so a service-side `.env` edit would fail with `EROFS`.
The token is written to `MINERU_TOKEN_FILE` (mode 0600, in a 0700 directory), and
**whichever of that file and `MINERU_TOKEN` expires later wins** — pasting a
fresh token into `.env` by hand still takes effect immediately, and a stale value
in either place can never shadow a newer one.

**When renewal fails**, it is logged and the conversion fails with the 401 as
before; it is never silently degraded to a different converter. Usual causes: a
wrong password, an account that requires an SMS/2FA step, or a captcha that
escalated to a slider. For the last two, run `renew --visible` (on the desktop
display, where you can drag the slider yourself) and the token is captured and
stored as usual.

## Resolver order

Cheap and reliable first, expensive and fragile last:

1. `arxiv-direct` — the DOI *is* an arXiv ID (`10.48550/...`)
2. `elsevier-api` — `10.1016/` + ScienceDirect API key
3. `direct-pdf` — open-access publishers whose PDF needs no session cookies.
   Only entries **verified** to return `application/pdf` belong here. SciPost is
   deliberately excluded: it sits behind a proof-of-work gate, and dodging that
   with a non-browser User-Agent would evade a control someone installed on
   purpose — it goes through the browser instead.
4. `camoufox-browser` — headful Firefox, CloudFlare bypass, paywalls. Walks the
   same routes as the HTTP client (`DOWNLOAD_PROXY_MODE`), one browser session
   per exit, because entitlements are per-IP: an exit that is a subscriber for
   one publisher is a stranger for the next, and the second route is what
   rescues the first.
5. `arxiv-title-search` — find the preprint by title (arXiv API first, then a
   web engine, with every candidate verified)

The last one is a fallback for *content*, not access: when every publisher path
is paywalled, an arXiv preprint of the same paper is still worth reading. It
runs last so the version-of-record wins when reachable. Candidates are verified
(title similarity + author) before use — fetching the wrong paper is worse than
failing. It asks arXiv's own API first and keeps web search as the rescue,
because a resolver whose job is to work when the publisher refused us cannot
depend on search engines that may be refusing us too.

The rescue engine is DeepSeek's **Search API**, not DDGS. Measured on
2026-10-04, every DDGS backend was refusing this host (Brave 429, Google 403,
DuckDuckGo an empty 202): 23 of 27 title searches failed while the preprints sat
on arXiv. DeepSeek answers `site:arxiv.org "<title>"` (then the leading six
words, for titles the phrase match cannot reach) from one endpoint in ~0.5 s,
directly — no proxy. It is billed per successful request (5 CNY / 1000, empty
results free) and capped at 5 QPS per account; a rescued paper costs at most two
requests. Only the result URLs are read: the arXiv ID they carry is a lead, and
`_verify` still checks every candidate against arxiv.org metadata before a byte
is downloaded. `ARXIV_SEARCH_ENGINE=ddgs` restores the old engine, `none` keeps
the API alone; with no `DEEPSEEK_API_KEY` the default (`auto`) falls back to
DDGS so an existing deployment keeps working.

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
│   ├── mineru_auth.py       # token expiry, storage and automatic renewal
│   ├── storage.py           # DOI→key, cache validity
│   ├── httpclient.py        # pooled client, retries, proxy policy
│   ├── websearch.py         # DeepSeek Search API (the arXiv rescue engine)
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
- Automatic MinerU renewal means `MINERU_PASSWORD` sits in clear text in `.env`.
  Keep that file mode 0600 and owned by the service user; the account is the one
  you use on mineru.net, so give it a unique password. A renewed token is stored
  separately in `MINERU_TOKEN_FILE` (also 0600) and is never logged — only a
  fingerprint (`first10…last4`) is ever printed.
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
