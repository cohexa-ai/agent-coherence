# Security & supply chain

This document covers what end users need to know to install, configure, and run
`agent-coherence` safely: the outbound-traffic posture, which callers the local
coordinator can tell apart, the kill switches that disable telemetry-shaped
behavior, the local files the package writes, how to install with hash pinning,
and how to verify the cryptographic provenance of a published wheel.

## Outbound network destinations

The core `agent-coherence` package and its `[diagnose]` extra make **zero
outbound network requests** in v0 — no telemetry, no submission code, no phoning
home. Exactly two capabilities make deliberate, opt-in, **user-configured**
outbound connections, and nothing else does: cross-host coordinator mode and the
bring-your-own-substrate bindings (both below). Each connects only to an endpoint
you configure — never the internet at large, never an agent-coherence-controlled
host.

If you find outbound traffic from this package that is neither of those two
opt-in paths, please [open a security advisory](https://github.com/Cohexa-ai/agent-coherence/security/advisories)
— it would be a bug.

**Cross-host mode (default OFF).** Setting `CCS_REMOTE_COORDINATOR=1` and pointing
a `CoherentVolume` at a remote coordinator (`CCS_REMOTE_HOST` / `CCS_REMOTE_PORT` /
`CCS_REMOTE_SECRET_FILE`) makes the client open connections to that
**user-configured, private-range coordinator endpoint** — never the internet, and
no telemetry. This host-leaving traffic is opt-in, and the coordinator only binds
beyond loopback to an RFC-1918/4193 address (see the cross-host demo,
`examples/cross_host/`). With the flag unset the zero-outbound posture above is
unchanged.

**BYO substrate bindings (default OFF).** The optional `[coherent-row]` (Postgres)
and `[coherent-object]` (S3) extras connect to the substrate you declare in a
Coherence Manifest — your own database or object store, never the internet at
large and never an agent-coherence-controlled host. Every connection target is
egress-controlled at manifest-load time, before any driver is built: the SSRF deny
runs on the **resolved** address (cloud-metadata, link-local, and CGNAT ranges are
hard-denied; RFC-1918/4193 private ranges require `CCS_SUBSTRATE_ALLOW_PRIVATE=1`),
TLS is required unless you acknowledge a plaintext link with
`CCS_SUBSTRATE_INSECURE=1`, an inline secret in a DSN or endpoint URL is refused
(supply a credential *reference*, never a literal), and a target this loader
cannot classify — a Postgres `service=` entry whose host lives in an unreadable
`pg_service.conf` — is refused outright. Two residuals to know: libpq and botocore
re-resolve DNS at connect time, so a bare hostname carries a narrow connect-time
rebind window; and the `PGHOST` / `PGHOSTADDR` / `PGSERVICE` environment variables
are invisible to a DSN-text parser — pin `host`/`hostaddr` in the manifest if you
rely on egress control. The coordinator itself still receives metadata only (a
content hash, never your bytes); the bytes live in your substrate, which is the
whole point.

**Client TLS (verified https).** The client can speak `https://` to a
TLS-terminating front with **enforced certificate verification**. Set
`CCS_REMOTE_TLS=1` to select https; the client verifies the server certificate
and, if verification fails, **fails closed — the bearer token is never sent** over
an unverifiable connection. There is no way to turn certificate verification off:
an insecure-https mode simply does not exist in the client's configuration. To
trust a private certificate authority instead of the system trust store, point
`CCS_REMOTE_CA_FILE` at a CA-bundle file; it is loaded fail-closed — a symlinked
bundle is refused, and a group- or world-writable bundle is refused (a trust
anchor an attacker can rewrite is not a trust anchor). That bundle is re-read on
every request, so replacing it takes effect on the next one. Without it, the
client loads the system trust store once per process, on its first https
request. Until the process restarts, a certificate removed from that store stays
trusted, and if the store was missing at that moment, every https request fails
verification. The client also **refuses
to follow redirects**: it talks to the one coordinator endpoint you configured, so
any redirect response is rejected rather than followed with the bearer attached.
TLS changes nothing for the loopback path: a plain `http://` loopback endpoint
needs no certificate and no acknowledgement.

**No coordinator request goes through an HTTP proxy.** The client ignores
`http_proxy`, `https_proxy` and the system proxy settings for every coordinator
endpoint, loopback or remote. On a machine with a proxy configured, honouring it
would send loopback requests, bearer token included, to the proxy. A remote
endpoint is the one host you configured and secured the link to, and a proxy is a
hop that neither `CCS_REMOTE_INSECURE` nor https verification covers. A remote
coordinator must therefore be reachable directly, for example over a tunnel or a
VPN; one reachable only through a forward proxy fails with
`CoordinatorUnavailable`. A TLS-terminating front is unaffected, because you point
the client *at* it as the endpoint.

**CA-profile requirements for the terminating proxy (read this before you
provision a certificate).** Coordinator endpoints are almost always IP literals
(private-range addresses like `10.0.0.5`), and that shapes what certificate the
TLS-terminating proxy must present:

- **The certificate must carry an IP subject alternative name** matching the
  endpoint address — for example `subjectAltName = IP:10.0.0.5`. A DNS-only
  certificate has no name that matches an IP-literal endpoint, so verification
  **fails closed** against it. This is the single most common cause of a
  connection that refuses to establish.
- **On Python 3.13 and newer the certificate must be RFC 5280-strict**, or
  verification rejects it: it needs a subject key identifier and an authority key
  identifier, `basicConstraints` marked critical, and an extended key usage of
  `serverAuth`. Older certificates that older Python tolerated will be refused
  here.
- **Self-signed certificates are not a supported posture.** Use a private
  certificate authority and supply its bundle via `CCS_REMOTE_CA_FILE`. A private
  CA is what lets you mint an IP-SAN, RFC 5280-strict server certificate the
  client will actually verify.

**Plaintext-bearer guard (fail-closed, cross-host mode).** When the client is
*not* using verified https, the remote transport is plaintext HTTP — the
coordinator terminates no TLS itself, so encryption is the operator's out-of-band
responsibility (a WireGuard tunnel or a TLS-terminating proxy). To stop a bearer
from silently crossing an unencrypted routed link, the client **refuses to send it
to a non-loopback host** over plaintext (a typed `InsecureTransportRefused` is
raised). There are now two clean ways to satisfy the guard:

- **Verified https (the clean path).** A verified-https connection satisfies the
  guard automatically — the link is encrypted and the certificate is checked, so
  no acknowledgement is needed. `CCS_REMOTE_INSECURE=1` is **not** required for a
  properly TLS-fronted deployment.
- **`CCS_REMOTE_INSECURE=1` (the narrow out-of-band case).** This remains only for
  a *plaintext* link you have secured yourself out-of-band — a WireGuard tunnel or
  equivalent — where you knowingly accept the bearer riding plaintext HTTP inside
  that secured channel.

Loopback is unaffected. The plaintext ack *reduces* the silent-plaintext footgun;
it does **not** itself encrypt anything. Set the ack **narrowly** (per-invocation /
per-compose-service), never in a persistent global shell profile — a forgotten
global ack would blanket-acknowledge every future non-loopback host.

**Coordinator-side bind guard (fail-closed).** Symmetrically, a coordinator that
binds **beyond loopback** now refuses to serve unless the operator makes one of two
explicit assertions:

- `CCS_TLS_TERMINATED=1` — you assert a TLS-terminating front sits ahead of the
  coordinator.
- `CCS_SERVE_INSECURE=1` — you explicitly acknowledge an insecure (plaintext) link.

With neither set, a routed bind fails at startup rather than silently serving
bearer-authenticated requests in the clear. **These are operator assertions, not
enforcement:** the coordinator cannot verify that a proxy is actually present or
that the link is actually encrypted — it takes your word for it and records the
posture in its log. They acknowledge; they do not themselves encrypt or verify.
Loopback binds read neither variable and are unchanged. As with the client ack,
set these **narrowly** (per-invocation / per-compose-service), never as a
persistent global — a forgotten global assertion would blanket every future routed
bind. Production TLS/mTLS termination is a separate hardening step; the cross-host
mode as a whole remains experimental and default-off.

### MCP server network posture

The `stale-write-guard-fs` MCP server (installed via the `[mcp]` extra —
`pip install "agent-coherence[mcp]"`) speaks the Model Context Protocol over
**stdio only**. It opens no listening sockets and makes no outbound network
calls: a host such as Claude Desktop or Cursor spawns it as a subprocess and
talks to it over stdin/stdout, and the server coordinates writes through an
in-process `CoherentVolume` on the local filesystem. There is nothing to firewall
and no endpoint to configure on this path. See the MCP sections of the
[README](../README.md#mcp-server-stale-write-guard-fs) and the
[guide](guide.md#stale-write-guard-fs-mcp-server) for setup and the five `swg_*`
tools.

## Who the coordinator can tell apart

The local coordinator authenticates the **workspace**, not the individual caller.
Every request carries the one bearer secret in `.coherence/hook.secret` (mode
`0600`, readable only by your OS user), and every process in the workspace uses
the same secret. The session a request acts as is a `session_id` field in the
request body, which the coordinator checks for shape and nothing else.

What follows from that:

- Any process that can read the secret has full authority over the workspace.
  This is the intended trust model — every process running as your OS user is
  trusted alike, the same boundary as your shell history or SSH agent socket.
- A session whose client has claimed a [caller principal](guide.md#caller-principal)
  is refused, on the routes that release grants, commit, record who wrote, answer
  the effect fence or record workspace ownership, when a request names it without
  that principal or with another. That stops one writer's mistake — a copied
  request, a stale session id, a wrong id in a retry — from ending another
  writer's work or writing under its name. A session that never claimed one
  behaves as before: any holder of the secret can act as it.
- The principal separates writers that follow the protocol, and nothing more.
  Every principal is stored in `.coherence/state.db`, which any process running
  as your OS user can read. `CoherentVolume`, the MCP server and the substrate
  session present only the principal issued for their own session, so a request
  from them naming another session is refused. The Claude Code hook client finds
  its principal by the session id in each hook event, so on that surface a wrong
  session id can find a matching principal; there the principal exposes a client
  that never claimed or presents the wrong principal, but does not tell sessions
  apart.
- `last_writer_id`, and the sessions listed by `/status`, record which session a
  caller *said* it was — verified against its principal when it has one. Treat
  them as a record of cooperating writers, useful for debugging and display, not
  as proof of who wrote.
- The snapshot-session routes (`/session/read`, `/session/commit`,
  `/session/commit_all`, `/session/heartbeat`) also require the token that
  `/session/begin` returns, and attribute a commit to the session that token was
  issued for. That stops one snapshot session committing through another's
  token; it does not verify the `session_id` named at `/session/begin`.

What the coordinator does not publish: `/status` shows session names — which
embed the raw session id — and the policy's pattern lists (the tracked, user-added,
ignored and strict globs, which are the operator's directory layout) only in the
operator view (`?detail=full` plus the `Coherence-Local-Operator: true` header). A
`CoherentVolume` reads that view once, at attach, to check that the coordinator
enforces the globs it declared. Which paths each session lost to the
coordinator's grant sweep, and why (`sessions[].reclaimed`), is likewise in the
operator view only; the other views carry just the reclaim counts
(`sweep_reclaims_total`, `sweep_reclaims_by_trigger`). The default `minimal` view reports `agent_name`
as `null` and the pattern counts without the patterns, and the `metrics` view
carries no sessions at all. Hook
responses identify another session by its agent id, a one-way hash of the
session id. The `agent-coherence-status` command is an operator tool and asks
for the operator view by default, so its output does carry session names: run
it with `--detail minimal` before pasting the output into a bug report, and
point dashboards at `--detail metrics`. All of this is disclosure hygiene
rather than a boundary: under the model above, knowing a session id grants
nothing the secret does not already grant.

## Env-var kill switches

Set any of these to a truthy value (`1`, `true`, `yes`) to disable
telemetry-shaped output completely (no consent prompt, no calibration write,
no payload generation even in `--dry-run`):

- `DO_NOT_TRACK=1` (cross-tool consensus per consoledonottrack.com)
- `DISABLE_TELEMETRY=1`
- `CCS_DIAGNOSE_NO_TELEMETRY=1`

The CLI flags `--no-telemetry` and `--no-network` provide the same suppression
at the invocation level.

### Rendering defaults

Two environment variables override the report CTA defaults baked into the
HTML report. They are read at import time of `ccs.diagnose.render`:

- `CCS_DIAGNOSE_BOOK_A_CALL_URL` — replaces the default cal.com link. Must
  start with `http://` or `https://`; `javascript:`, `data:`, `vbscript:` are
  rejected by `RenderOptions.__post_init__`.
- `CCS_DIAGNOSE_CONTACT_EMAIL` — replaces the default reply-to address. Must
  match a plain `local@host` form; URL schemes embedded in the value are
  rejected.

Both env vars are validated by the same allowlist that gates caller-supplied
`RenderOptions(book_a_call_url=...)` / `contact_email=...` arguments — so an
attacker who can set the env cannot smuggle an XSS sink past the renderer.

## Local config and data files

`ccs-diagnose` writes to two well-known locations under XDG paths:

| File | Path | Mode | Created when |
|---|---|---|---|
| Consent state | `$XDG_CONFIG_HOME/ccs-diagnose/consent.json` | `0600` | First TTY run with consent prompt |
| Calibration corpus | `$XDG_DATA_HOME/ccs-diagnose/calibration.jsonl` | `0600` | First `--calibration-record` invocation |

Both directories are created with mode `0700`. Both fall back to `~/.config/...`
and `~/.local/share/...` when the XDG vars are unset. Reset the consent token any
time with `ccs-diagnose --reset-token`.

`CCSStore.record_to(path)` (the v0.8.2+ replay capture API) writes to a
caller-supplied directory. Files are mode `0o600`; the directory is created
with `mkdir(parents=True, exist_ok=True)` so the caller controls the path
and the surrounding-directory permissions.

| File | Path | Mode | Created when |
|---|---|---|---|
| Capture manifest | `<path>/manifest.json` | `0600` | `CCSStore.record_to.__enter__` (atomic write via tempfile + `os.replace`) |
| MESI state log | `<path>/state_log.jsonl` | `0600` | First emitted event when `state_log` stream is enabled (default) |
| Content audit log | `<path>/content_audit_log.jsonl` | `0600` | First emitted event when `content_audit_log` stream is enabled (default). Pass `streams={"state_log"}` to opt out — useful for PII-constrained partners. |

The capture path **refuses to start** when `<path>/manifest.json` already
exists (`SessionDirectoryNotEmptyError`) to prevent silent multi-instance
trace interleave. No content is read by `agent-coherence-replay` from any
location other than the explicit `session_dir` argument.

### Caller-principal files (Claude Code hook client)

The Claude Code hook client runs one process per hook event, so it keeps each
session's [caller principal](guide.md#caller-principal) on disk beside
`hook.secret`, keyed by the session's agent id — the one-way hash of the session
id that `/status` shows, never the raw session id. `CoherentVolume`, the MCP
server and the substrate session hold theirs in memory and write neither file.

| File | Path | Mode | Created when |
|---|---|---|---|
| Mint nonce | `<workspace>/.coherence/caller-principal-<agent-id>.nonce` | `0600` | The session's first hook event with no stored principal, unless `server.pid` names the plugin's Node backend — *before* the claim is sent, so an older Python coordinator that then answers `404` leaves the file behind too. Created exclusively (`O_CREAT` with `O_EXCL`) and never rewritten; two hook processes racing on a new session share the winner's nonce |
| Caller principal | `<workspace>/.coherence/caller-principal-<agent-id>.principal` | `0600` | When the claim binds. Written to a private temporary file (`O_CREAT` with `O_EXCL`, `0600`) and renamed over the stored file, and replaced the same way when a refused request is recovered by claiming again |

The nonce is what lets the session re-obtain its principal after a lost claim
answer or a reset `state.db`; the principal is what every hook of the session
presents. Treat both like `hook.secret`: any process running as your OS user can
read them, and one that holds them together with `hook.secret` can act as that
session on every route. The client never deletes them. Remove them only when no
hook of that session can still run: a session whose nonce file is gone claims
again under a new nonce, which the coordinator refuses
(`caller_principal_claimed`), and from then on the routes that require a
principal refuse that session.

### Durable version retention (opt-in)

`SqliteArtifactRegistry` can durably retain a bounded history of committed
artifact versions (enabled with `retain_versions=True` plus a `RetentionPolicy`;
**off by default**). This is the one place the coordinator's durable store holds
artifact **content bytes** rather than only `content_hash` — a deliberate,
bounded reversal of the prior hash-only posture, scoped to retained versions and
to in-process embedders (the Claude Code hook/HTTP coordinator topology is
unchanged and stays hash-only).

| File | Path | Mode | Holds |
|---|---|---|---|
| Version store | `<workspace>/.coherence/state.db` (`artifact_versions` table) | `0600` | Retained version bodies (str/bytes), bounded by the configured `max_versions` / `max_age_seconds` |

What this means for sensitive content:

- **Content on disk.** With retention on, version bodies are written to
  `state.db`. The file and its `-wal` / `-shm` sidecars are created `0600`
  *before* the first write — there is no umask window — and `.coherence` is
  `0700`. Migrating an older database re-applies `0600` and warns once. Bodies
  can contain whatever your artifacts contain (credentials, PII), so treat the
  file as sensitive: never commit it to git, and copy it only with mode
  preserved (`install -m 0600` / `cp -p`) — a copy is as sensitive as the
  content.
- **Capture at registration.** A version body is captured when an artifact is
  first registered, not only on write, so a merely-observed artifact's initial
  body can land on disk.
- **Deletion, not unreachability.** Collecting a version (policy GC) or an epoch
  reset (delete-and-recreate of `state.db`) *deletes* the rows, but SQLite may
  keep freed-page residue in the file and its WAL until a checkpoint / `VACUUM`.
  To purge rotated content fully, remove `state.db` together with its `-wal` and
  `-shm` sidecars. Disabling retention does **not** purge existing rows — they
  stay readable; purge is a re-open under a tighter policy or an epoch reset.
- **What a retained version is.** The store records the content the coordinator
  *committed* at that version — not necessarily the bytes a client later
  persisted elsewhere.

The read side (`CoordinatorService.read_at_version`, `agent-coherence-replay
resolve`) opens the store **read-only** and returns retained bytes only on
explicit request; the replay CLI is metadata-only by default and emits bodies
only via `--include-content` / `--output-file`, so terminals, CI logs, and shell
history don't capture content inadvertently. No HTTP route serves version
content.

### Workspace checkpoints ride this retention window (file members)

A workspace checkpoint's **file members** depend on exactly the retention
described above — a **declared, bounded window** (the `artifact_versions` store
under its configured `max_versions` / `max_age_seconds` bounds), never an
indefinite hold:

- **The window is a bound, not a pin.** Retention keeps a bounded history and
  offers no per-version hold a checkpoint could place, so the file version a
  checkpoint points at can still age out (or be collected by the policy) before
  you restore. That is why a file member is labeled `restorable-unpinned` —
  never `restorable`: the label means "history exists now and may expire."
- **Checkpoint pins on file members verify; they do not extend.** Pinning a
  checkpoint only checks that the captured file version is currently retained
  and still matches its captured fingerprint. Verification cannot lengthen the
  retention window or exempt the version from the configured bounds.
- **Expiry is reported, never papered over.** If the retained version is gone by
  restore time, the restore reports that member's target as lost rather than
  restoring different content. With retention off (the default) — or the version
  already unreachable when the checkpoint is pinned — the member is labeled
  `forward_only`: described in the checkpoint, but not restorable.

S3 members are different: their checkpoint pin is a legal hold placed in
**your** bucket on the captured version — the substrate's own retention, subject
to your bucket's configuration and costs, never the coordinator's.

**Where the CLI keeps checkpoint state.** `agent-coherence-workspace` owns its
own durable store, deliberately separate from the Claude Code hook coordinator's
`state.db`:

| File | Path | Mode | Holds |
|---|---|---|---|
| Workspace checkpoint store | `<workspace>/.coherence/workspace.db` | `0600` | Checkpoint manifests (member paths, restore pointers, fingerprints, tier/pin/outcome state) **and** retained file-member version bodies |

Treat it exactly like `state.db` above: it is created `0600` inside a `0700`
`.coherence/` directory carrying a `*` gitignore, its `-wal` / `-shm` sidecars
are as sensitive as the file itself, and it holds the **content** of every file
member the CLI has observed — the CLI runs retention on, because without
retained bytes a file restore could only ever report `target_lost`. Those bodies
can contain whatever your files contain (credentials, PII), and they persist for
as long as the store's retention window keeps them: the CLI sets no count/age
bound, so in practice they accumulate until you remove the database together with
its sidecars. That is a content-at-rest and disk-growth fact, **not** a stronger
restore guarantee — the tier stays `restorable-unpinned` regardless, because
retention offers no per-version hold to back a stronger claim. Never commit the
file; copy it only with mode preserved.

### Member-path containment for workspace checkpoints

A checkpoint member names a path that the CLI later **reads at capture and
writes at restore**, so the path is re-validated on every filesystem access —
not once at argument-parse time — because restore replays paths persisted by an
earlier invocation. A member is refused when it is not a plain file living
wholly inside the workspace root:

| Refused | Why |
|---|---|
| A path escaping the workspace root, or containing `..` | Capture and restore stay inside the root, always |
| Any symlink component, including the leaf | A symlink swapped in after validation would redirect the write; reads and writes use `O_NOFOLLOW` so a late swap is rejected atomically at open |
| A regular file with more than one link | A hard link means an outside co-owner: capturing it reads foreign bytes, and restoring it writes through the shared inode outside the root |
| A non-regular file (FIFO, socket, block/char device) | Opening one can block indefinitely; the open is non-blocking and the check is re-taken on the file descriptor |
| Anything under `.coherence/**` | The coordinator's own state is never a workspace member |

Refusals are typed and land differently by leg, deliberately: at **capture** a
refusal aborts with exit `2` and nothing persists; inside a **restore** leg the
same refusal is absorbed as that member's `target_lost` so the rest of the
restore still concludes and reports honestly, rather than leaving the checkpoint
stuck mid-restore. One residual is documented rather than claimed away: an
intermediate *directory* component swapped between validation and open is not
fully closed (closing it needs a directory-descriptor walk), which the
single-host, single-uid trust model accepts.

### S3 credential posture for workspace checkpoints

**Bindings carry credentials; the CLI never does.** S3 members of a workspace
checkpoint are captured and restored through a `CoherentObject` binding you
construct in Python — the binding resolves credentials the way the BYO-substrate
posture above requires (references, never literals). The
`agent-coherence-workspace` CLI holds no credential path at all: `restore` on a
checkpoint with pending S3 members refuses cleanly and points you at the Python
API, rather than growing a second credential surface; `status` and `list` still
render those members from the durable manifest.

**No credentials in manifests or the registry.** A checkpoint's per-member
record stores the member path, an opaque restore pointer (the S3 versionId, or
a file member's version number), a fixed-width content fingerprint, and
tier/pin/outcome state — nothing credential-shaped, and never your bytes for S3
members. Copying or inspecting the registry database cannot leak a substrate
credential, because none is ever written.

**Least-privilege IAM for workspace versioning.** The base binding's emitted
writer policy (`s3:GetObject` + `s3:PutObject` on the exact key/prefix, with
explicit delete denies) covers plain coherence use. Workspace checkpoint and
restore exercise a wider, still-narrow surface — these are exactly the
operations the binding issues, no others:

| Capability | S3 calls made | IAM actions needed |
|---|---|---|
| Capture + live comparand reads | `GetObject` (current version) | `s3:GetObject` |
| Version-pinned reads (restore source, pin verification) | `GetObject` with a `VersionId` | `s3:GetObjectVersion` |
| Restore writes | conditional `PutObject` (`If-Match`) | `s3:PutObject` |
| Checkpoint pins | `PutObjectLegalHold` / `GetObjectLegalHold` | `s3:PutObjectLegalHold`, `s3:GetObjectLegalHold` |
| Delete legs (restoring an ABSENT fact) | `DeleteObject` (unconditional-latest) | `s3:DeleteObject` |

Scope every action to the same exact key/prefix ARN as the base policy. Two
deliberate narrownesses: the binding never issues a version-targeted delete —
on a versioned bucket its delete mints a delete marker and history survives —
so `s3:DeleteObjectVersion` (true history destruction) is not needed and should
stay denied; and it never lists bucket versions, so no list permission is
needed. Grant `s3:DeleteObject` only to the principal that runs restores: it is
the one addition beyond the base policy's explicit delete deny, and only the
delete leg uses it.

**Bucket versioning is an expectation, not an assumption.** A `restorable` S3
member requires a versioned bucket. Against an unversioned bucket the capture
does not guess: the missing version pointer is a typed refusal at capture time,
and the member is recorded `forward_only` — described in the manifest, not
restorable, stated before you ever depend on it. Checkpoint pins additionally
require Object Lock to be enabled on the bucket (an at-creation setting);
without it the pin attempt durably downgrades the member to
`restorable-unpinned` with `pin_state="pin_unavailable"` — loudly, never as a
quiet claim.

## Hash-pinned install for security-sensitive users

For reproducible installs with full dependency-graph pinning:

    pip install --require-hashes -r requirements-diagnose.txt

The `requirements-diagnose.txt` file in the repo root is regenerated on each
release via `uv export --format requirements-txt --frozen --extra diagnose
--no-emit-project --no-dev`. It pins every transitive dependency by SHA-256
hash.

`uv.lock` in the repo is the developer lockfile. Downstream installers should
prefer `requirements-diagnose.txt` for reproducible installs.

## Verifying release attestations (PEP 740)

Each wheel published to PyPI ships with a Sigstore-backed PEP 740 attestation
tied to the GitHub Actions workflow that built it. The attesting repository is
**version-scoped** — the project moved from `hipvlady/agent-coherence` to
`Cohexa-ai/agent-coherence`, and each already-published wheel immutably attests
the repository it was built under:

- **Wheels v0.11.0 and earlier** attest repository `hipvlady/agent-coherence`
  (built before the org migration; same maintainer, and the old URL redirects to
  the new org). Verify these with `--repo hipvlady/agent-coherence`.
- **Wheels v0.12.0 and later** attest repository `Cohexa-ai/agent-coherence`
  (published from the Cohexa-ai Trusted Publisher). Verify these with
  `--repo Cohexa-ai/agent-coherence`.

To verify before installing, pass the repository that matches the version you
are installing:

    pip install pypi-attestations
    # v0.12.0 and later  →  --repo Cohexa-ai/agent-coherence
    # v0.11.0 and earlier →  --repo hipvlady/agent-coherence
    pypi-attestations verify --provenance \
        --repo <REPO-FOR-THIS-VERSION> \
        --workflow release.yml \
        agent_coherence-X.Y.Z-py3-none-any.whl

The PyPI page also displays the verified provenance in the release sidebar.
You can also inspect the raw signed attestation directly:

    curl -s \
      https://pypi.org/integrity/agent-coherence/X.Y.Z/agent_coherence-X.Y.Z-py3-none-any.whl/provenance \
      | python3 -m json.tool

The `publisher` block in each attestation bundle should report
`{kind: GitHub, repository: <REPO-FOR-THIS-VERSION>, workflow: release.yml, environment: pypi}`,
where `<REPO-FOR-THIS-VERSION>` is `hipvlady/agent-coherence` for wheels v0.11.0
and earlier and `Cohexa-ai/agent-coherence` for wheels v0.12.0 and later. A
publisher that reports neither expected value — or reports the wrong one for the
version being installed — is the signature of a Trusted Publisher
misconfiguration or a tampered release: do not install if the values diverge from
the version-scoped expectation above.

> **Note on `gh attestation verify`.** That command queries GitHub's SLSA
> build-provenance attestation store, which the current release workflow does
> not populate. It will return HTTP 404 against this package's wheels. The
> PEP 740 attestation lives on PyPI; use `pypi-attestations verify` or the
> raw `curl` inspection above. A future release-workflow enhancement could
> add an `actions/attest-build-provenance` step to also publish SLSA
> attestations to GitHub, at which point `gh attestation verify` would work.

## CycloneDX SBOM

Each GitHub Release attaches a CycloneDX SBOM (`sbom.cyclonedx.json`) listing
the full transitive dependency surface at build time. Diff across releases to
see dependency-graph changes. Generated in CI via `cyclonedx-py environment`.

## Supply-chain threat model

The deepeval incident (April 2026, GitHub deepeval#2497) is the canonical
recent example of a malicious release published under a trusted name. The
controls in place to mitigate similar attacks:

| Threat | Control |
|---|---|
| Stolen/leaked maintainer PyPI token | PyPI Trusted Publishers via OIDC — no static token exists to steal |
| Hijacked maintainer GitHub account | Required PR review on `.github/workflows/release.yml`; ruleset protecting `refs/tags/v*` (admin-only bypass) |
| Typosquat package install | Reserved typo variants (`agent-coherance`, `agentcoherence`, `agent_coherence`, `ccs-diagnose`, `ccsdiagnose`) under the same publisher |
| Dependency-confusion attack via `--extra-index-url` | Canonical install command documented (below); no private mirror references |
| Unverified release tampering | PEP 740 attestations + SBOM (see sections above) |
| Malicious local package shadows `langgraph` | `_detect_stack()` reads `importlib.metadata.version("langgraph")`; a shadowing package could spoof version. Low risk for the calibration-only v0 surface (no live submission), hardening note for v1 |
| Runtime exfiltration | No import-time side effects (audit-hook test); `--no-network` / kill switches; consent-gated calibration write to local file only |

## Canonical install command

    pip install --index-url https://pypi.org/simple/ "agent-coherence[diagnose]"

Avoid `--extra-index-url` to a private mirror — that's the dependency-confusion
attack vector. If you must use a private mirror, ensure `agent-coherence` is
served only from the official PyPI index.

## Reporting security issues

Open a private security advisory at
`https://github.com/Cohexa-ai/agent-coherence/security/advisories/new` rather
than a public issue. We aim to respond within 72 hours.
