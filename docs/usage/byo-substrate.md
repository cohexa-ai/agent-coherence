# BYO substrate: a hands-on walkthrough

This is the task-oriented companion to the reference in
[the guide](../guide.md#byo-substrate-bindings-coherentrow-coherentobject). It
walks from the zero-setup demo to a real Postgres row and a real S3 object, and
explains — in operational terms — what happens when two agents race.

**What a BYO-substrate binding is.** Coherence over shared state that lives in a
store *you already run* — a Postgres row, an S3 object — instead of a store this
library ships. The coordinator holds only coherence metadata (a monotonic
version, per-agent MESI state, a fixed-width `content_hash`, and an opaque
substrate token); it **never holds your bytes**. The bytes stay in your
substrate. The coherence layer drops *under* it.

**What it buys you over the substrate's own conditional write.** A native
`UPDATE … WHERE version = ?` or an S3 `If-Match` already rejects a single lost
update — but only *at the moment you write*. The binding adds the cross-agent
layer on top: a peer's commit marks your cached read **stale**, so your next
read or write is denied *before* you act on state that already moved — the thing
a bare conditional write can never surface — and every substrate speaks the
**same** typed conflict and the same `reacquire()` recovery.

---

## 1. Sixty seconds, offline, no credentials

Both demos run against a local coordinator with an in-memory substrate stand-in,
so the cross-agent value is visible with zero setup:

```bash
python -m examples.coherent_row.main                # the two-agent race, guarded
python -m examples.coherent_object.main --baseline  # the SAME race, unguarded — the silent clobber
```

Run the `--baseline` form first: it shows the stale write landing silently (the
lost update). The guarded form shows the second agent denied with a typed
`StaleView` before it can clobber, then recovering. That contrast is the whole
product.

---

## 2. A real Postgres row

Install the driver (an optional extra):

```bash
pip install "agent-coherence[coherent-row]"     # psycopg v3
```

### 2a. Provision the table once (least privilege)

The binding never runs DDL for you — it *emits* it so you (or your DBA) apply it
deliberately. `provisioning_sql()` returns a version-guard trigger, the
least-privilege role, and a one-time backfill:

```python
from ccs.adapters.coherent_row import provisioning_sql

print(provisioning_sql("workspaces").as_script())
```

What it emits, and why:

- An **owner-managed** `BEFORE INSERT/UPDATE` trigger that sets the new version
  from the *stored* prior row (`COALESCE(OLD.version, 0) + 1`) — so a client that
  supplies its own `NEW.version` cannot forge one, and a pre-existing row sitting
  at `NULL`/`0` still advances to a usable value.
- A **dedicated, non-owner, login-limited** role granted only `SELECT, UPDATE` on
  the one table — no `ALTER`, no `TRIGGER`, no `DELETE`, no re-grant — so the
  coherence role cannot disable the guard it runs under.
- A one-time **backfill** (`UPDATE … SET version = 1 WHERE version IS NULL OR
  version <= 0`) so rows that predate the version column can be onboarded — the
  binding refuses a version `<= 0` as a comparand, so without this a legacy row
  would be unreadable and therefore un-writable through the binding.

Apply the script as the table owner. The agents then connect as the limited role.

### 2b. Use it

```python
from ccs.adapters.coherent_row import CoherentRow

row = CoherentRow(dsn="postgresql://coherence_writer@db.internal/app", table="workspaces")

data, token = row.read("ws-42")                 # (bytes, token) from ONE read
revised = transform(data)
row.commit("ws-42", expected_token=token, new_bytes=revised)
# If a peer committed since your read, this raises StaleView — reacquire and retry:
#   data, token = row.reacquire("ws-42"); revised = transform(data); row.commit(...)
```

A self-owned connection (constructed from a `dsn`) carries a `connect_timeout`
and a per-statement `statement_timeout` by default, so a hung database surfaces a
reconcilable error instead of blocking forever; tune them via the constructor.
An injected `connection=` is treated as caller-owned — its pool and its timeouts
are yours to manage.

---

## 3. A real S3 object

```bash
pip install "agent-coherence[coherent-object]"  # boto3
```

The surface is identical; the token is the object **ETag**, captured from the
`put_object` response (never computed client-side):

```python
from ccs.adapters.coherent_object import CoherentObject

obj = CoherentObject(bucket="briefs", key_prefix="ws/")
data, token = obj.read("ws-42")
obj.commit("ws-42", expected_token=token, new_bytes=revise(data))
```

### Provision the bucket down to what the binding needs

`CoherentObject` emits (never applies) two verified policy shapes:

- `least_privilege_iam_policy(...)` — a role policy scoped to the exact
  key/prefix ARN with `s3:GetObject` + `s3:PutObject` only, and explicit denies
  (no `s3:*`, no `s3:DeleteObject`).
- `conditional_write_bucket_policy(...)` — an **owner-managed** bucket policy that
  *requires* conditional writes (`Deny s3:PutObject` when `s3:if-match` is null),
  so a writer that skips the `If-Match` guard is refused by the bucket itself.

---

## 4. Wiring several artifacts declaratively (the manifest)

Instead of constructing bindings in code, a **Coherence Manifest** binds each
artifact id to a substrate, a connection, a credential *reference*, and a tier.
It is a named trust boundary: every security decision is taken at load time,
before any driver is built.

```yaml
# see docs/examples/manifest.example.yaml for the full annotated form
artifacts:
  - id: workspace-row
    substrate: postgres
    tier: native-cas
    version_source: trigger-managed version column
    connection:
      dsn: "postgresql://coherence_writer@db.internal/app"
      credential: "secret-file:/run/secrets/pg"     # a reference, never a literal
```

```python
from ccs.adapters.substrate_manifest import load_manifest

manifest = load_manifest("coherence.yaml")   # SSRF + credential-form + TLS checks run HERE
manifest.dry_run()                            # prints each artifact's tier before anything is touched
```

Load-time guarantees (see [the security guide](../security.md#outbound-network-destinations)
for the full posture):

- **Credentials are references, never literals** — a `dsn` (or S3 `endpoint_url`)
  carrying an inline password/userinfo is refused.
- **Connection targets are SSRF-constrained** — the deny runs on the *resolved*
  address (cloud-metadata, link-local, and carrier-NAT ranges are hard-denied;
  private ranges require an explicit `CCS_SUBSTRATE_ALLOW_PRIVATE=1` opt-in).
- **Plaintext to a routable host is refused** unless you acknowledge it with
  `CCS_SUBSTRATE_INSECURE=1`.

---

## 5. What happens when two agents race

| You see | It means | You do |
|---|---|---|
| `StaleView` | a peer committed since your read; your cached view is stale | `reacquire()` for fresh bytes, recompute, retry |
| `CasVersionConflict` | the CAS was refused — read `.reason` for which of the four: `version_mismatch` (you lost the version race), `other_holder` (a peer holds the grant and the version has **not** moved), `stale_read_generation` (the claim your read was taken under was reclaimed), `caller_in_transient_state` (a peer invalidated you between read and CAS) | `version_mismatch` — re-read at the current version, recompute, retry. `other_holder` — back off and retry; a re-read returns the same bytes, so recomputing changes nothing. `stale_read_generation` — `reacquire()`, then re-read. `caller_in_transient_state` — recover with a fresh identity |
| `CommitUnconfirmed` | the write may or may not have landed (a driver/coordinator blip) | **do not blind-retry** — re-read; the binding's reconciliation decides whether it already landed |
| `ViewWedged` | the substrate moved out of band of the coordinator (e.g. a foreign edit, or a writer that died between its two commit legs) | `reacquire()` and re-decide |

The one rule that matters: **never blind-retry an unconfirmed write.** The binding
reconciles by re-reading and comparing the stored version and content fingerprint
against what you intended — so a write that *did* land is recognized as yours (no
double-apply), and one that didn't is retried cleanly.

---

## 6. Honest scope (read before you rely on it)

The full list is in [the guide](../guide.md#honest-scope); the load-bearing
points:

- **Single-host.** The cross-agent guarantee (invalidation, uniformity) is
  single-host. When the substrate is itself distributed (S3, managed Postgres),
  the no-lost-update guarantee is the *substrate's* and is identical with or
  without this layer. Do not run the S3 loop through a Multi-Region Access Point
  or a cross-Region replica, and do not place agents on two hosts against one
  distributed substrate.
- **A crash between the two commit legs is unbounded in v1.** If a writer's
  substrate write lands but its process dies before the coordinator bump, peers
  are not invalidated until their *next* read of that artifact. Repair-forward is
  on the roadmap.
- **More backends are demand-gated.** A Letta vendor-memory sidecar is a post-v1
  candidate, not shipped: Letta exposes no atomic conditional-write token, so a
  binding could only *detect* a sequential stale write (via a client-held content
  shadow), not prevent a concurrent one — it would be a `detect-only` tier, and
  it lands only when a concrete need pulls it in.

---

## 7. Prove it against a real substrate

The tier-honesty conformance suite exercises both bindings against **real**
Postgres / S3 behind the `real_substrate` pytest marker (excluded from the
default run; Moto/LocalStack are deliberately excluded because they serialize and
would false-green a concurrency test):

```bash
export CCS_TEST_PG_DSN="postgresql://…"
export CCS_REAL_S3_BUCKET="…"
pytest -m real_substrate
```
