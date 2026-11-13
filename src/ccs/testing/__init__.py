# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""In-repo deterministic substrate fakes and the packaged conformance corpus.

``ccs.testing`` ships INSIDE the package (``where = ["src"]``) so a foreign
implementation can import the corpus and the fakes without vendoring the test
tree. Registered as an ``interface``-layer namespace in
``ccs.hardening.architecture`` — the corpus imports adapters.

Current members:

- :mod:`ccs.testing.s3_local` — the deterministic S3-semantics fake
  (versioning, conditional writes, delete markers, legal hold) that
  ``CoherentObject`` runs against via its client-injection seam.
- :mod:`ccs.testing.process_harness` — the cross-process contender harness
  (guarantee-ladder plan U1): spawn-context contender processes rendezvoused at
  a real barrier (the concurrency witness), injected read→write delays, typed
  outcomes, and the declared-exemption seam (R-8) for bindings that are
  in-process-only by construction. In-process serialization satisfies no clause
  of the conformance contract; this harness is what makes a kit race probative.
  Its ``run_park_after_ack`` stops a contender right after it acks, gives the
  parent a window while the child cannot run, and resumes it on demand,
  returning what the contender produces after the resume.
- :mod:`ccs.testing.claim_ladder` — the machine-checked claim ladder
  (guarantee-ladder plan U7): frozen rungs pairing each claimed guarantee with
  its README claim-pins and the pytest node-ids that would fail without it.
  Foreign consumers see rungs and guarantee text only; node-id resolution and
  the two-way README drift guard live in the repo test tree
  (``tests/conformance/test_claim_ladder.py``). No cross-host rung exists to
  claim.
- :mod:`ccs.testing.substrate_conformance` — the conformance corpus: the
  tier-honesty kit (``ConformanceBinding``, the in-memory fake, the
  split-comparand teeth) plus the Workspace-Versioning family
  (``WorkspaceConformanceBinding``, MUST-MATCH/DECLARED scenarios, the
  ``LocalWorkspaceBinding`` reference arm, the torn-cut-blind teeth stub).
  The in-repo runner under ``tests/conformance/`` imports FROM here through a
  thin shim; the corpus never imports from the test tree.
- :mod:`ccs.testing.silent_listener` — loopback listeners that take a
  coordinator request and never answer it: one that holds every connection
  silent, and one that relays the first N requests to a real coordinator and
  then holds the rest. Both count the connections and record the requests at
  the far end, so a test sees what an operation sent without patching the
  client. Stdlib only.

The corpus imports and runs WITHOUT pytest (its raise-expectations ride an
internal shim); only scenario SKIP reporting requires it — install via the
``conformance`` extra: ``pip install 'agent-coherence[conformance]'``.
"""
