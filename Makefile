# Requires: pip install -e ".[langgraph,benchmark]"

.PHONY: benchmark benchmark-check cost-benchmark cost-benchmark-check tla-java-check tla-check-1 tla-check-2 tla-check-3 tla-check test-durations help

benchmark:  ## Run all three LangGraph benchmarks and write latest.json
	python tools/run_benchmarks.py

benchmark-check:  ## Check latest.json drift against expected.json (skips benchmark run)
	python tools/benchmark_drift_check.py

cost-benchmark:  ## Run the change-rate × answer-sensitivity cost sweep and write cost_sweep.json
	python tools/run_cost_sweep.py

cost-benchmark-check: cost-benchmark  ## Re-run the cost sweep, then drift-check it against expected_cost.json
	python tools/cost_drift_check.py

TLA2TOOLS := formal/tla/lib/tla2tools.jar
TLC := java -XX:+UseParallelGC -cp $(TLA2TOOLS) tlc2.TLC -workers auto

tla-java-check:  ## Fail unless Java 17+ is on PATH; every TLA+ target depends on this
	@java -version 2>&1 | head -1 | grep -qE '"(1[7-9]|[2-9][0-9])\.' || { echo "ERROR: Java 17+ required for TLC model checker"; exit 1; }

# CI runs the three bins as parallel jobs (about 4.5 min each on a slow
# runner); together they are tla-check's nine specs, with the same configs.
tla-check-1: tla-java-check  ## TLA+ bin 1: EffectGate
	$(TLC) -config formal/tla/EffectGate_CI.cfg formal/tla/EffectGate.tla

tla-check-2: tla-java-check  ## TLA+ bin 2: Retention, Snapshot, OCC
	$(TLC) -config formal/tla/Retention_CI.cfg formal/tla/Retention.tla
	$(TLC) -config formal/tla/Snapshot_CI.cfg formal/tla/Snapshot.tla
	$(TLC) -config formal/tla/OCC_CI.cfg formal/tla/OCC.tla

tla-check-3: tla-java-check  ## TLA+ bin 3: Fencing, ZombieRevoke, MESI_Standalone, CrashRecovery, WorkspaceVersion
	$(TLC) -config formal/tla/Fencing_CI.cfg formal/tla/Fencing.tla
	$(TLC) -config formal/tla/ZombieRevoke_CI.cfg formal/tla/ZombieRevoke.tla
	$(TLC) -config formal/tla/MESI_Standalone.cfg formal/tla/MESI_Standalone.tla
	$(TLC) -config formal/tla/CrashRecovery_CI.cfg formal/tla/CrashRecovery.tla
	$(TLC) -config formal/tla/WorkspaceVersion_CI.cfg formal/tla/WorkspaceVersion.tla

tla-check: tla-check-1 tla-check-2 tla-check-3  ## Run TLC model checker on MESI, CrashRecovery, OCC, Fencing, ZombieRevoke, EffectGate, Retention, Snapshot, and WorkspaceVersion specs

# Each TLC run already uses every core (-workers auto), so `make -j` would
# start three JVMs fighting for the same cores: run every target in order.
.NOTPARALLEL:

# NOTE: AtomicPublish (SB-18) is intentionally NOT in the CI sweep yet — its
# write-set state space (2^|Artifacts| x MaxVersion^|Artifacts|) needs a bounded
# encoding / state constraint to converge in the CI budget (plan Unit 1, deferred).
# The spec parses + semantically validates; wire it in once the CI config is tuned:
#   $(TLC) -config formal/tla/AtomicPublish_CI.cfg formal/tla/AtomicPublish.tla

TEST_DURATIONS := tests/.test_durations

test-durations:  ## Time the whole suite serially and rewrite tests/.test_durations, which CI uses to balance its test shards
	python -m pytest -q -p no:cacheprovider --store-durations --clean-durations --durations-path $(TEST_DURATIONS)

help:  ## List available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
