#!/usr/bin/env bash
# Which Node backend does a parity run prove against?
#
# Run by the "Resolve plugin ref" step of each ci.yml job that builds the
# plugin's coordinator. One file, so those jobs cannot pick the plugin
# branch by different rules; tests/test_parity_gate_ref_resolution.py runs
# the step with git and curl stubbed.
#
# Pinning `main` unconditionally makes a PAIRED cross-repo change
# unverifiable before merge: the sibling Node half sits on its own
# plugin branch, so every fixture the pair adds fails here until a
# plugin release promotes dev to main -- long after review. The pair is
# then merged on a maintainer's local run, which is the one place the
# parity contract must not be taken on trust.
#
# So: build the plugin branch whose name matches the branch under
# test -- the PR's head branch on pull_request (`head_ref`), the pushed
# branch on push (`ref_name`, so a push to `dev` pairs with the
# plugin's `dev` and a push to `main` with its `main`). With no
# same-named plugin branch, fall back to the plugin branch named like
# the line the change is heading for: a PR into `dev` (`base_ref`) and
# a hand-dispatched run on a feature branch -- stacked PRs get no PR
# checks, so their CI is dispatched -- pair with the plugin's `dev`,
# because that is what this repo's `dev` push will pair with after the
# merge. A PR into `main`, a dispatch on `main` and a tag fall back to
# the plugin's `main` (no plugin branch is named `v*`). An absent or
# unreachable fallback degrades to `main` rather than failing the job.
#
# Input, from the environment only: SIBLING_REF (`head_ref` or
# `ref_name`), BASE_REF, EVENT_NAME and REF_NAME. `head_ref` is
# attacker-controlled on a fork PR, so it reaches this script through
# the environment and is never interpolated into it.
# Output: `ref=<branch>` in $GITHUB_OUTPUT and one sentence in
# $GITHUB_STEP_SUMMARY.

# GitHub runs a `run:` that names no `shell:` as `bash -e {0}`: -e, and no
# pipefail. The step starts this file in a bash of its own, which does not
# inherit that -e, so it sets -e itself and, like the inline step it
# replaces, leaves pipefail off: a failed lookup inside the SHA's pipe
# below leaves the SHA empty and the parity job running, as it always has.
set -e

# The summary's em dash, spelled in octal so this file stays ASCII.
em_dash=$(printf '\342\200\224')

remote=https://github.com/Cohexa-ai/agent-coherence-plugin.git
ref=main
# Whether we FELL BACK is a property of the path taken, not of the
# value landed on: a push to this repo's `main` pairs with the
# plugin's `main` deliberately (see above), so `$ref = main` alone
# cannot tell the intended pairing from a fallback.
fellback=yes
if [ -n "$SIBLING_REF" ] &&
   git ls-remote --exit-code --heads "$remote" "$SIBLING_REF" \
     >/dev/null 2>&1; then
  ref="$SIBLING_REF"
  fellback=no
else
  # The fallback follows the integration line: `dev` for a PR into
  # `dev` or a dispatch on a non-`main` branch, `main` otherwise.
  fallback=main
  if [ "$BASE_REF" = dev ] ||
     { [ "$EVENT_NAME" = workflow_dispatch ] && [ "$REF_NAME" != main ]; }; then
    fallback=dev
  fi
  if [ "$fallback" != main ] &&
     git ls-remote --exit-code --heads "$remote" "$fallback" \
       >/dev/null 2>&1; then
    ref="$fallback"
  fi
fi
# Record the SHA too: the sibling branch can move without a commit
# here, so the ref name alone does not identify what was proven.
sha=$(git ls-remote "$remote" "refs/heads/$ref" | cut -f1)
echo "ref=$ref" >> "$GITHUB_OUTPUT"
echo "Node backend: agent-coherence-plugin@$ref ($sha)"
# When this is the FALLBACK, say how far behind it is. "Parity
# proven against `main`" is true and reads like reassurance, and a
# `main` trailing the plugin's own `dev` by a release is a weaker
# claim than it sounds -- the reader cannot tell the difference
# from the ref name, and this job's whole job is to be believable.
# Advisory only: every failure path below leaves $note empty, so a
# rate-limited or unreachable API never turns a passing parity run
# red. `|| true` is load-bearing under `set -e`.
note=""
if [ "$fellback" = yes ] && [ "$ref" = dev ]; then
  note=" $em_dash the fallback, the plugin's \`dev\` (the line this change merges into)"
elif [ "$fellback" = yes ]; then
  # `.ahead_by` on a three-dot compare is how far `dev` leads, i.e.
  # how far `main` trails. `.behind_by` would report the opposite
  # and is wrong here -- today the two are diverged (33 / 2).
  behind=$(curl -sS --max-time 10 -H 'Accept: application/vnd.github+json' \
    "https://api.github.com/repos/Cohexa-ai/agent-coherence-plugin/compare/main...dev" \
    2>/dev/null | jq -r '.ahead_by // empty' 2>/dev/null || true)
  # Three outcomes, kept distinct. Folding the first into the second
  # would let a rate-limited lookup read as reassurance -- the exact
  # bare sentence this note exists to qualify.
  case "$behind" in
    ''|*[!0-9]*) note=" $em_dash the fallback; staleness check unavailable" ;;
    0)           note=" $em_dash the fallback, level with the plugin's \`dev\`" ;;
    *)           note=" $em_dash the fallback, $behind commit(s) behind the plugin's \`dev\`" ;;
  esac
fi
echo "Parity proven against \`agent-coherence-plugin@$ref\` \`$sha\`$note." \
  >> "$GITHUB_STEP_SUMMARY"
