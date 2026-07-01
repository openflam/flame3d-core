#!/usr/bin/env bash
#
# reprocess.sh — re-run the pipeline for a dataset that's already been processed.
#
# Same entry point as the UI's "reprocess" button: it POSTs a config to the
# running server's /api/datasets/<name>/reprocess route, which enqueues a
# Celery job that reuses the dataset's existing data/ and outputs/ (no re-upload
# needed). Use this to re-run with tweaked parameters or a subset of steps.
#
# The config's "steps_to_run" controls which steps execute (null/empty = all).
# When running a subset, the inputs those steps need must already exist on disk
# from a previous run — see docs/Configs.md.
#
# Usage:
#   scripts/reprocess.sh --dataset <name> --config <config.json> [options]
#
#   --dataset <name>   Existing dataset to re-run.
#   --config <path>    Master config JSON to run with. Its "steps_to_run"
#                      selects the steps; "dataset_name" is overwritten by the
#                      server. See data/<name>/master_config.json for the config
#                      the dataset was last processed with.
#   --as-copy          Run on a fresh copy instead of in place; leaves the
#                      original untouched. Requires --new-name.
#   --new-name <name>  Name for the copy (only with --as-copy).
#   --host <url>       Server base URL. Default: http://localhost:5005
#   --wait             Poll job status until it completes or fails.
#
# Examples:
#   # Re-run a subset of steps in place
#   scripts/reprocess.sh --dataset kitchen \
#       --config data/kitchen/master_config.json --wait
#
#   # Try new parameters on a copy, keeping the original
#   scripts/reprocess.sh --dataset kitchen --config tweaked.json \
#       --as-copy --new-name kitchen_v2
#
set -euo pipefail

DATASET=""
CONFIG=""
AS_COPY=false
NEW_NAME=""
HOST="http://localhost:5005"
WAIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)  DATASET="$2";  shift 2 ;;
    --config)   CONFIG="$2";   shift 2 ;;
    --as-copy)  AS_COPY=true;  shift   ;;
    --new-name) NEW_NAME="$2"; shift 2 ;;
    --host)     HOST="$2";     shift 2 ;;
    --wait)     WAIT=1;        shift   ;;
    -h|--help)  sed -n '2,37p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$DATASET" ]] || { echo "Error: --dataset is required" >&2; exit 1; }
[[ -n "$CONFIG"  ]] || { echo "Error: --config is required"  >&2; exit 1; }
[[ -f "$CONFIG"  ]] || { echo "Error: config file not found: $CONFIG" >&2; exit 1; }
if [[ "$AS_COPY" == true && -z "$NEW_NAME" ]]; then
  echo "Error: --as-copy requires --new-name" >&2; exit 1
fi

# Build the JSON body: {config: {...}, as_copy: bool, new_name?: str}.
BODY="$(jq -n \
  --argjson config "$(cat "$CONFIG")" \
  --argjson as_copy "$AS_COPY" \
  --arg new_name "$NEW_NAME" \
  '{config: $config, as_copy: $as_copy}
   + (if $new_name != "" then {new_name: $new_name} else {} end)')"

echo "Reprocessing \"$DATASET\"${NEW_NAME:+ as copy \"$NEW_NAME\"} on $HOST ..."
RESPONSE="$(curl -fsS -X POST "$HOST/api/datasets/$DATASET/reprocess" \
  -H "Content-Type: application/json" \
  -d "$BODY")"

JOB_ID="$(echo "$RESPONSE" | jq -r '.job_id // empty')"
if [[ -z "$JOB_ID" ]]; then
  echo "Reprocess failed: $RESPONSE" >&2
  exit 1
fi
TARGET="$(echo "$RESPONSE" | jq -r '.dataset_name // empty')"
echo "Pipeline started for \"$TARGET\". job_id=$JOB_ID"

if [[ "$WAIT" -eq 0 ]]; then
  echo "Track it with: curl $HOST/api/jobs/$JOB_ID"
  exit 0
fi

echo "Waiting for completion (Ctrl-C to stop watching; the job keeps running) ..."
while true; do
  sleep 5
  JOB="$(curl -fsS "$HOST/api/jobs/$JOB_ID")"
  STATUS="$(echo "$JOB" | jq -r '.status')"
  echo "  status=$STATUS"
  case "$STATUS" in
    completed) echo "Done."; exit 0 ;;
    failed)    echo "Failed: $(echo "$JOB" | jq -r '.error')" >&2; exit 1 ;;
  esac
done
