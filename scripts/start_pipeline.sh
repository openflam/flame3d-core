#!/usr/bin/env bash
#
# start_pipeline.sh — kick off the processing pipeline from the command line.
#
# Same entry point as the UI's "upload" button: it POSTs the data + config to
# the running server's /api/upload route, which saves them under
# data/<dataset>/ and enqueues the Celery job. Use this when the data is
# already on the machine and you'd rather not click through the UI.
#
# Works for any data source the server supports: both
# expect the capture packed as a single input.zip, which is what this uploads.
#
# Usage:
#   scripts/start_pipeline.sh --data <path> --config <config.json> [options]
#
#   --data <path>     The capture. Either a .zip (uploaded as-is) or a
#                     directory (its CONTENTS are zipped, then uploaded).
#   --config <path>   Master config JSON. Must contain "dataset_name" and
#                     "data_source". See server/default_config.json.
#   --host <url>      Server base URL. Default: http://localhost:5005
#   --wait            Poll job status until it completes or fails.
#
# Examples:
#   scripts/start_pipeline.sh --data ~/captures/kitchen.zip \
#       --config server/default_config.json --wait
#
#   scripts/start_pipeline.sh --data ~/captures/kitchen_export/ \
#       --config my_config.json
#
set -euo pipefail

DATA=""
CONFIG=""
HOST="http://localhost:5005"
WAIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)   DATA="$2";   shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --host)   HOST="$2";   shift 2 ;;
    --wait)   WAIT=1;      shift   ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

[[ -n "$DATA"   ]] || { echo "Error: --data is required"   >&2; exit 1; }
[[ -n "$CONFIG" ]] || { echo "Error: --config is required" >&2; exit 1; }
[[ -e "$DATA"   ]] || { echo "Error: data path not found: $DATA"     >&2; exit 1; }
[[ -f "$CONFIG" ]] || { echo "Error: config file not found: $CONFIG" >&2; exit 1; }

DATASET="$(jq -r '.dataset_name // empty' "$CONFIG")"
[[ -n "$DATASET" ]] || { echo "Error: config is missing \"dataset_name\"" >&2; exit 1; }

# Resolve the data to a single zip. If a directory was given, zip its contents
# (not the directory itself) into a temp file so the archive's top level is the
# capture's files — the layout the extractor expects.
CLEANUP_DIR=""
if [[ -d "$DATA" ]]; then
  # zip refuses to add to an existing file, so build the archive in a fresh
  # temp DIR (a path that doesn't exist yet), not a pre-created temp file.
  CLEANUP_DIR="$(mktemp -d)"
  ZIP="$CLEANUP_DIR/input.zip"
  echo "Zipping directory $DATA ..."
  ( cd "$DATA" && zip -r -q "$ZIP" . )
else
  ZIP="$DATA"
fi
trap '[[ -n "$CLEANUP_DIR" ]] && rm -rf "$CLEANUP_DIR"' EXIT

echo "Uploading $ZIP for dataset \"$DATASET\" to $HOST ..."
RESPONSE="$(curl -fsS -X POST "$HOST/api/upload" \
  -F "file=@$ZIP" \
  -F "config=$(cat "$CONFIG")")"

JOB_ID="$(echo "$RESPONSE" | jq -r '.job_id // empty')"
if [[ -z "$JOB_ID" ]]; then
  echo "Upload failed: $RESPONSE" >&2
  exit 1
fi
echo "Pipeline started. job_id=$JOB_ID"

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
