#!/usr/bin/env bash
# Started by systemd. Reads WEIGHTS / SOURCE / EXTRA_ARGS from tracker.env (via EnvironmentFile).
set -eu
cd "$(dirname "$0")/../.."
mkdir -p logs
if [ ! -f "$WEIGHTS" ]; then
    echo "Engine $WEIGHTS not found. Build it: python3 -m export.build_engine --onnx <model>.onnx --fp16" >&2
    exit 1
fi
# word-splitting of EXTRA_ARGS is intended
# shellcheck disable=SC2086
exec /usr/bin/python3 -m deploy.run --weights "$WEIGHTS" --source "$SOURCE" $EXTRA_ARGS
