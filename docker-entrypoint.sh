#!/bin/sh
set -eu

: "${DISPLAY:=:99}"
export DISPLAY

if ! command -v xvfb-run >/dev/null 2>&1; then
    echo "xvfb-run is missing; install the xvfb package." >&2
    exit 1
fi

if [ ! -x "${PROCESSING_JAVA:-}" ]; then
    echo "Processing executable not found: ${PROCESSING_JAVA:-unset}" >&2
    echo "Set PROCESSING_JAVA to the processing-java executable." >&2
    exit 1
fi

exec "$@"
