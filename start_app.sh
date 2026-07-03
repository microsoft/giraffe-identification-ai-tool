#!/bin/bash
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/streamlit.log"
PORT="${1:-8501}"

export LD_LIBRARY_PATH="/app/conda/envs/ai4gl-base/lib/python3.13/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH}"

# Kill any existing instance on the same port
fuser -k "${PORT}/tcp" 2>/dev/null

nohup /app/conda/envs/ai4gl-base/bin/streamlit run "${SCRIPT_DIR}/app.py" \
    --server.port "${PORT}" \
    > "${LOG_FILE}" 2>&1 &

echo "App starting on port ${PORT} — logs at ${LOG_FILE}"
