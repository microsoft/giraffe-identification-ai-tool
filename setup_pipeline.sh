# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

#!/bin/bash

# Set the SESSION_NAME variable
SESSION_NAME="streamlit_script"
LOG_DIR="./tmux_logs"
LOG_FILE="${LOG_DIR}/${SESSION_NAME}_log.txt"

# Function to send commands to the tmux session
function exec-cmd {
    tmux send-keys -t "$SESSION_NAME" "$@"
}

# Check if the Python script name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <python_script> [optional_argument]"
  exit 1
fi

SCRIPT_NAME=$1
SCRIPT_ARG=$2

# Initialize Conda
echo "Initializing Conda..."
eval "$(conda shell.bash hook)" || { echo "Failed to initialize Conda"; exit 1; }

# Ensure the log directory exists
mkdir -p $LOG_DIR

# Stop any existing sessions
echo "Stopping any existing tmux sessions..."
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Start a new tmux session
echo "Starting a new tmux session: $SESSION_NAME"
tmux new -d -s "$SESSION_NAME"

# Construct the command to run in tmux with logging
if [ -n "$SCRIPT_ARG" ]; then
  CMD="conda activate giraffe && python -u $SCRIPT_NAME $SCRIPT_ARG 2>&1 | tee $LOG_FILE && conda deactivate"
else
  CMD="conda activate giraffe && python -u $SCRIPT_NAME 2>&1 | tee $LOG_FILE && conda deactivate"
fi

# Display the command to be executed
echo "Running: $CMD"

# Execute the constructed command in the tmux session
exec-cmd "$CMD" C-m  

# Provide feedback on the tmux session start
if [ $? == 0 ]; then
  echo "Started $SCRIPT_NAME in tmux session: $SESSION_NAME"
  echo "tmux logging output saved to: $LOG_FILE"
else
  echo "Failed to start tmux session"
fi
