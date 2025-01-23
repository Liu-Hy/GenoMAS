#!/bin/bash

# Set models directory
export OLLAMA_MODELS=../models

# Check and start ollama if not running
if ! pgrep -x "ollama" >/dev/null; then
    echo "Starting Ollama service..."
    nohup ollama serve > $HOME/ollama.log 2>&1 &
    echo "Waiting for Ollama service to be ready..."
    sleep 5
fi
