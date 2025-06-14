#!/bin/bash

# run ollama
pushd /mnt/h/ollama
./ollama serve &
sleep 5
./ollama run qwen3:8b &
sleep 5
popd

# run excel-mcp-server
pushd /mnt/h/excel-mcp-server
excel-mcp-server stdio &
sleep 5
popd

# run agent
python run_excel_mcp_server.py &
