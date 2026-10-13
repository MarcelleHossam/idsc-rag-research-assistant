#!/bin/bash
# Double-click this file to start the IDSC research assistant.
# Keep it inside the idsc-rag-v2 project folder (same level as server.py).

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
  echo "Could not find venv/ in this folder. Make sure this file sits"
  echo "directly inside your idsc-rag-v2 project folder, next to server.py."
  read -p "Press Enter to close..."
  exit 1
fi

source venv/bin/activate

echo "Starting the PDF server (for citation links)..."
python serve_pdfs.py &
PDF_PID=$!

echo "Starting the assistant server..."
python server.py &
SERVER_PID=$!

# Give the server a moment to come up before opening the browser
sleep 2
open http://127.0.0.1:5000

echo ""
echo "IDSC assistant is running."
echo "Leave this window open while you use it in the browser."
echo "Close this window (or press Ctrl+C) to stop everything."
echo ""

trap "echo; echo 'Stopping...'; kill $PDF_PID $SERVER_PID 2>/dev/null; exit" INT TERM
wait
