#!/bin/sh
set -eu
cd "$(dirname "$0")"
GODOT=tools/Godot.app/Contents/MacOS/Godot
"$GODOT" --headless --editor --path . --import >/dev/null
"$GODOT" --path . -- --capture
python3 validate_capture.py
