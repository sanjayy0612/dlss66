#!/bin/sh
set -eu
cd "$(dirname "$0")"
GODOT=tools/Godot.app/Contents/MacOS/Godot
"$GODOT" --headless --editor --path . --import >/dev/null
DLSS66_VARIANT=cheap "$GODOT" --path . -- --dataset
DLSS66_VARIANT=reference "$GODOT" --path . -- --dataset
python3 validate_dataset.py
