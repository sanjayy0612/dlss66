# dlss66

An original three-box 3D scene for testing capture on the MacBook Air M2. No game assets or model dependencies.

Run with the Godot 4.7.2 binary in `tools/`:

```sh
tools/Godot.app/Contents/MacOS/Godot --path . -- --capture
```

Expected output: `captures/scene.png` and a `BUFFER_PROBE` line reporting color, depth, normal, and velocity texture availability. This tests a real Forward+ scene and its render-thread buffer handles. It does not yet export G-buffer pixels or train a model. See [TEST_RESULTS.md](TEST_RESULTS.md) for measured observations and the readback constraint.

The official Godot binary is kept in `tools/` and ignored by Git. The test scene uses procedural boxes and no copyrighted game assets.
