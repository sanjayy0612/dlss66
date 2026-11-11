# M2 test results — 2026-11-11

Machine: MacBook Air M2, 8 GB unified memory, macOS 26.5.1. Engine: official Godot 4.7.2 stable (`ed1daf0bf`) in `tools/Godot.app`; no game assets.

## Executed

- The scene launched with `Metal 4.0 - Forward+ - Apple M2` and captured a visible 256×144 RGB PNG at `captures/scene.png`.
- The compositor callback reported valid color, depth and velocity handles. Its normal/roughness handle was absent with SSAO disabled, even though `needs_normal_roughness=true`.
- Enabling built-in SSAO with `ssao_intensity=0.0` produced a valid normal/roughness handle. This is a version-specific workaround that needs visual and cost validation before using it for training.
- Repeating the probe with Vulkan-over-Metal also lacked the normal handle while SSAO was disabled, so the issue did not appear unique to the native Metal driver.
- Direct `RenderingDevice.texture_get_data_async()` calls on all four internal textures returned error 31: their usage flags lack `TEXTURE_USAGE_CAN_COPY_FROM_BIT`. The failed export code was removed. RGB capture through `Viewport.get_texture().get_image()` works.

## Next engineering step

Add a compositor compute pass that samples the engine textures and writes compact channels into an owned texture created with `TEXTURE_USAGE_CAN_COPY_FROM_BIT`; then test asynchronous readback and exact format decoding. Measure GPU pass and transfer cost separately. Do not infer that valid texture handles imply accessible CPU data or real-time neural integration.

No model was trained. No FPS, latency, or memory-performance claim has been established.
