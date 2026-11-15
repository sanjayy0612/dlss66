extends Node3D

const SIZE := Vector2i(256, 144)
const LAYOUTS := 40
const FRAMES := 12
var output_dir := "res://dataset/raw"

var world_root: Node3D
var camera: Camera3D
var environment: Environment
var sun: DirectionalLight3D
var probe
var manifest: FileAccess
var variant: String
var task: String
var shadow_atlas_size: int

func _ready() -> void:
	get_window().size = SIZE
	variant = OS.get_environment("DLSS66_VARIANT")
	task = OS.get_environment("DLSS66_TASK") if OS.has_environment("DLSS66_TASK") else "shadow_off"
	output_dir = "res://" + OS.get_environment("DLSS66_OUTPUT") if OS.has_environment("DLSS66_OUTPUT") else output_dir
	if variant != "cheap" and variant != "reference":
		push_error("Set DLSS66_VARIANT to cheap or reference")
		get_tree().quit(1)
		return
	if task != "shadow_off" and task != "shadow_resolution":
		push_error("Unknown dataset task: " + task)
		get_tree().quit(1)
		return
	shadow_atlas_size = (512 if variant == "cheap" else 4096) if task == "shadow_resolution" else 4096
	RenderingServer.directional_shadow_atlas_set_size(shadow_atlas_size, true)
	world_root = Node3D.new()
	add_child(world_root)
	camera = Camera3D.new()
	camera.near = 0.05
	camera.far = 100.0
	add_child(camera)
	camera.current = true
	environment = Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color(0.08, 0.09, 0.11)
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = Color(0.42, 0.45, 0.5)
	environment.ambient_light_energy = 0.4
	environment.ssao_enabled = true
	environment.ssao_intensity = 0.0
	camera.environment = environment
	if OS.get_cmdline_user_args().has("--live"):
		var live_ok := await run_live()
		if not live_ok and probe != null:
			camera.compositor = null
			RenderingServer.call_on_render_thread(probe.cleanup)
			await get_tree().process_frame
		get_tree().quit(0 if live_ok else 1)
		return
	if OS.has_environment("DLSS66_BENCHMARK"):
		await benchmark_shadow_atlas()
		get_tree().quit(0)
		return
	DirAccess.make_dir_recursive_absolute(output_dir)
	manifest = FileAccess.open(output_dir + "/manifest_" + variant + ".jsonl", FileAccess.WRITE)
	if manifest == null:
		push_error("Cannot open dataset manifest")
		get_tree().quit(1)
		return
	if variant == "cheap":
		var compositor := Compositor.new()
		probe = preload("res://dataset_probe.gd").new()
		compositor.compositor_effects = [probe]
		camera.compositor = compositor
	var layouts := mini(LAYOUTS, maxi(1, int(OS.get_environment("DLSS66_LAYOUTS")) if OS.has_environment("DLSS66_LAYOUTS") else LAYOUTS))
	var frames := mini(FRAMES, maxi(1, int(OS.get_environment("DLSS66_FRAMES")) if OS.has_environment("DLSS66_FRAMES") else FRAMES))
	var scene_start := maxi(0, int(OS.get_environment("DLSS66_SCENE_START")) if OS.has_environment("DLSS66_SCENE_START") else 0)
	for scene_id in range(scene_start, scene_start + layouts):
		make_layout(scene_id)
		var scene_seed := (2000 if task == "shadow_resolution" else 1000) + scene_id
		DirAccess.make_dir_recursive_absolute(output_dir + "/scene_%02d" % scene_id)
		for frame_id in range(frames):
			var angle := 0.2 + TAU * float(frame_id) / float(FRAMES)
			camera.position = Vector3(cos(angle) * 3.1, 2.7, sin(angle) * 3.1)
			camera.look_at(Vector3(0, 0.65, 0))
			var light_view := camera.global_transform.basis.inverse() * (-sun.global_transform.basis.z)
			var stem := "scene_%02d/frame_%03d" % [scene_id, frame_id]
			for warmup in range(2):
				await get_tree().process_frame
			var geometry_path := stem + "_geometry.rgba32f"
			if variant == "cheap":
				probe.saved = false
				RenderingServer.call_on_render_thread(probe.arm.bind(output_dir + "/" + geometry_path))
				for wait_frame in range(90):
					RenderingServer.force_draw()
					await get_tree().process_frame
					if probe.saved:
						break
				if not probe.saved:
					push_error("Geometry export timed out: " + geometry_path)
					manifest.close()
					camera.compositor = null
					RenderingServer.call_on_render_thread(probe.cleanup)
					await get_tree().process_frame
					get_tree().quit(1)
					return
			RenderingServer.force_draw()
			var image := get_viewport().get_texture().get_image()
			var path: String = output_dir + "/" + stem + "_" + variant + ".png"
			var err := image.save_png(path)
			if err != OK or image.get_size() != SIZE:
				push_error("Capture failed: " + path)
				manifest.close()
				if variant == "cheap":
					camera.compositor = null
					RenderingServer.call_on_render_thread(probe.cleanup)
					await get_tree().process_frame
				get_tree().quit(1)
				return
			var record := {"scene_id": scene_id, "frame_id": frame_id,
				"seed": scene_seed, "camera_position": [camera.position.x, camera.position.y, camera.position.z],
				"look_at": [0.0, 0.65, 0.0], "variant": variant, "image": stem + "_" + variant + ".png",
				"geometry": geometry_path if variant == "cheap" else "",
				"camera_near": camera.near, "camera_far": camera.far,
				"light_rotation_degrees": [sun.rotation_degrees.x, sun.rotation_degrees.y, sun.rotation_degrees.z],
				"light_direction_view": [light_view.x, light_view.y, light_view.z],
				"shadow_atlas_size": shadow_atlas_size,
				"effect": "Shadow atlas 512 versus 4096" if task == "shadow_resolution" else "Directional shadows off versus on",
				"resolution": [SIZE.x, SIZE.y]}
			manifest.store_line(JSON.stringify(record))
			manifest.flush()
		print("DATASET_LAYOUT variant=", variant, " scene=", scene_id, " frames=", frames)
	manifest.close()
	if variant == "cheap":
		camera.compositor = null
		RenderingServer.call_on_render_thread(probe.cleanup)
		await get_tree().process_frame
	print("DATASET_COMPLETE variant=", variant, " frames=", layouts * frames)
	get_tree().quit(0)

func run_live() -> bool:
	# A local experimental bridge. The 3D viewport stays separate from the output overlay.
	var source_viewport := SubViewport.new()
	source_viewport.size = SIZE
	source_viewport.own_world_3d = true
	source_viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(source_viewport)
	remove_child(world_root)
	source_viewport.add_child(world_root)
	remove_child(camera)
	source_viewport.add_child(camera)
	make_layout(28)
	var compositor := Compositor.new()
	probe = preload("res://dataset_probe.gd").new()
	compositor.compositor_effects = [probe]
	camera.compositor = compositor
	camera.current = true
	var layer := CanvasLayer.new()
	add_child(layer)
	var output_sprite := Sprite2D.new()
	output_sprite.centered = false
	layer.add_child(output_sprite)
	var output_texture: ImageTexture
	var peer := StreamPeerTCP.new()
	var port := int(OS.get_environment("DLSS66_LIVE_PORT")) if OS.has_environment("DLSS66_LIVE_PORT") else 8766
	if peer.connect_to_host("127.0.0.1", port) != OK:
		push_error("Cannot connect to inference server")
		return false
	for wait_index in range(120):
		peer.poll()
		if peer.get_status() == StreamPeerTCP.STATUS_CONNECTED:
			break
		await get_tree().process_frame
	if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		push_error("Inference connection timed out")
		return false
	peer.set_no_delay(true)
	var frame_count := int(OS.get_environment("DLSS66_LIVE_FRAMES")) if OS.has_environment("DLSS66_LIVE_FRAMES") else 24
	var measurements: Array[Dictionary] = []
	for frame_id in range(frame_count):
		var frame_start := Time.get_ticks_usec()
		var angle := 0.2 + TAU * float(frame_id % FRAMES) / float(FRAMES)
		camera.position = Vector3(cos(angle) * 3.1, 2.7, sin(angle) * 3.1)
		camera.look_at(Vector3(0, 0.65, 0))
		for _warmup in range(2):
			RenderingServer.force_draw()
			await get_tree().process_frame
		RenderingServer.call_on_render_thread(probe.arm.bind(""))
		var geometry_ready := false
		for _wait in range(90):
			RenderingServer.force_draw()
			await get_tree().process_frame
			if probe.saved:
				geometry_ready = true
				break
		if not geometry_ready:
			push_error("Live geometry readback timed out")
			return false
		var export_us := Time.get_ticks_usec()
		RenderingServer.force_draw()
		var rgb_image := source_viewport.get_texture().get_image()
		rgb_image.convert(Image.FORMAT_RGB8)
		var rgb_bytes := rgb_image.get_data()
		if rgb_bytes.size() != SIZE.x * SIZE.y * 3 or probe.latest_bytes.size() != SIZE.x * SIZE.y * 16:
			push_error("Live frame buffer has wrong size")
			return false
		var rgb_us := Time.get_ticks_usec()
		var light_view := camera.global_transform.basis.inverse() * (-sun.global_transform.basis.z)
		var light := PackedFloat32Array([light_view.x, light_view.y, light_view.z]).to_byte_array()
		if peer.put_data(rgb_bytes) != OK or peer.put_data(probe.latest_bytes) != OK or peer.put_data(light) != OK:
			push_error("Inference request failed")
			return false
		var response := PackedByteArray()
		for _receive in range(120):
			peer.poll()
			var available := peer.get_available_bytes()
			if available > 0:
				var chunk := peer.get_data(mini(available, SIZE.x * SIZE.y * 3 - response.size()))
				if chunk[0] != OK:
					push_error("Inference response failed")
					return false
				response.append_array(chunk[1])
			if response.size() == SIZE.x * SIZE.y * 3:
				break
			await get_tree().process_frame
		if response.size() != SIZE.x * SIZE.y * 3:
			push_error("Inference response timed out")
			return false
		var inference_us := Time.get_ticks_usec()
		var output_image := Image.create_from_data(SIZE.x, SIZE.y, false, Image.FORMAT_RGB8, response)
		if output_texture == null:
			output_texture = ImageTexture.create_from_image(output_image)
			output_sprite.texture = output_texture
		else:
			output_texture.update(output_image)
		var upload_us := Time.get_ticks_usec()
		RenderingServer.force_draw()
		await get_tree().process_frame
		var shown_us := Time.get_ticks_usec()
		measurements.append({"id": frame_id,
			"export_ms": float(export_us - frame_start) / 1000.0,
			"rgb_ms": float(rgb_us - export_us) / 1000.0,
			"ipc_infer_ms": float(inference_us - rgb_us) / 1000.0,
			"upload_ms": float(upload_us - inference_us) / 1000.0,
			"pipeline_ms": float(shown_us - frame_start) / 1000.0})
		print("LIVE_FRAME id=", frame_id,
			" export_ms=", float(export_us - frame_start) / 1000.0,
			" rgb_ms=", float(rgb_us - export_us) / 1000.0,
			" ipc_infer_ms=", float(inference_us - rgb_us) / 1000.0,
			" upload_ms=", float(upload_us - inference_us) / 1000.0,
			" pipeline_ms=", float(shown_us - frame_start) / 1000.0)
	DirAccess.make_dir_recursive_absolute("res://artifacts")
	var measurements_file := FileAccess.open("res://artifacts/live_godot.json", FileAccess.WRITE)
	measurements_file.store_string(JSON.stringify(measurements, "  ") + "\n")
	measurements_file.close()
	var display_image := get_viewport().get_texture().get_image()
	display_image.save_png("res://artifacts/live_display.png")
	peer.disconnect_from_host()
	camera.compositor = null
	RenderingServer.call_on_render_thread(probe.cleanup)
	await get_tree().process_frame
	return true

func benchmark_shadow_atlas() -> void:
	# Reuse the dataset scene and viewport, without readback or PNG encoding.
	make_layout(28)
	var effect_mode := OS.get_environment("DLSS66_BENCHMARK_EFFECT")
	var pcss_mode := effect_mode == "pcss"
	var ssil_mode := effect_mode == "ssil"
	if ssil_mode:
		RenderingServer.environment_set_ssil_quality(RenderingServer.ENV_SSIL_QUALITY_ULTRA, false, 1.0, 2, 0.0, 50.0)
	camera.position = Vector3(cos(0.2) * 3.1, 2.7, sin(0.2) * 3.1)
	camera.look_at(Vector3(0, 0.65, 0))
	var viewport_rid := get_viewport().get_viewport_rid()
	RenderingServer.viewport_set_measure_render_time(viewport_rid, true)
	for phase in range(4):
		var atlas_size := 512 if pcss_mode or ssil_mode or phase % 2 == 0 else 4096
		var angular_distance := 0.5 if pcss_mode and phase % 2 == 1 else 0.0
		RenderingServer.directional_shadow_atlas_set_size(atlas_size, true)
		sun.light_angular_distance = angular_distance
		environment.ssil_enabled = ssil_mode and phase % 2 == 1
		for _warmup in range(40):
			RenderingServer.force_draw()
			await get_tree().process_frame
		var gpu_times: Array[float] = []
		var cpu_times: Array[float] = []
		var phase_start := Time.get_unix_time_from_system()
		for _sample in range(100):
			RenderingServer.force_draw()
			await get_tree().process_frame
			gpu_times.append(RenderingServer.viewport_get_measured_render_time_gpu(viewport_rid))
			cpu_times.append(RenderingServer.viewport_get_measured_render_time_cpu(viewport_rid))
		var phase_end := Time.get_unix_time_from_system()
		gpu_times.sort()
		cpu_times.sort()
		var gpu_sum := 0.0
		var cpu_sum := 0.0
		for value in gpu_times:
			gpu_sum += value
		for value in cpu_times:
			cpu_sum += value
		print("ATLAS_BENCHMARK size=", atlas_size,
			" pcss_angular_deg=", angular_distance,
			" ssil_enabled=", int(environment.ssil_enabled),
			" phase_start_unix_s=", phase_start, " phase_end_unix_s=", phase_end,
			" gpu_mean_ms=", gpu_sum / gpu_times.size(),
			" gpu_p50_ms=", gpu_times[50], " gpu_p95_ms=", gpu_times[95],
			" cpu_mean_ms=", cpu_sum / cpu_times.size(),
			" cpu_p50_ms=", cpu_times[50], " cpu_p95_ms=", cpu_times[95])
	RenderingServer.viewport_set_measure_render_time(viewport_rid, false)

func make_layout(scene_id: int) -> void:
	for child in world_root.get_children():
		world_root.remove_child(child)
		child.queue_free()
	var rng := RandomNumberGenerator.new()
	rng.seed = (2000 if task == "shadow_resolution" else 1000) + scene_id
	make_box(Vector3(0, -0.15, 0), Vector3(9, 0.3, 9), Color(rng.randf_range(0.35, 0.65), rng.randf_range(0.35, 0.65), rng.randf_range(0.35, 0.65)))
	make_box(Vector3(0, 1.7, -4.3), Vector3(9, 3.4, 0.3), Color(0.67, 0.61, 0.53))
	make_box(Vector3(-4.3, 1.7, 0), Vector3(0.3, 3.4, 9), Color(0.57, 0.64, 0.68))
	for i in range(rng.randi_range(6, 12)):
		var size := Vector3(rng.randf_range(0.35, 1.0), rng.randf_range(0.5, 1.8), rng.randf_range(0.35, 1.0))
		var pos := Vector3(rng.randf_range(-2.7, 2.7), size.y * 0.5, rng.randf_range(-2.7, 2.7))
		var color := Color(rng.randf_range(0.25, 0.85), rng.randf_range(0.25, 0.85), rng.randf_range(0.25, 0.85))
		match rng.randi_range(0, 2):
			0:
				make_box(pos, size, color)
			1:
				var cylinder := CylinderMesh.new()
				cylinder.height = size.y
				cylinder.top_radius = minf(size.x, size.z) * 0.5
				cylinder.bottom_radius = cylinder.top_radius
				make_mesh(cylinder, pos, color)
			2:
				var sphere := SphereMesh.new()
				sphere.radius = minf(size.x, size.z) * 0.5
				make_mesh(sphere, Vector3(pos.x, sphere.radius, pos.z), color)
	sun = DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(rng.randf_range(-70, -35), rng.randf_range(-100, 15), 0)
	sun.shadow_enabled = task == "shadow_resolution" or variant == "reference"
	world_root.add_child(sun)

func make_box(pos: Vector3, size: Vector3, color: Color) -> void:
	var mesh := BoxMesh.new()
	mesh.size = size
	make_mesh(mesh, pos, color)

func make_mesh(mesh: PrimitiveMesh, pos: Vector3, color: Color) -> void:
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	material.roughness = 0.65
	mesh.material = material
	var instance := MeshInstance3D.new()
	instance.mesh = mesh
	instance.position = pos
	world_root.add_child(instance)
