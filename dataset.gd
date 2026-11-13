extends Node3D

const SIZE := Vector2i(256, 144)
const LAYOUTS := 16
const FRAMES := 12
const OUTPUT := "res://dataset/raw"

var world_root: Node3D
var camera: Camera3D
var environment: Environment
var sun: DirectionalLight3D
var probe
var manifest: FileAccess
var variant: String

func _ready() -> void:
	get_window().size = SIZE
	variant = OS.get_environment("DLSS66_VARIANT")
	if variant != "cheap" and variant != "reference":
		push_error("Set DLSS66_VARIANT to cheap or reference")
		get_tree().quit(1)
		return
	DirAccess.make_dir_recursive_absolute(OUTPUT)
	manifest = FileAccess.open(OUTPUT + "/manifest_" + variant + ".jsonl", FileAccess.WRITE)
	if manifest == null:
		push_error("Cannot open dataset manifest")
		get_tree().quit(1)
		return
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
	if variant == "cheap":
		var compositor := Compositor.new()
		probe = preload("res://dataset_probe.gd").new()
		compositor.compositor_effects = [probe]
		camera.compositor = compositor
	var layouts := mini(LAYOUTS, maxi(1, int(OS.get_environment("DLSS66_LAYOUTS")) if OS.has_environment("DLSS66_LAYOUTS") else LAYOUTS))
	var frames := mini(FRAMES, maxi(1, int(OS.get_environment("DLSS66_FRAMES")) if OS.has_environment("DLSS66_FRAMES") else FRAMES))
	for scene_id in range(layouts):
		make_layout(scene_id)
		DirAccess.make_dir_recursive_absolute(OUTPUT + "/scene_%02d" % scene_id)
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
				RenderingServer.call_on_render_thread(probe.arm.bind(OUTPUT + "/" + geometry_path))
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
			var path: String = OUTPUT + "/" + stem + "_" + variant + ".png"
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
				"seed": 1000 + scene_id, "camera_position": [camera.position.x, camera.position.y, camera.position.z],
				"look_at": [0.0, 0.65, 0.0], "variant": variant, "image": stem + "_" + variant + ".png",
				"geometry": geometry_path if variant == "cheap" else "",
				"camera_near": camera.near, "camera_far": camera.far,
				"light_rotation_degrees": [sun.rotation_degrees.x, sun.rotation_degrees.y, sun.rotation_degrees.z],
				"light_direction_view": [light_view.x, light_view.y, light_view.z],
				"effect": "Directional shadows off versus on", "resolution": [SIZE.x, SIZE.y]}
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

func make_layout(scene_id: int) -> void:
	for child in world_root.get_children():
		world_root.remove_child(child)
		child.queue_free()
	var rng := RandomNumberGenerator.new()
	rng.seed = 1000 + scene_id
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
	sun.shadow_enabled = variant == "reference"
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
