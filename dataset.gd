extends Node3D

const SIZE := Vector2i(256, 144)
const LAYOUTS := 8
const FRAMES := 12
const OUTPUT := "res://dataset/raw"

var world_root: Node3D
var camera: Camera3D
var environment: Environment
var sun: DirectionalLight3D
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
	camera.environment = environment
	var layouts := mini(LAYOUTS, maxi(1, int(OS.get_environment("DLSS66_LAYOUTS")) if OS.has_environment("DLSS66_LAYOUTS") else LAYOUTS))
	var frames := mini(FRAMES, maxi(1, int(OS.get_environment("DLSS66_FRAMES")) if OS.has_environment("DLSS66_FRAMES") else FRAMES))
	for scene_id in range(layouts):
		make_layout(scene_id)
		DirAccess.make_dir_recursive_absolute(OUTPUT + "/scene_%02d" % scene_id)
		for frame_id in range(frames):
			var angle := 0.2 + TAU * float(frame_id) / float(FRAMES)
			camera.position = Vector3(cos(angle) * 3.1, 2.7, sin(angle) * 3.1)
			camera.look_at(Vector3(0, 0.65, 0))
			var stem := "scene_%02d/frame_%03d" % [scene_id, frame_id]
			for warmup in range(2):
				await get_tree().process_frame
			await RenderingServer.frame_post_draw
			var image := get_viewport().get_texture().get_image()
			var path: String = OUTPUT + "/" + stem + "_" + variant + ".png"
			var err := image.save_png(path)
			if err != OK or image.get_size() != SIZE:
				push_error("Capture failed: " + path)
				manifest.close()
				get_tree().quit(1)
				return
			var record := {"scene_id": scene_id, "frame_id": frame_id,
				"seed": 1000 + scene_id, "camera_position": [camera.position.x, camera.position.y, camera.position.z],
				"look_at": [0.0, 0.65, 0.0], "variant": variant, "image": stem + "_" + variant + ".png",
				"effect": "Directional shadows off versus on", "resolution": [SIZE.x, SIZE.y]}
			manifest.store_line(JSON.stringify(record))
			manifest.flush()
			print("DATASET_PAIR scene=", scene_id, " frame=", frame_id)
		print("DATASET_LAYOUT variant=", variant, " scene=", scene_id, " frames=", frames)
	manifest.close()
	print("DATASET_COMPLETE variant=", variant, " frames=", layouts * frames)
	get_tree().quit(0)

func make_layout(scene_id: int) -> void:
	for child in world_root.get_children():
		world_root.remove_child(child)
		child.queue_free()
	var rng := RandomNumberGenerator.new()
	rng.seed = 1000 + scene_id
	make_box(Vector3(0, -0.15, 0), Vector3(9, 0.3, 9), Color(0.45, 0.48, 0.52))
	make_box(Vector3(0, 1.7, -4.3), Vector3(9, 3.4, 0.3), Color(0.67, 0.61, 0.53))
	make_box(Vector3(-4.3, 1.7, 0), Vector3(0.3, 3.4, 9), Color(0.57, 0.64, 0.68))
	for i in range(8):
		var size := Vector3(rng.randf_range(0.35, 1.0), rng.randf_range(0.5, 1.8), rng.randf_range(0.35, 1.0))
		var pos := Vector3(rng.randf_range(-2.7, 2.7), size.y * 0.5, rng.randf_range(-2.7, 2.7))
		make_box(pos, size, Color(rng.randf_range(0.25, 0.85), rng.randf_range(0.25, 0.85), rng.randf_range(0.25, 0.85)))
	sun = DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-50, -20 - scene_id * 7, 0)
	sun.shadow_enabled = variant == "reference"
	world_root.add_child(sun)

func make_box(pos: Vector3, size: Vector3, color: Color) -> void:
	var mesh := BoxMesh.new()
	mesh.size = size
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	material.roughness = 0.65
	mesh.material = material
	var instance := MeshInstance3D.new()
	instance.mesh = mesh
	instance.position = pos
	world_root.add_child(instance)
