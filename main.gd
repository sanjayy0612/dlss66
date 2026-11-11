extends Node3D

const WIDTH := 256
const HEIGHT := 144

func _ready() -> void:
	get_window().size = Vector2i(WIDTH, HEIGHT)
	make_box(Vector3(0, -0.3, 0), Vector3(12, 0.4, 12), Color(0.45, 0.53, 0.62))
	make_box(Vector3(-1.2, 0.7, 0), Vector3(1.2, 1.6, 1.2), Color(0.84, 0.35, 0.2))
	make_box(Vector3(1.0, 0.4, -0.7), Vector3(0.9, 1.0, 0.9), Color(0.2, 0.52, 0.85))
	var light := DirectionalLight3D.new()
	light.rotation_degrees = Vector3(-55, -30, 0)
	light.shadow_enabled = true
	add_child(light)
	var camera := Camera3D.new()
	camera.position = Vector3(3, 2.5, 6)
	add_child(camera)
	camera.look_at(Vector3(0, 0.4, 0))
	var environment := Environment.new()
	environment.ssao_enabled = true
	environment.ssao_intensity = 0.0
	camera.environment = environment
	var compositor := Compositor.new()
	compositor.compositor_effects = [preload("res://buffer_probe.gd").new()]
	camera.compositor = compositor
	camera.current = true
	if OS.get_cmdline_user_args().has("--capture"):
		for i in range(30):
			camera.position.x = 3.0 + 0.01 * i
			camera.look_at(Vector3(0, 0.4, 0))
			await get_tree().process_frame
		await RenderingServer.frame_post_draw
		var image := get_viewport().get_texture().get_image()
		DirAccess.make_dir_recursive_absolute("res://captures")
		var path := "res://captures/scene.png"
		var err := image.save_png(path)
		print("CAPTURE ", path, " size=", image.get_size(), " err=", err)
		get_tree().quit(0 if err == OK else 1)

func make_box(pos: Vector3, size: Vector3, color: Color) -> void:
	var mesh := BoxMesh.new()
	mesh.size = size
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	mesh.material = material
	var instance := MeshInstance3D.new()
	instance.mesh = mesh
	instance.position = pos
	add_child(instance)
