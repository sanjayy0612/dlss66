extends CompositorEffect

const WIDTH := 256
const HEIGHT := 144
const PIXELS := WIDTH * HEIGHT
const ACTIVATION_FLOATS := 442368
const WEIGHT_COUNT := 80152
const MANIFEST_PATH := "res://godot/model/manifest.json"

var near_plane := 0.05
var far_plane := 100.0
var light_direction_world := Vector3(0.365, 0.819, 0.442)

var rd: RenderingDevice
var weights: RID
var buffers: Dictionary = {}
var sampler: RID
var linear_sampler: RID
var pack_shader: RID
var pack_pipeline: RID
var conv_shader: RID
var conv_pipeline: RID
var compose_shader: RID
var compose_pipeline: RID
var apply_shader: RID
var apply_pipeline: RID
var model_steps: Array[Dictionary] = []
var offsets: Dictionary = {}
var layer_sets: Array[RID] = []
var final_set: RID
var mask_bias := 4.0
var max_delta := 0.6
var frame_index := 0
var callback_cpu_ms: Array[float] = []
var model_gpu_ms: Array[float] = []
var recorded_gpu_frames: Dictionary = {}
var cleaned := false
var ready := false

func _init() -> void:
	effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
	access_resolved_color = true
	access_resolved_depth = true
	needs_normal_roughness = true
	if not FileAccess.file_exists(MANIFEST_PATH):
		push_error("GPU shadow model weights missing; run evaluation.export_godot_weights")
		return
	var manifest = JSON.parse_string(FileAccess.get_file_as_string(MANIFEST_PATH))
	if not manifest is Dictionary or int(manifest.get("parameter_count", -1)) != WEIGHT_COUNT:
		push_error("GPU shadow model manifest is invalid or has the wrong parameter count")
		return
	var weight_path := MANIFEST_PATH.get_base_dir().path_join(str(manifest.get("binary_file", "weights.bin")))
	var weight_bytes := FileAccess.get_file_as_bytes(weight_path)
	if weight_bytes.size() != WEIGHT_COUNT * 4:
		push_error("GPU shadow model weight file has the wrong size")
		return
	mask_bias = float(manifest.get("mask_bias", 4.0))
	max_delta = float(manifest.get("max_delta", 0.6))
	RenderingServer.call_on_render_thread(_setup.bind(weight_bytes, manifest))

func _setup(weight_bytes: PackedByteArray, manifest: Dictionary) -> void:
	rd = RenderingServer.get_rendering_device()
	if rd == null:
		push_error("GPU shadow effect requires Godot's renderer-owned RenderingDevice")
		return
	weights = rd.storage_buffer_create(weight_bytes.size(), weight_bytes, 0,
		RenderingDevice.BUFFER_CREATION_AS_STORAGE_BIT)
	var zero_bytes := PackedByteArray()
	zero_bytes.resize(ACTIVATION_FLOATS * 4)
	for name in ["input", "work_a", "work_b", "skip0", "skip1", "skip2", "mask", "strength", "output"]:
		var size := PIXELS * 11 * 4 if name == "input" else ACTIVATION_FLOATS * 4
		var data := PackedByteArray()
		data.resize(size)
		buffers[name] = rd.storage_buffer_create(size, data, 0,
			RenderingDevice.BUFFER_CREATION_AS_STORAGE_BIT)
	var sampler_state := RDSamplerState.new()
	sampler_state.min_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	sampler_state.mag_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	sampler = rd.sampler_create(sampler_state)
	var linear_sampler_state := RDSamplerState.new()
	linear_sampler_state.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	linear_sampler_state.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	linear_sampler = rd.sampler_create(linear_sampler_state)
	var pack_file := load("res://godot/pack_live_input.glsl") as RDShaderFile
	var conv_file := load("res://godot/conv_layer.glsl") as RDShaderFile
	var compose_file := load("res://godot/compose_shadow.glsl") as RDShaderFile
	var apply_file := load("res://godot/apply_live_output.glsl") as RDShaderFile
	if pack_file == null or conv_file == null or compose_file == null or apply_file == null:
		push_error("GPU shadow effect shader import failed")
		return
	pack_shader = rd.shader_create_from_spirv(pack_file.get_spirv())
	pack_pipeline = rd.compute_pipeline_create(pack_shader)
	conv_shader = rd.shader_create_from_spirv(conv_file.get_spirv())
	conv_pipeline = rd.compute_pipeline_create(conv_shader)
	compose_shader = rd.shader_create_from_spirv(compose_file.get_spirv())
	compose_pipeline = rd.compute_pipeline_create(compose_shader)
	apply_shader = rd.shader_create_from_spirv(apply_file.get_spirv())
	apply_pipeline = rd.compute_pipeline_create(apply_shader)
	if not weights.is_valid() or not pack_pipeline.is_valid() or not conv_pipeline.is_valid() or not compose_pipeline.is_valid() or not apply_pipeline.is_valid():
		push_error("GPU shadow effect failed to create a pipeline or storage buffer")
		return
	for tensor in manifest.get("tensors", []):
		offsets[tensor["name"]] = int(tensor["offset_bytes"]) >> 2
	model_steps = [
		_layer("enc0.0", "input", "work_a", 256, 144, 256, 144, 11, 0, 12, 0, 1),
		_layer("enc0.2", "work_a", "skip0", 256, 144, 256, 144, 12, 0, 12, 0, 1),
		_layer("enc1.0", "skip0", "work_a", 128, 72, 256, 144, 12, 0, 16, 1, 1),
		_layer("enc1.2", "work_a", "skip1", 128, 72, 128, 72, 16, 0, 16, 0, 1),
		_layer("enc2.0", "skip1", "work_a", 64, 36, 128, 72, 16, 0, 24, 1, 1),
		_layer("enc2.2", "work_a", "skip2", 64, 36, 64, 36, 24, 0, 24, 0, 1),
		_layer("enc3.0", "skip2", "work_a", 32, 18, 64, 36, 24, 0, 32, 1, 1),
		_layer("enc3.2", "work_a", "work_b", 32, 18, 32, 18, 32, 0, 32, 0, 1),
		_layer("mid.0", "work_b", "work_a", 32, 18, 32, 18, 32, 0, 32, 0, 1),
		_layer("mid.2", "work_a", "work_b", 32, 18, 32, 18, 32, 0, 32, 0, 1),
		_layer("dec2.0", "work_b", "work_a", 64, 36, 32, 18, 32, 24, 24, 2, 1, "skip2"),
		_layer("dec2.2", "work_a", "work_b", 64, 36, 64, 36, 24, 0, 24, 0, 1),
		_layer("dec1.0", "work_b", "work_a", 128, 72, 64, 36, 24, 16, 16, 2, 1, "skip1"),
		_layer("dec1.2", "work_a", "work_b", 128, 72, 128, 72, 16, 0, 16, 0, 1),
		_layer("dec0.0", "work_b", "work_a", 256, 144, 128, 72, 16, 12, 12, 2, 1, "skip0"),
		_layer("dec0.2", "work_a", "work_b", 256, 144, 256, 144, 12, 0, 12, 0, 1),
		_layer("mask_head", "work_b", "mask", 256, 144, 256, 144, 12, 0, 1, 0, 0),
		_layer("strength_head", "work_b", "strength", 256, 144, 256, 144, 12, 0, 3, 0, 0),
	]
	for step in model_steps:
		if not offsets.has(step.name + ".weight") or not offsets.has(step.name + ".bias"):
			push_error("GPU shadow model manifest is missing " + step.name)
			return
		var set := _layer_uniform_set(step)
		if not set.is_valid():
			push_error("GPU shadow model failed to bind " + step.name)
			return
		layer_sets.append(set)
	final_set = _create_final_set()
	if not final_set.is_valid():
		push_error("GPU shadow model failed to bind output composition")
		return
	ready = true

func _layer(name: String, a: String, out: String, w: int, h: int, aw: int, ah: int,
		ac: int, bc: int, oc: int, mode: int, relu: int, b: String = "") -> Dictionary:
	return {"name": name, "a": a, "b": b if not b.is_empty() else a, "out": out,
		"w": w, "h": h, "aw": aw, "ah": ah, "ac": ac, "bc": bc,
		"oc": oc, "mode": mode, "relu": relu}

func _layer_uniform_set(step: Dictionary) -> RID:
	var uniforms: Array[RDUniform] = []
	for binding in 4:
		var uniform := RDUniform.new()
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		uniform.binding = binding
		uniform.add_id([weights, buffers[step.a], buffers[step.b], buffers[step.out]][binding])
		uniforms.append(uniform)
	return rd.uniform_set_create(uniforms, conv_shader, 0)

func _create_final_set() -> RID:
	var uniforms: Array[RDUniform] = []
	for binding in 4:
		var uniform := RDUniform.new()
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		uniform.binding = binding
		uniform.add_id([buffers.input, buffers.mask, buffers.strength, buffers.output][binding])
		uniforms.append(uniform)
	return rd.uniform_set_create(uniforms, compose_shader, 0)

func _render_callback(_type: EffectCallbackType, data: RenderData) -> void:
	if not ready:
		return
	if frame_index < 3:
		frame_index += 1
		return
	var scene_buffers := data.get_render_scene_buffers() as RenderSceneBuffersRD
	if scene_buffers == null or scene_buffers.get_internal_size().x <= 0 or scene_buffers.get_internal_size().y <= 0 or scene_buffers.get_view_count() != 1:
		return
	var color := scene_buffers.get_color_layer(0)
	var depth := scene_buffers.get_depth_texture()
	var normal := scene_buffers.get_texture("forward_clustered", "normal_roughness")
	if not color.is_valid() or not depth.is_valid() or not normal.is_valid():
		return
	var color_format := rd.texture_get_format(color)
	if color_format.format != RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT or not (color_format.usage_bits & RenderingDevice.TEXTURE_USAGE_STORAGE_BIT):
		if frame_index == 3:
			push_error("GPU shadow effect expected an rgba16f storage color buffer; found " + str(color_format.format))
		return
	var scene_data := data.get_render_scene_data()
	var camera_basis := scene_data.get_cam_transform().basis
	var light_view := camera_basis.inverse() * light_direction_world.normalized()
	var pack_uniforms: Array[RDUniform] = []
	for binding in 3:
		var uniform := RDUniform.new()
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		uniform.binding = binding
		uniform.add_id(linear_sampler if binding == 0 else sampler)
		uniform.add_id([color, depth, normal][binding])
		pack_uniforms.append(uniform)
	var input_uniform := RDUniform.new()
	input_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	input_uniform.binding = 3
	input_uniform.add_id(buffers.input)
	pack_uniforms.append(input_uniform)
	var pack_set := UniformSetCacheRD.get_cache(pack_shader, 0, pack_uniforms)
	var apply_uniforms: Array[RDUniform] = []
	var output_uniform := RDUniform.new()
	output_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	output_uniform.binding = 0
	output_uniform.add_id(buffers.output)
	apply_uniforms.append(output_uniform)
	var color_uniform := RDUniform.new()
	color_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	color_uniform.binding = 1
	color_uniform.add_id(color)
	apply_uniforms.append(color_uniform)
	var apply_set := UniformSetCacheRD.get_cache(apply_shader, 0, apply_uniforms)
	if not pack_set.is_valid() or not apply_set.is_valid() or not final_set.is_valid():
		if frame_index == 3:
			push_error("GPU shadow effect could not bind the current frame textures")
		return
	var callback_start_us := Time.get_ticks_usec()
	var timestamp_id := frame_index
	rd.capture_timestamp("neural_model_start_%d" % timestamp_id)
	var constants := PackedFloat32Array([near_plane, far_plane, light_view.x, light_view.y,
		light_view.z, 0.0, float(WIDTH), float(HEIGHT)]).to_byte_array()
	var commands := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(commands, pack_pipeline)
	rd.compute_list_bind_uniform_set(commands, pack_set, 0)
	rd.compute_list_set_push_constant(commands, constants, constants.size())
	rd.compute_list_dispatch(commands, 32, 18, 1)
	rd.compute_list_add_barrier(commands)
	for index in model_steps.size():
		var step: Dictionary = model_steps[index]
		var params := PackedInt32Array([step.w, step.h, step.aw, step.ah, step.ac, step.bc,
			step.oc, step.mode, step.relu, offsets[step.name + ".weight"], offsets[step.name + ".bias"]]).to_byte_array()
		rd.compute_list_bind_compute_pipeline(commands, conv_pipeline)
		rd.compute_list_bind_uniform_set(commands, layer_sets[index], 0)
		rd.compute_list_set_push_constant(commands, params, params.size())
		rd.compute_list_dispatch(commands, ceili(float(step.w) / 8.0), ceili(float(step.h) / 8.0), step.oc)
		rd.compute_list_add_barrier(commands)
	rd.compute_list_bind_compute_pipeline(commands, compose_pipeline)
	rd.compute_list_bind_uniform_set(commands, final_set, 0)
	var compose_constants := PackedFloat32Array([mask_bias, max_delta]).to_byte_array()
	rd.compute_list_set_push_constant(commands, compose_constants, compose_constants.size())
	rd.compute_list_dispatch(commands, 32, 18, 3)
	rd.compute_list_add_barrier(commands)
	rd.compute_list_bind_compute_pipeline(commands, apply_pipeline)
	rd.compute_list_bind_uniform_set(commands, apply_set, 0)
	var output_size := rd.texture_get_format(color)
	rd.compute_list_dispatch(commands, ceili(float(output_size.width) / 8.0),
		ceili(float(output_size.height) / 8.0), 1)
	rd.compute_list_end()
	rd.capture_timestamp("neural_model_end_%d" % timestamp_id)
	callback_cpu_ms.append(float(Time.get_ticks_usec() - callback_start_us) / 1000.0)
	frame_index += 1
	_collect_gpu_timestamps()
	if frame_index % 12 == 0:
		_print_timing_summary()

func _collect_gpu_timestamps() -> void:
	var markers: Dictionary = {}
	for index in rd.get_captured_timestamps_count():
		var name := rd.get_captured_timestamp_name(index)
		if not name.begins_with("neural_model_"):
			continue
		var parts := name.split("_")
		var timestamp_id := int(parts[parts.size() - 1])
		if not markers.has(timestamp_id):
			markers[timestamp_id] = {}
		markers[timestamp_id]["start" if name.begins_with("neural_model_start_") else "end"] = rd.get_captured_timestamp_gpu_time(index)
	for timestamp_id in markers:
		if recorded_gpu_frames.has(timestamp_id) or not markers[timestamp_id].has("start") or not markers[timestamp_id].has("end"):
			continue
		var elapsed := int(markers[timestamp_id].end) - int(markers[timestamp_id].start)
		if elapsed > 0:
			model_gpu_ms.append(float(elapsed) / 1000.0)
			recorded_gpu_frames[timestamp_id] = true

func _print_timing_summary() -> void:
	var cpu := callback_cpu_ms.duplicate()
	cpu.sort()
	var gpu := model_gpu_ms.duplicate()
	gpu.sort()
	print("GPU_SHADOW_TIMING frame=", frame_index,
		" callback_cpu_p50_ms=", _percentile(cpu, 0.50),
		" callback_cpu_p95_ms=", _percentile(cpu, 0.95),
		" model_gpu_p50_ms=", _percentile(gpu, 0.50),
		" model_gpu_p95_ms=", _percentile(gpu, 0.95),
		" model_gpu_samples=", gpu.size())

func _percentile(sorted_values: Array[float], percentile: float) -> float:
	if sorted_values.is_empty():
		return -1.0
	return sorted_values[clampi(int(ceil(float(sorted_values.size()) * percentile)) - 1, 0, sorted_values.size() - 1)]

func cleanup() -> void:
	if rd == null or cleaned:
		return
	cleaned = true
	for rid in layer_sets + [final_set, apply_pipeline, apply_shader, compose_pipeline,
		compose_shader, conv_pipeline, conv_shader, pack_pipeline, pack_shader,
		linear_sampler, sampler, weights] + buffers.values():
		if rid is RID and rid.is_valid():
			rd.free_rid(rid)
