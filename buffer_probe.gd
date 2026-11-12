extends CompositorEffect

const SIZE := Vector2i(256, 144)
var rd: RenderingDevice
var pipeline: RID
var shader: RID
var sampler: RID
var output: RID
var requested := false
var saved := false
var callback_frame := 0
var request_us := 0
var submit_us := 0

func _init() -> void:
	effect_callback_type = EFFECT_CALLBACK_TYPE_POST_OPAQUE
	access_resolved_depth = true
	needs_normal_roughness = true
	RenderingServer.call_on_render_thread(_setup)

func _setup() -> void:
	rd = RenderingServer.get_rendering_device()
	var source: RDShaderFile = load("res://export.glsl")
	shader = rd.shader_create_from_spirv(source.get_spirv())
	pipeline = rd.compute_pipeline_create(shader)
	var descriptor := RDSamplerState.new()
	descriptor.min_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	descriptor.mag_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	sampler = rd.sampler_create(descriptor)
	var fmt := RDTextureFormat.new()
	fmt.width = SIZE.x
	fmt.height = SIZE.y
	fmt.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	fmt.usage_bits = RenderingDevice.TEXTURE_USAGE_STORAGE_BIT | RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	assert(rd.texture_is_format_supported_for_usage(fmt.format, fmt.usage_bits))
	output = rd.texture_create(fmt, RDTextureView.new())

func _render_callback(_type: EffectCallbackType, data: RenderData) -> void:
	callback_frame += 1
	if requested or not pipeline.is_valid() or callback_frame < 5:
		return
	var buffers := data.get_render_scene_buffers() as RenderSceneBuffersRD
	if buffers == null or buffers.get_internal_size() != SIZE:
		return
	if not buffers.has_texture("forward_clustered", "normal_roughness"):
		return
	var depth := buffers.get_depth_texture()
	var normal := buffers.get_texture("forward_clustered", "normal_roughness")
	for tex in [depth, normal]:
		var fmt := rd.texture_get_format(tex)
		if not tex.is_valid() or not (fmt.usage_bits & RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT):
			push_error("Input texture cannot be sampled")
			return
	var uniforms: Array[RDUniform] = []
	for i in 2:
		var uniform := RDUniform.new()
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
		uniform.binding = i
		uniform.add_id(sampler)
		uniform.add_id(depth if i == 0 else normal)
		uniforms.append(uniform)
	var target := RDUniform.new()
	target.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	target.binding = 2
	target.add_id(output)
	uniforms.append(target)
	var uniform_set := rd.uniform_set_create(uniforms, shader, 0)
	var start_us := Time.get_ticks_usec()
	rd.capture_timestamp("dlss66_export_begin")
	var commands := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(commands, pipeline)
	rd.compute_list_bind_uniform_set(commands, uniform_set, 0)
	rd.compute_list_dispatch(commands, 32, 18, 1)
	rd.compute_list_end()
	rd.capture_timestamp("dlss66_export_end")
	submit_us = Time.get_ticks_usec() - start_us
	request_us = Time.get_ticks_usec()
	var err := rd.texture_get_data_async(output, 0, _on_data)
	print("EXPORT_REQUEST frame=", callback_frame, " err=", err)
	requested = err == OK

func _on_data(bytes: PackedByteArray) -> void:
	var elapsed := Time.get_ticks_usec() - request_us
	if bytes.size() != SIZE.x * SIZE.y * 16:
		push_error("Unexpected byte count: " + str(bytes.size()))
		return
	DirAccess.make_dir_recursive_absolute("res://captures")
	var file := FileAccess.open("res://captures/geometry.rgba32f", FileAccess.WRITE)
	file.store_buffer(bytes)
	file.close()
	print("EXPORT_SAVED bytes=", bytes.size(), " submit_cpu_us=", submit_us, " readback_elapsed_us=", elapsed)
	var begin_gpu := -1
	var end_gpu := -1
	for i in range(rd.get_captured_timestamps_count()):
		match rd.get_captured_timestamp_name(i):
			"dlss66_export_begin": begin_gpu = rd.get_captured_timestamp_gpu_time(i)
			"dlss66_export_end": end_gpu = rd.get_captured_timestamp_gpu_time(i)
	if begin_gpu >= 0 and end_gpu >= begin_gpu:
		print("EXPORT_GPU_TIMESTAMP_DELTA=", end_gpu - begin_gpu)
	else:
		print("EXPORT_GPU_TIMESTAMP_UNAVAILABLE")
	saved = true

func cleanup() -> void:
	for rid in [output, sampler, pipeline, shader]:
		if rid.is_valid():
			rd.free_rid(rid)
