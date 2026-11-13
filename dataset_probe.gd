extends CompositorEffect

const SIZE := Vector2i(256, 144)

var rd: RenderingDevice
var shader: RID
var pipeline: RID
var sampler: RID
var output: RID
var path := ""
var pending := false
var saved := false

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

func arm(destination: String) -> void:
	path = destination
	saved = false
	pending = true

func _render_callback(_type: EffectCallbackType, data: RenderData) -> void:
	if not pending or not pipeline.is_valid():
		return
	var buffers := data.get_render_scene_buffers() as RenderSceneBuffersRD
	if buffers == null or buffers.get_internal_size() != SIZE:
		return
	if not buffers.has_texture("forward_clustered", "normal_roughness"):
		return
	var depth := buffers.get_depth_texture()
	var normal := buffers.get_texture("forward_clustered", "normal_roughness")
	for texture in [depth, normal]:
		if not texture.is_valid() or not (rd.texture_get_format(texture).usage_bits & RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT):
			push_error("Dataset geometry input cannot be sampled")
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
	var commands := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(commands, pipeline)
	rd.compute_list_bind_uniform_set(commands, uniform_set, 0)
	rd.compute_list_dispatch(commands, 32, 18, 1)
	rd.compute_list_end()
	pending = false
	var err := rd.texture_get_data_async(output, 0, _on_data)
	if err != OK:
		push_error("Dataset geometry readback failed: " + str(err))

func _on_data(bytes: PackedByteArray) -> void:
	if bytes.size() != SIZE.x * SIZE.y * 16:
		push_error("Dataset geometry byte count: " + str(bytes.size()))
		return
	var file := FileAccess.open(path, FileAccess.WRITE)
	if file == null:
		push_error("Cannot write geometry: " + path)
		return
	file.store_buffer(bytes)
	file.close()
	saved = true

func cleanup() -> void:
	for rid in [output, sampler, pipeline, shader]:
		if rid.is_valid():
			rd.free_rid(rid)
