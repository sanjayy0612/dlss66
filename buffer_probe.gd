extends CompositorEffect

var reported := false
var frames := 0

func _init() -> void:
	effect_callback_type = EFFECT_CALLBACK_TYPE_POST_OPAQUE
	access_resolved_color = true
	access_resolved_depth = true
	needs_normal_roughness = true
	needs_motion_vectors = true

func _render_callback(_type: EffectCallbackType, data: RenderData) -> void:
	if reported:
		return
	frames += 1
	var buffers := data.get_render_scene_buffers() as RenderSceneBuffersRD
	if buffers == null:
		return
	var color := buffers.get_texture("render_buffers", "color")
	var depth := buffers.get_texture("render_buffers", "depth")
	var has_normal := buffers.has_texture("forward_clustered", "normal_roughness")
	var velocity := buffers.get_velocity_texture()
	if not has_normal and frames < 10:
		return
	print("BUFFER_PROBE frames=", frames, " size=", buffers.get_internal_size(),
		" color=", color.is_valid(), " depth=", depth.is_valid(),
		" normal=", has_normal, " velocity=", velocity.is_valid())
	reported = true
