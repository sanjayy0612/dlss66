extends Node

const MANIFEST_PATH := "res://artifacts/godot_model/manifest.json"
const EXPECTED_PARAMETER_COUNT := 80152

func _ready() -> void:
	if not FileAccess.file_exists(MANIFEST_PATH):
		_finish({"ok": false, "error": "weights are missing; run python3 -m evaluation.export_godot_weights first"})
		return
	var manifest = JSON.parse_string(FileAccess.get_file_as_string(MANIFEST_PATH))
	if not manifest is Dictionary:
		_finish({"ok": false, "error": "invalid weights manifest"})
		return
	var binary_path := MANIFEST_PATH.get_base_dir().path_join(str(manifest.get("binary_file", "weights.bin")))
	var bytes := FileAccess.get_file_as_bytes(binary_path)
	if bytes.size() != int(manifest.get("binary_size_bytes", -1)):
		_finish({"ok": false, "error": "weights file size does not match manifest"})
		return
	if int(manifest.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
		_finish({"ok": false, "error": "probe shader expects the current 80152-parameter model"})
		return
	var validation: Dictionary = manifest.get("validation_probe", {})
	var input_bytes := FileAccess.get_file_as_bytes(MANIFEST_PATH.get_base_dir().path_join(validation.get("input_file", "")))
	var expected_bytes := FileAccess.get_file_as_bytes(MANIFEST_PATH.get_base_dir().path_join(validation.get("expected_enc0_conv_file", "")))
	var expected_model_bytes := FileAccess.get_file_as_bytes(MANIFEST_PATH.get_base_dir().path_join(validation.get("expected_model_output_file", "")))
	if input_bytes.size() != 12 * 256 * 144 * 4 or expected_bytes.size() != 12 * 256 * 144 * 4 or expected_model_bytes.size() != 3 * 256 * 144 * 4:
		_finish({"ok": false, "error": "convolution reference files are missing or have the wrong shape"})
		return
	if RenderingServer.get_rendering_device() == null:
		# Headless mode has no renderer-owned device. A local device can still
		# validate the exported buffer and compute shader in isolation.
		_run_gpu_probe.call_deferred(bytes, manifest, true, input_bytes, expected_bytes, expected_model_bytes)
	else:
		RenderingServer.call_on_render_thread(_run_gpu_probe.bind(bytes, manifest, false, input_bytes, expected_bytes, expected_model_bytes))

func _run_gpu_probe(bytes: PackedByteArray, manifest: Dictionary, local_device: bool,
		probe_input: PackedByteArray, probe_expected: PackedByteArray,
		probe_model_expected: PackedByteArray) -> void:
	var rd := RenderingServer.create_local_rendering_device() if local_device else RenderingServer.get_rendering_device()
	if rd == null:
		call_deferred("_finish", {"ok": false, "error": "RenderingDevice is unavailable"})
		return
	var weights := rd.storage_buffer_create(bytes.size(), bytes,
		0,
		RenderingDevice.BUFFER_CREATION_AS_STORAGE_BIT)
	var summary_bytes := PackedByteArray()
	summary_bytes.resize(16)
	var summary_buffer := rd.storage_buffer_create(summary_bytes.size(), summary_bytes,
		0, RenderingDevice.BUFFER_CREATION_AS_STORAGE_BIT)
	var shader_file := load("res://godot/weights_checksum.glsl") as RDShaderFile
	if shader_file == null or not weights.is_valid() or not summary_buffer.is_valid():
		if weights.is_valid():
			rd.free_rid(weights)
		if summary_buffer.is_valid():
			rd.free_rid(summary_buffer)
		call_deferred("_finish", {"ok": false, "error": "failed to load the shader or GPU buffers"})
		return
	var shader := rd.shader_create_from_spirv(shader_file.get_spirv())
	var pipeline := rd.compute_pipeline_create(shader)
	var weight_uniform := RDUniform.new()
	weight_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	weight_uniform.binding = 0
	weight_uniform.add_id(weights)
	var summary_uniform := RDUniform.new()
	summary_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	summary_uniform.binding = 1
	summary_uniform.add_id(summary_buffer)
	var uniform_set := rd.uniform_set_create([weight_uniform, summary_uniform], shader, 0)
	if not shader.is_valid() or not pipeline.is_valid() or not uniform_set.is_valid():
		for rid in [uniform_set, pipeline, shader, summary_buffer, weights]:
			if rid.is_valid():
				rd.free_rid(rid)
		call_deferred("_finish", {"ok": false, "error": "failed to create the GPU compute pipeline"})
		return
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	rd.compute_list_dispatch(compute_list, 1, 1, 1)
	rd.compute_list_end()
	if local_device:
		rd.submit()
		rd.sync()
	var result := rd.buffer_get_data(summary_buffer)
	var parameter_count := int(manifest.get("parameter_count", -1))
	var measured_count := int(round(result.decode_float(12))) if result.size() >= 16 else -1
	var ok := result.size() == 16 and measured_count == parameter_count and result.decode_float(8) > 0.0
	var report := {"ok": ok, "parameters": measured_count,
		"manifest_parameters": parameter_count,
		"weight_abs_sum": result.decode_float(4) if result.size() >= 16 else 0.0,
		"weight_max_abs": result.decode_float(8) if result.size() >= 16 else 0.0}
	for rid in [uniform_set, pipeline, shader, summary_buffer]:
		if rid.is_valid():
			rd.free_rid(rid)
	if ok:
		var conv_report := _run_convolution_probe(rd, weights, manifest, local_device,
			probe_input, probe_expected)
		report.merge(conv_report, true)
		report["ok"] = bool(report["ok"]) and bool(conv_report.get("conv_ok", false))
		if report["ok"]:
			var model_report := _run_full_model_probe(rd, weights, manifest, local_device,
				probe_input, probe_model_expected)
			report.merge(model_report, true)
			report["ok"] = bool(report["ok"]) and bool(model_report.get("full_model_ok", false))
	rd.free_rid(weights)
	call_deferred("_finish", report)

func _run_convolution_probe(rd: RenderingDevice, weights: RID, manifest: Dictionary,
		local_device: bool, probe_input: PackedByteArray, probe_expected: PackedByteArray) -> Dictionary:
	var input_buffer := rd.storage_buffer_create(probe_input.size(), probe_input,
		0, RenderingDevice.BUFFER_CREATION_AS_STORAGE_BIT)
	var output_bytes := PackedByteArray()
	output_bytes.resize(probe_expected.size())
	var output_buffer := rd.storage_buffer_create(output_bytes.size(), output_bytes,
		0, RenderingDevice.BUFFER_CREATION_AS_STORAGE_BIT)
	var shader_file := load("res://godot/conv_probe.glsl") as RDShaderFile
	if shader_file == null or not input_buffer.is_valid() or not output_buffer.is_valid():
		for rid in [output_buffer, input_buffer]:
			if rid.is_valid():
				rd.free_rid(rid)
		return {"conv_ok": false, "conv_error": "failed to create convolution probe resources"}
	var shader := rd.shader_create_from_spirv(shader_file.get_spirv())
	var pipeline := rd.compute_pipeline_create(shader)
	var input_uniform := RDUniform.new()
	input_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	input_uniform.binding = 1
	input_uniform.add_id(input_buffer)
	var output_uniform := RDUniform.new()
	output_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	output_uniform.binding = 2
	output_uniform.add_id(output_buffer)
	var weight_uniform := RDUniform.new()
	weight_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	weight_uniform.binding = 0
	weight_uniform.add_id(weights)
	var uniform_set := rd.uniform_set_create([weight_uniform, input_uniform, output_uniform], shader, 0)
	if not shader.is_valid() or not pipeline.is_valid() or not uniform_set.is_valid():
		for rid in [uniform_set, pipeline, shader, output_buffer, input_buffer]:
			if rid.is_valid():
				rd.free_rid(rid)
		return {"conv_ok": false, "conv_error": "failed to create convolution compute pipeline"}
	var tensor_offsets: Dictionary = {}
	for tensor in manifest.get("tensors", []):
		tensor_offsets[tensor["name"]] = int(tensor["offset_bytes"]) >> 2
	if not tensor_offsets.has("enc0.0.weight") or not tensor_offsets.has("enc0.0.bias"):
		for rid in [uniform_set, pipeline, shader, output_buffer, input_buffer]:
			if rid.is_valid():
				rd.free_rid(rid)
		return {"conv_ok": false, "conv_error": "convolution parameters are missing from the manifest"}
	var constants := PackedByteArray()
	constants.resize(8)
	constants.encode_s32(0, tensor_offsets["enc0.0.weight"])
	constants.encode_s32(4, tensor_offsets["enc0.0.bias"])
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	rd.compute_list_set_push_constant(compute_list, constants, constants.size())
	rd.compute_list_dispatch(compute_list, 32, 18, 12)
	rd.compute_list_end()
	if local_device:
		rd.submit()
		rd.sync()
	var actual := rd.buffer_get_data(output_buffer)
	var actual_values := actual.to_float32_array()
	var expected_values := probe_expected.to_float32_array()
	var max_error := 0.0
	var error_sum := 0.0
	if actual_values.size() == expected_values.size():
		for index in range(actual_values.size()):
			var error := absf(actual_values[index] - expected_values[index])
			max_error = maxf(max_error, error)
			error_sum += error
	else:
		max_error = INF
	var mean_error := error_sum / maxf(1.0, float(expected_values.size()))
	for rid in [uniform_set, pipeline, shader, output_buffer, input_buffer]:
		if rid.is_valid():
			rd.free_rid(rid)
	return {"conv_ok": actual_values.size() == expected_values.size() and max_error <= 0.0001,
		"conv_outputs": actual_values.size(), "conv_max_abs_error": max_error,
		"conv_mean_abs_error": mean_error}

func _run_full_model_probe(rd: RenderingDevice, weights: RID, manifest: Dictionary,
		local_device: bool, probe_input: PackedByteArray,
		probe_expected: PackedByteArray) -> Dictionary:
	const ACTIVATION_BYTES := 442368 * 4
	var zero_bytes := PackedByteArray()
	zero_bytes.resize(ACTIVATION_BYTES)
	var input_buffer := rd.storage_buffer_create(probe_input.size(), probe_input,
		0, RenderingDevice.BUFFER_CREATION_AS_STORAGE_BIT)
	var buffers: Dictionary = {"input": input_buffer}
	for name in ["work_a", "work_b", "skip0", "skip1", "skip2", "mask", "strength", "output"]:
		buffers[name] = rd.storage_buffer_create(ACTIVATION_BYTES, zero_bytes,
			0, RenderingDevice.BUFFER_CREATION_AS_STORAGE_BIT)
	var shader_file := load("res://godot/conv_layer.glsl") as RDShaderFile
	var compose_file := load("res://godot/compose_shadow.glsl") as RDShaderFile
	var convolution_shader := rd.shader_create_from_spirv(shader_file.get_spirv()) if shader_file != null else RID()
	var convolution_pipeline := rd.compute_pipeline_create(convolution_shader) if convolution_shader.is_valid() else RID()
	var compose_shader := rd.shader_create_from_spirv(compose_file.get_spirv()) if compose_file != null else RID()
	var compose_pipeline := rd.compute_pipeline_create(compose_shader) if compose_shader.is_valid() else RID()
	var resources: Array[RID] = []
	for buffer in buffers.values():
		resources.append(buffer)
	resources.append(convolution_pipeline)
	resources.append(convolution_shader)
	resources.append(compose_pipeline)
	resources.append(compose_shader)
	if shader_file == null or compose_file == null or not convolution_pipeline.is_valid() or not compose_pipeline.is_valid():
		_free_rids(rd, resources)
		return {"full_model_ok": false, "full_model_error": "failed to create full-model compute pipelines"}
	var tensor_offsets: Dictionary = {}
	for tensor in manifest.get("tensors", []):
		tensor_offsets[tensor["name"]] = int(tensor["offset_bytes"]) >> 2
	var steps: Array[Dictionary] = [
		{"name":"enc0.0", "a":"input", "out":"work_a", "w":256, "h":144, "aw":256, "ah":144, "ac":11, "bc":0, "oc":12, "mode":0, "relu":1},
		{"name":"enc0.2", "a":"work_a", "out":"skip0", "w":256, "h":144, "aw":256, "ah":144, "ac":12, "bc":0, "oc":12, "mode":0, "relu":1},
		{"name":"enc1.0", "a":"skip0", "out":"work_a", "w":128, "h":72, "aw":256, "ah":144, "ac":12, "bc":0, "oc":16, "mode":1, "relu":1},
		{"name":"enc1.2", "a":"work_a", "out":"skip1", "w":128, "h":72, "aw":128, "ah":72, "ac":16, "bc":0, "oc":16, "mode":0, "relu":1},
		{"name":"enc2.0", "a":"skip1", "out":"work_a", "w":64, "h":36, "aw":128, "ah":72, "ac":16, "bc":0, "oc":24, "mode":1, "relu":1},
		{"name":"enc2.2", "a":"work_a", "out":"skip2", "w":64, "h":36, "aw":64, "ah":36, "ac":24, "bc":0, "oc":24, "mode":0, "relu":1},
		{"name":"enc3.0", "a":"skip2", "out":"work_a", "w":32, "h":18, "aw":64, "ah":36, "ac":24, "bc":0, "oc":32, "mode":1, "relu":1},
		{"name":"enc3.2", "a":"work_a", "out":"work_b", "w":32, "h":18, "aw":32, "ah":18, "ac":32, "bc":0, "oc":32, "mode":0, "relu":1},
		{"name":"mid.0", "a":"work_b", "out":"work_a", "w":32, "h":18, "aw":32, "ah":18, "ac":32, "bc":0, "oc":32, "mode":0, "relu":1},
		{"name":"mid.2", "a":"work_a", "out":"work_b", "w":32, "h":18, "aw":32, "ah":18, "ac":32, "bc":0, "oc":32, "mode":0, "relu":1},
		{"name":"dec2.0", "a":"work_b", "b":"skip2", "out":"work_a", "w":64, "h":36, "aw":32, "ah":18, "ac":32, "bc":24, "oc":24, "mode":2, "relu":1},
		{"name":"dec2.2", "a":"work_a", "out":"work_b", "w":64, "h":36, "aw":64, "ah":36, "ac":24, "bc":0, "oc":24, "mode":0, "relu":1},
		{"name":"dec1.0", "a":"work_b", "b":"skip1", "out":"work_a", "w":128, "h":72, "aw":64, "ah":36, "ac":24, "bc":16, "oc":16, "mode":2, "relu":1},
		{"name":"dec1.2", "a":"work_a", "out":"work_b", "w":128, "h":72, "aw":128, "ah":72, "ac":16, "bc":0, "oc":16, "mode":0, "relu":1},
		{"name":"dec0.0", "a":"work_b", "b":"skip0", "out":"work_a", "w":256, "h":144, "aw":128, "ah":72, "ac":16, "bc":12, "oc":12, "mode":2, "relu":1},
		{"name":"dec0.2", "a":"work_a", "out":"work_b", "w":256, "h":144, "aw":256, "ah":144, "ac":12, "bc":0, "oc":12, "mode":0, "relu":1},
		{"name":"mask_head", "a":"work_b", "out":"mask", "w":256, "h":144, "aw":256, "ah":144, "ac":12, "bc":0, "oc":1, "mode":0, "relu":0},
		{"name":"strength_head", "a":"work_b", "out":"strength", "w":256, "h":144, "aw":256, "ah":144, "ac":12, "bc":0, "oc":3, "mode":0, "relu":0},
	]
	for step in steps:
		if not tensor_offsets.has(step["name"] + ".weight") or not tensor_offsets.has(step["name"] + ".bias"):
			_free_rids(rd, resources)
			return {"full_model_ok": false, "full_model_error": "missing tensor offsets for " + step["name"]}
	var uniform_sets: Array[RID] = []
	var compute_list := rd.compute_list_begin()
	var start_us := Time.get_ticks_usec()
	for step in steps:
		var input_a: RID = buffers[step["a"]]
		var input_b: RID = buffers[step.get("b", step["a"])]
		var output: RID = buffers[step["out"]]
		if not _dispatch_layer(rd, compute_list, convolution_pipeline, convolution_shader, weights,
				input_a, input_b, output, uniform_sets, step,
				tensor_offsets[step["name"] + ".weight"], tensor_offsets[step["name"] + ".bias"]):
			rd.compute_list_end()
			_free_rids(rd, uniform_sets)
			_free_rids(rd, resources)
			return {"full_model_ok": false, "full_model_error": "failed to dispatch " + step["name"]}
	var compose_uniforms: Array[RDUniform] = []
	for binding in 4:
		var uniform := RDUniform.new()
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		uniform.binding = binding
		uniform.add_id([input_buffer, buffers["mask"], buffers["strength"], buffers["output"]][binding])
		compose_uniforms.append(uniform)
	var compose_set := rd.uniform_set_create(compose_uniforms, compose_shader, 0)
	if not compose_set.is_valid():
		rd.compute_list_end()
		_free_rids(rd, uniform_sets)
		_free_rids(rd, resources)
		return {"full_model_ok": false, "full_model_error": "failed to bind output composition buffers"}
	uniform_sets.append(compose_set)
	rd.compute_list_bind_compute_pipeline(compute_list, compose_pipeline)
	rd.compute_list_bind_uniform_set(compute_list, compose_set, 0)
	var manifest_mask_bias := float(manifest.get("mask_bias", 4.0))
	var compose_constants := PackedFloat32Array([manifest_mask_bias, float(manifest.get("max_delta", 0.6))]).to_byte_array()
	rd.compute_list_set_push_constant(compute_list, compose_constants, compose_constants.size())
	rd.compute_list_dispatch(compute_list, 32, 18, 3)
	rd.compute_list_end()
	if local_device:
		rd.submit()
		rd.sync()
	var output_bytes := rd.buffer_get_data(buffers["output"], 0, 3 * 256 * 144 * 4)
	var elapsed_ms := float(Time.get_ticks_usec() - start_us) / 1000.0
	var actual_values := output_bytes.to_float32_array()
	var expected_values := probe_expected.to_float32_array()
	var max_error := 0.0
	var error_sum := 0.0
	if actual_values.size() == expected_values.size():
		for index in range(actual_values.size()):
			var error := absf(actual_values[index] - expected_values[index])
			max_error = maxf(max_error, error)
			error_sum += error
	else:
		max_error = INF
	var mean_error := error_sum / maxf(1.0, float(expected_values.size()))
	_free_rids(rd, uniform_sets)
	_free_rids(rd, resources)
	return {"full_model_ok": actual_values.size() == expected_values.size() and max_error <= 0.0001,
		"full_model_outputs": actual_values.size(), "full_model_max_abs_error": max_error,
		"full_model_mean_abs_error": mean_error, "full_model_compute_readback_ms": elapsed_ms}

func _dispatch_layer(rd: RenderingDevice, compute_list: int, pipeline: RID, shader: RID, weights: RID,
		input_a: RID, input_b: RID, output: RID, uniform_sets: Array[RID],
		step: Dictionary, weight_offset: int, bias_offset: int) -> bool:
	var uniforms: Array[RDUniform] = []
	for binding_id in 4:
		var uniform := RDUniform.new()
		uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		uniform.binding = binding_id
		uniform.add_id([weights, input_a, input_b, output][binding_id])
		uniforms.append(uniform)
	var uniform_set := rd.uniform_set_create(uniforms, shader, 0)
	if not uniform_set.is_valid():
		return false
	uniform_sets.append(uniform_set)
	var constants := PackedInt32Array([
		step["w"], step["h"], step["aw"], step["ah"], step["ac"], step["bc"],
		step["oc"], step["mode"], step["relu"], weight_offset, bias_offset,
	]).to_byte_array()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
	rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
	rd.compute_list_set_push_constant(compute_list, constants, constants.size())
	rd.compute_list_dispatch(compute_list, ceili(float(step["w"]) / 8.0),
		ceili(float(step["h"]) / 8.0), step["oc"])
	rd.compute_list_add_barrier(compute_list)
	return true

func _free_rids(rd: RenderingDevice, rids: Array[RID]) -> void:
	for rid in rids:
		if rid.is_valid():
			rd.free_rid(rid)

func _finish(report: Dictionary) -> void:
	print("GODOT_GPU_WEIGHTS_PROBE ", JSON.stringify(report))
	get_tree().quit(0 if report.get("ok", false) else 1)
