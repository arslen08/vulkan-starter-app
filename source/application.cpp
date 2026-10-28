#include "application.hpp"

#include <imgui.h>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace application {

namespace {

constexpr uint32_t cylinder_segments = 50;
constexpr float pi = 3.14159265358979323846f;
constexpr float floor_height = -2.5f;
constexpr float floor_half_size = 10.0f;

struct Vec3 {
	float x;
	float y;
	float z;
};

struct Vertex {
	Vec3 position;
	Vec3 normal;
	Vec3 color;
};

struct Matrix {
	std::array<float, 16> values{};
};

struct SceneUniform {
	Matrix view_projection;
	float camera_position[4];
	float ambient[4];
	float directional_direction[4];
	float directional_color[4]; 
};

struct ModelUniform {
	Matrix model;
	Matrix normal_matrix;
	float albedo[4];
	float specular[4]; 
};

struct Material {
	float albedo[3];
	float specular[3];
	float shininess;
};

enum class Projection {
	perspective,
	orthographic,
};

struct Mesh {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	uint32_t vertex_count = 0;
};

struct HostBuffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* data = nullptr;
};

struct Model {
	Mesh mesh;
	HostBuffer uniform;
	VkDescriptorSet set = VK_NULL_HANDLE;
};

constexpr size_t cylinder_index = 0;
constexpr size_t floor_index = 1;
constexpr size_t model_count = 2;

VkPipeline graphics_pipeline;
VkPipelineLayout pipeline_layout;
VkDescriptorSetLayout scene_set_layout;
VkDescriptorSetLayout model_set_layout;
VkDescriptorPool descriptor_pool;
VkDescriptorSet scene_set;
HostBuffer scene_buffer;
std::array<Model, model_count> models;

Projection projection = Projection::perspective;
float translation[3] = { 0.0f, 0.0f, 0.0f };
float rotation[3] = { 20.0f, -25.0f, 0.0f };
float scale[3] = { 1.0f, 1.0f, 1.0f };
std::array<Material, model_count> materials = {{
	{ { 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f }, 64.0f },
	{ { 0.7f, 0.7f, 0.7f }, { 0.3f, 0.3f, 0.3f }, 16.0f },
}};

float camera_position[3] = { 0.0f, 1.0f, 6.0f };
float camera_yaw = 0.0f;
float camera_pitch = -10.0f;
float camera_fov = 60.0f;
float camera_ortho_size = 3.0f;
float camera_move_speed = 4.0f;

float ambient_color[3] = { 1.0f, 1.0f, 1.0f };
float ambient_intensity = 0.12f;
bool directional_enabled = true;
float directional_direction[3] = { 0.45f, -0.8f, -0.55f };
float directional_color[3] = { 1.0f, 1.0f, 1.0f };
float directional_intensity = 0.6f;

bool animation_playing = true;
float animation_phase;
float animation_speed = 1.0f;
float trajectory_radius = 0.75f;
float trajectory_depth = 0.45f;
float trajectory_height = 0.35f;
float trajectory_frequency = 2.0f;
float animation_rotation_speed = 45.0f;
double previous_update_time;
bool has_previous_update_time;

Matrix identityMatrix() {
	Matrix result{};
	result.values[0] = 1.0f;
	result.values[5] = 1.0f;
	result.values[10] = 1.0f;
	result.values[15] = 1.0f;
	return result;
}

Matrix multiply(const Matrix& a, const Matrix& b) {
	Matrix result{};
	for (size_t column = 0; column < 4; ++column) {
		for (size_t row = 0; row < 4; ++row) {
			for (size_t inner = 0; inner < 4; ++inner) {
				result.values[column * 4 + row] +=
					a.values[inner * 4 + row] * b.values[column * 4 + inner];
			}
		}
	}
	return result;
}

Matrix translationMatrix(float x, float y, float z) {
	Matrix result = identityMatrix();
	result.values[12] = x;
	result.values[13] = y;
	result.values[14] = z;
	return result;
}

Matrix scaleMatrix(float x, float y, float z) {
	Matrix result = identityMatrix();
	result.values[0] = x;
	result.values[5] = y;
	result.values[10] = z;
	return result;
}

Matrix rotationX(float angle) {
	const float radians = angle * pi / 180.0f;
	const float c = std::cos(radians);
	const float s = std::sin(radians);
	Matrix result = identityMatrix();
	result.values[5] = c;
	result.values[6] = s;
	result.values[9] = -s;
	result.values[10] = c;
	return result;
}

Matrix rotationY(float angle) {
	const float radians = angle * pi / 180.0f;
	const float c = std::cos(radians);
	const float s = std::sin(radians);
	Matrix result = identityMatrix();
	result.values[0] = c;
	result.values[2] = -s;
	result.values[8] = s;
	result.values[10] = c;
	return result;
}

Matrix rotationZ(float angle) {
	const float radians = angle * pi / 180.0f;
	const float c = std::cos(radians);
	const float s = std::sin(radians);
	Matrix result = identityMatrix();
	result.values[0] = c;
	result.values[1] = s;
	result.values[4] = -s;
	result.values[5] = c;
	return result;
}

Matrix perspectiveMatrix(float aspect_ratio, float fov_degrees) {
	constexpr float near_plane = 0.1f;
	constexpr float far_plane = 100.0f;
	const float vertical_fov = fov_degrees * pi / 180.0f;
	const float focal_length = 1.0f / std::tan(vertical_fov * 0.5f);

	Matrix result{};
	result.values[0] = focal_length / aspect_ratio;
	result.values[5] = -focal_length;
	result.values[10] = far_plane / (near_plane - far_plane);
	result.values[11] = -1.0f;
	result.values[14] = (far_plane * near_plane) / (near_plane - far_plane);
	return result;
}

Matrix orthographicMatrix(float aspect_ratio, float half_height) {
	constexpr float near_plane = 0.1f;
	constexpr float far_plane = 100.0f;

	Matrix result = identityMatrix();
	result.values[0] = 1.0f / (half_height * aspect_ratio);
	result.values[5] = -1.0f / half_height;
	result.values[10] = 1.0f / (near_plane - far_plane);
	result.values[14] = near_plane / (near_plane - far_plane);
	return result;
}

Matrix viewMatrix() {
	return multiply(rotationX(-camera_pitch),
	                multiply(rotationY(-camera_yaw),
	                         translationMatrix(-camera_position[0], -camera_position[1],
	                                           -camera_position[2])));
}

void normalize3(const float in[3], float out[3], const float fallback[3]) {
	const float length = std::sqrt(in[0] * in[0] + in[1] * in[1] + in[2] * in[2]);
	if (length < 1e-6f) {
		std::memcpy(out, fallback, 3 * sizeof(float));
		return;
	}
	out[0] = in[0] / length;
	out[1] = in[1] / length;
	out[2] = in[2] / length;
}

void addVertex(std::vector<Vertex>& vertices, Vec3 position, Vec3 normal, Vec3 color) {
	vertices.push_back({ position, normal, color });
}

Vec3 gradientColor(Vec3 position) {
	return {
		0.5f * (position.x + 1.0f),
		0.5f * (position.y + 1.0f),
		0.5f * (position.z + 1.0f),
	};
}

void addTriangle(std::vector<Vertex>& vertices,
                 Vec3 a, Vec3 b, Vec3 c,
                 Vec3 normal_a, Vec3 normal_b, Vec3 normal_c) {
	addVertex(vertices, a, normal_a, gradientColor(a));
	addVertex(vertices, b, normal_b, gradientColor(b));
	addVertex(vertices, c, normal_c, gradientColor(c));
}

std::vector<Vertex> createCylinder() {
	std::vector<Vertex> vertices;
	vertices.reserve(cylinder_segments * 12);

	constexpr float radius = 1.0f;
	constexpr float half_height = 1.0f;

	for (uint32_t i = 0; i < cylinder_segments; ++i) {
		const float angle0 = 2.0f * pi * static_cast<float>(i) / cylinder_segments;
		const float angle1 = 2.0f * pi * static_cast<float>(i + 1) / cylinder_segments;
		const float x0 = radius * std::cos(angle0);
		const float z0 = radius * std::sin(angle0);
		const float x1 = radius * std::cos(angle1);
		const float z1 = radius * std::sin(angle1);

		const Vec3 bottom0{ x0, -half_height, z0 };
		const Vec3 top0{ x0, half_height, z0 };
		const Vec3 bottom1{ x1, -half_height, z1 };
		const Vec3 top1{ x1, half_height, z1 };
		const Vec3 normal0{ std::cos(angle0), 0.0f, std::sin(angle0) };
		const Vec3 normal1{ std::cos(angle1), 0.0f, std::sin(angle1) };

		addTriangle(vertices, bottom0, top0, top1, normal0, normal0, normal1);
		addTriangle(vertices, bottom0, top1, bottom1, normal0, normal1, normal1);

		const Vec3 top_center{ 0.0f, half_height, 0.0f };
		const Vec3 bottom_center{ 0.0f, -half_height, 0.0f };
		const Vec3 top_normal{ 0.0f, 1.0f, 0.0f };
		const Vec3 bottom_normal{ 0.0f, -1.0f, 0.0f };
		addTriangle(vertices, top_center, top1, top0, top_normal, top_normal, top_normal);
		addTriangle(vertices, bottom_center, bottom0, bottom1,
		            bottom_normal, bottom_normal, bottom_normal);
	}

	return vertices;
}

std::vector<Vertex> createFloor() {
	std::vector<Vertex> vertices;
	const Vec3 normal{ 0.0f, 1.0f, 0.0f };
	const Vec3 color{ 1.0f, 1.0f, 1.0f };
	const float h = floor_half_size;
	const Vec3 a{ -h, 0.0f, -h };
	const Vec3 b{ -h, 0.0f, h };
	const Vec3 c{ h, 0.0f, h };
	const Vec3 d{ h, 0.0f, -h };
	for (const Vec3& p : { a, b, c, a, c, d }) {
		addVertex(vertices, p, normal, color);
	}
	return vertices;
}

bool createHostBuffer(HostBuffer& buffer, VkDeviceSize size, VkBufferUsageFlags usage,
                      const char* name) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	VmaAllocationInfo allocation_result{};
	if (vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &allocation_info,
	                    &buffer.buffer, &buffer.allocation, &allocation_result) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan buffer: " << name << '\n';
		return false;
	}
	buffer.data = allocation_result.pMappedData;
	return true;
}

bool uploadToBuffer(const HostBuffer& buffer, const void* data, size_t size) {
	if (size == 0) {
		return true;
	}
	std::memcpy(buffer.data, data, size);
	if (vmaFlushAllocation(graphics::internal::context.allocator, buffer.allocation, 0, size) !=
	    VK_SUCCESS) {
		std::cerr << "Failed to flush buffer\n";
		return false;
	}
	return true;
}

bool createMesh(Mesh& mesh, const std::vector<Vertex>& vertices, const char* name) {
	HostBuffer host;
	if (!createHostBuffer(host, vertices.size() * sizeof(Vertex),
	                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, name)) {
		return false;
	}
	mesh.buffer = host.buffer;
	mesh.allocation = host.allocation;
	mesh.vertex_count = static_cast<uint32_t>(vertices.size());
	return uploadToBuffer(host, vertices.data(), vertices.size() * sizeof(Vertex));
}

bool createDescriptors() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayoutBinding scene_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &scene_binding,
	};
	if (vkCreateDescriptorSetLayout(context.device, &scene_layout_info, nullptr,
	                                &scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create scene descriptor set layout\n";
		return false;
	}

	const VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};
	if (vkCreateDescriptorSetLayout(context.device, &model_layout_info, nullptr,
	                                &model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create model descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 + model_count };
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + model_count,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};
	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) !=
	    VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo scene_set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &scene_set_layout,
	};
	if (vkAllocateDescriptorSets(context.device, &scene_set_info, &scene_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate scene descriptor set\n";
		return false;
	}

	const VkDescriptorSetLayout model_layouts[model_count] = { model_set_layout, model_set_layout };
	VkDescriptorSet model_sets[model_count] = {};
	const VkDescriptorSetAllocateInfo model_set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = model_count,
		.pSetLayouts = model_layouts,
	};
	if (vkAllocateDescriptorSets(context.device, &model_set_info, model_sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate model descriptor sets\n";
		return false;
	}

	if (!createHostBuffer(scene_buffer, sizeof(SceneUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                      "scene uniform")) {
		return false;
	}
	for (size_t i = 0; i < model_count; ++i) {
		models[i].set = model_sets[i];
		if (!createHostBuffer(models[i].uniform, sizeof(ModelUniform),
		                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, "model uniform")) {
			return false;
		}
	}

	const VkDescriptorBufferInfo buffer_infos[] = {
		{ scene_buffer.buffer, 0, sizeof(SceneUniform) },
		{ models[cylinder_index].uniform.buffer, 0, sizeof(ModelUniform) },
		{ models[floor_index].uniform.buffer, 0, sizeof(ModelUniform) },
	};
	const VkWriteDescriptorSet writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_infos[0],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = models[cylinder_index].set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_infos[1],
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = models[floor_index].set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_infos[2],
		},
	};
	vkUpdateDescriptorSets(context.device, 3, writes, 0, nullptr);
	return true;
}

bool loadShader(const char* path, VkShaderModule& shader_module) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader file: " << path << '\n';
		return false;
	}

	const std::streamsize size = file.tellg();
	if (size <= 0 || size % sizeof(uint32_t) != 0) {
		std::cerr << "Invalid SPIR-V shader file: " << path << '\n';
		return false;
	}

	std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
	file.seekg(0);
	if (!file.read(reinterpret_cast<char*>(code.data()), size)) {
		std::cerr << "Failed to read shader file: " << path << '\n';
		return false;
	}

	const VkShaderModuleCreateInfo shader_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = static_cast<size_t>(size),
		.pCode = code.data(),
	};
	if (vkCreateShaderModule(graphics::internal::context.device, &shader_info, nullptr,
	                         &shader_module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module from: " << path << '\n';
		return false;
	}
	return true;
}

bool createPipeline() {
	auto& context = graphics::internal::context;
	VkShaderModule vertex_shader = VK_NULL_HANDLE;
	VkShaderModule fragment_shader = VK_NULL_HANDLE;
	if (!loadShader("shaders/cylinder.vert.spv", vertex_shader) ||
	    !loadShader("shaders/cylinder.frag.spv", fragment_shader)) {
		if (vertex_shader != VK_NULL_HANDLE) {
			vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		}
		if (fragment_shader != VK_NULL_HANDLE) {
			vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		}
		return false;
	}

	const VkPipelineShaderStageCreateInfo shader_stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = offsetof(Vertex, position) },
		{ .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = offsetof(Vertex, normal) },
		{ .location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = offsetof(Vertex, color) },
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 3,
		.pVertexAttributeDescriptions = vertex_attributes,
	};
	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};
	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};
	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};
	const VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};
	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};
	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};
	const VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};
	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};
	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};
	const VkDescriptorSetLayout set_layouts[] = { scene_set_layout, model_set_layout };
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = set_layouts,
	};
	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		return false;
	}

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blending,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};
	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
	                                                   &pipeline_info, nullptr, &graphics_pipeline);
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);
	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan graphics pipeline\n";
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
		pipeline_layout = VK_NULL_HANDLE;
		return false;
	}
	return true;
}

void destroyHostBuffer(HostBuffer& buffer) {
	if (buffer.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics::internal::context.allocator, buffer.buffer, buffer.allocation);
	}
	buffer = HostBuffer{};
}

void destroyResources() {
	auto& context = graphics::internal::context;
	if (graphics_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, graphics_pipeline, nullptr);
		graphics_pipeline = VK_NULL_HANDLE;
	}
	if (pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
		pipeline_layout = VK_NULL_HANDLE;
	}
	for (Model& model : models) {
		if (model.mesh.buffer != VK_NULL_HANDLE) {
			vmaDestroyBuffer(context.allocator, model.mesh.buffer, model.mesh.allocation);
		}
		model.mesh = Mesh{};
		destroyHostBuffer(model.uniform);
		model.set = VK_NULL_HANDLE;
	}
	destroyHostBuffer(scene_buffer);
	if (descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
		descriptor_pool = VK_NULL_HANDLE;
		scene_set = VK_NULL_HANDLE;
	}
	if (model_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, model_set_layout, nullptr);
		model_set_layout = VK_NULL_HANDLE;
	}
	if (scene_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);
		scene_set_layout = VK_NULL_HANDLE;
	}
}


void drawCameraWindow(float delta_time) {
	ImGuiIO& io = ImGui::GetIO();

	if (!io.WantCaptureKeyboard) {
		const float yaw = camera_yaw * pi / 180.0f;
		const float pitch = camera_pitch * pi / 180.0f;
		const float forward[3] = {
			-std::sin(yaw) * std::cos(pitch),
			std::sin(pitch),
			-std::cos(yaw) * std::cos(pitch),
		};
		const float right[3] = { std::cos(yaw), 0.0f, -std::sin(yaw) };
		float move[3] = { 0.0f, 0.0f, 0.0f };
		const auto add = [&](const float direction[3], float factor) {
			for (int i = 0; i < 3; ++i) {
				move[i] += direction[i] * factor;
			}
		};
		if (ImGui::IsKeyDown(ImGuiKey_W)) add(forward, 1.0f);
		if (ImGui::IsKeyDown(ImGuiKey_S)) add(forward, -1.0f);
		if (ImGui::IsKeyDown(ImGuiKey_D)) add(right, 1.0f);
		if (ImGui::IsKeyDown(ImGuiKey_A)) add(right, -1.0f);
		if (ImGui::IsKeyDown(ImGuiKey_E)) move[1] += 1.0f;
		if (ImGui::IsKeyDown(ImGuiKey_Q)) move[1] -= 1.0f;
		const float speed = camera_move_speed * delta_time * (io.KeyShift ? 3.0f : 1.0f);
		for (int i = 0; i < 3; ++i) {
			camera_position[i] += move[i] * speed;
		}
	}
	if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
		camera_yaw -= io.MouseDelta.x * 0.15f;
		camera_pitch -= io.MouseDelta.y * 0.15f;
		camera_pitch = std::clamp(camera_pitch, -89.0f, 89.0f);
	}

	ImGui::Begin("Camera");
	ImGui::TextWrapped("WASD/QE - move, Shift - faster, right mouse button - look");

	int selected_projection = projection == Projection::perspective ? 0 : 1;
	const char* projection_modes[] = { "Perspective", "Orthographic" };
	if (ImGui::Combo("Projection", &selected_projection, projection_modes, 2)) {
		projection = selected_projection == 0 ? Projection::perspective : Projection::orthographic;
	}
	ImGui::DragFloat3("Position", camera_position, 0.05f);
	ImGui::SliderFloat("Yaw", &camera_yaw, -180.0f, 180.0f);
	ImGui::SliderFloat("Pitch", &camera_pitch, -89.0f, 89.0f);
	if (projection == Projection::perspective) {
		ImGui::SliderFloat("Field of view", &camera_fov, 20.0f, 120.0f);
	} else {
		ImGui::SliderFloat("Orthographic size", &camera_ortho_size, 0.5f, 10.0f);
	}
	ImGui::SliderFloat("Move speed", &camera_move_speed, 0.5f, 20.0f);
	if (ImGui::Button("Reset camera")) {
		camera_position[0] = 0.0f;
		camera_position[1] = 1.0f;
		camera_position[2] = 6.0f;
		camera_yaw = 0.0f;
		camera_pitch = -10.0f;
		camera_fov = 60.0f;
		camera_ortho_size = 3.0f;
	}
	ImGui::End();
}

void drawLightingWindow() {
	ImGui::Begin("Lighting");

	ImGui::SeparatorText("Ambient light");
	ImGui::ColorEdit3("Ambient color", ambient_color);
	ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0.0f, 1.0f);

	ImGui::SeparatorText("Directional light");
	ImGui::Checkbox("Enabled", &directional_enabled);
	ImGui::DragFloat3("Direction", directional_direction, 0.01f, -1.0f, 1.0f);
	ImGui::ColorEdit3("Color##directional", directional_color);
	ImGui::SliderFloat("Intensity##directional", &directional_intensity, 0.0f, 5.0f);

	ImGui::End();
}

void drawMaterialEditor(const char* title, Material& material) {
	ImGui::SeparatorText(title);
	ImGui::PushID(title);
	ImGui::ColorEdit3("Albedo", material.albedo);
	ImGui::ColorEdit3("Specular color", material.specular);
	ImGui::SliderFloat("Shininess", &material.shininess, 1.0f, 512.0f, "%.1f",
	                   ImGuiSliderFlags_Logarithmic);
	ImGui::PopID();
}

void drawObjectWindow() {
	ImGui::Begin("Cylinder - variant 8");
	ImGui::Text("Right circular cylinder, 50 segments per base");

	ImGui::SeparatorText("Translation");
	ImGui::SliderFloat3("X / Y / Z", translation, -2.0f, 2.0f);
	ImGui::SeparatorText("Rotation (degrees)");
	ImGui::SliderFloat3("X / Y / Z##rotation", rotation, -180.0f, 180.0f);
	ImGui::SeparatorText("Scale");
	ImGui::SliderFloat3("X / Y / Z##scale", scale, 0.2f, 2.0f);
	if (ImGui::Button("Reset transformations")) {
		translation[0] = translation[1] = translation[2] = 0.0f;
		rotation[0] = 20.0f;
		rotation[1] = -25.0f;
		rotation[2] = 0.0f;
		scale[0] = scale[1] = scale[2] = 1.0f;
	}

	drawMaterialEditor("Cylinder material", materials[cylinder_index]);
	drawMaterialEditor("Floor material", materials[floor_index]);

	ImGui::SeparatorText("Animation trajectory");
	if (ImGui::Button(animation_playing ? "Pause animation" : "Play animation")) {
		animation_playing = !animation_playing;
	}
	ImGui::SameLine();
	if (ImGui::Button("Restart trajectory")) {
		animation_phase = 0.0f;
	}
	ImGui::SliderFloat("Speed", &animation_speed, 0.0f, 3.0f);
	ImGui::SliderFloat("Horizontal radius", &trajectory_radius, 0.0f, 2.0f);
	ImGui::SliderFloat("Depth radius", &trajectory_depth, 0.0f, 1.5f);
	ImGui::SliderFloat("Vertical amplitude", &trajectory_height, 0.0f, 1.0f);
	ImGui::SliderFloat("Vertical frequency", &trajectory_frequency, 0.25f, 4.0f);
	ImGui::SliderFloat("Rotation speed", &animation_rotation_speed, 0.0f, 180.0f);
	ImGui::End();
}

void fillModelUniform(ModelUniform& uniform, const Matrix& model, const Matrix& normal_matrix,
                      const Material& material) {
	uniform.model = model;
	uniform.normal_matrix = normal_matrix;
	for (int i = 0; i < 3; ++i) {
		uniform.albedo[i] = material.albedo[i];
		uniform.specular[i] = material.specular[i];
	}
	uniform.albedo[3] = 1.0f;
	uniform.specular[3] = material.shininess;
}


bool initialize() {
	if (!createMesh(models[cylinder_index].mesh, createCylinder(), "cylinder vertices") ||
	    !createMesh(models[floor_index].mesh, createFloor(), "floor vertices") ||
	    !createDescriptors() || !createPipeline()) {
		destroyResources();
		return false;
	}
	return true;
}

void shutdown() {
	vkQueueWaitIdle(graphics::internal::context.graphics_queue);
	destroyResources();
}

void update(double time) {
	float delta_time = 0.0f;
	if (has_previous_update_time) {
		delta_time = std::clamp(static_cast<float>(time - previous_update_time), 0.0f, 0.1f);
	}
	previous_update_time = time;
	has_previous_update_time = true;

	drawCameraWindow(delta_time);
	drawLightingWindow();
	drawObjectWindow();

	if (animation_playing) {
		animation_phase += delta_time * animation_speed;
	}
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;
	if (fd.command_buffer == VK_NULL_HANDLE || fd.framebuffer == VK_NULL_HANDLE ||
	    context.swapchain_extent.height == 0) {
		return;
	}

	const float aspect_ratio = static_cast<float>(context.swapchain_extent.width) /
	                           static_cast<float>(context.swapchain_extent.height);
	const Matrix projection_matrix = projection == Projection::perspective
		? perspectiveMatrix(aspect_ratio, camera_fov)
		: orthographicMatrix(aspect_ratio, camera_ortho_size);

	const float default_direction[3] = { 0.0f, -1.0f, 0.0f };
	SceneUniform scene_data{};
	scene_data.view_projection = multiply(projection_matrix, viewMatrix());
	for (int i = 0; i < 3; ++i) {
		scene_data.camera_position[i] = camera_position[i];
		scene_data.ambient[i] = ambient_color[i] * ambient_intensity;
		scene_data.directional_color[i] = directional_color[i];
	}
	normalize3(directional_direction, scene_data.directional_direction, default_direction);
	scene_data.directional_color[3] = directional_enabled ? directional_intensity : 0.0f;

	const float animated_x = trajectory_radius * std::cos(animation_phase);
	const float animated_y = trajectory_height * std::sin(trajectory_frequency * animation_phase);
	const float animated_z = trajectory_depth * std::sin(2.0f * animation_phase);
	const Matrix rotation_matrix = multiply(
		rotationZ(rotation[2]),
		multiply(rotationY(rotation[1] + animation_phase * animation_rotation_speed),
		         rotationX(rotation[0])));
	const Matrix cylinder_model = multiply(
		translationMatrix(translation[0] + animated_x,
		                  translation[1] + animated_y,
		                  translation[2] + animated_z),
		multiply(rotation_matrix, scaleMatrix(scale[0], scale[1], scale[2])));
	const Matrix cylinder_normal_matrix = multiply(
		rotation_matrix, scaleMatrix(1.0f / scale[0], 1.0f / scale[1], 1.0f / scale[2]));

	ModelUniform cylinder_uniform{};
	fillModelUniform(cylinder_uniform, cylinder_model, cylinder_normal_matrix,
	                 materials[cylinder_index]);
	ModelUniform floor_uniform{};
	fillModelUniform(floor_uniform, translationMatrix(0.0f, floor_height, 0.0f), identityMatrix(),
	                 materials[floor_index]);

	if (!uploadToBuffer(scene_buffer, &scene_data, sizeof(scene_data)) ||
	    !uploadToBuffer(models[cylinder_index].uniform, &cylinder_uniform,
	                    sizeof(cylinder_uniform)) ||
	    !uploadToBuffer(models[floor_index].uniform, &floor_uniform, sizeof(floor_uniform))) {
		return;
	}

	const VkCommandBufferBeginInfo command_buffer_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	if (vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin) != VK_SUCCESS) {
		std::cerr << "Failed to begin Vulkan command buffer\n";
		return;
	}

	const VkClearValue clear_values[] = {
		{ .color = { .float32 = { 0.055f, 0.07f, 0.11f, 1.0f } } },
		{ .depthStencil = { 1.0f, 0 } },
	};
	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.width = static_cast<float>(context.swapchain_extent.width),
		.height = static_cast<float>(context.swapchain_extent.height),
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor = { .extent = context.swapchain_extent };
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        0, 1, &scene_set, 0, nullptr);

	for (const Model& model : models) {
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        1, 1, &model.set, 0, nullptr);
		const VkDeviceSize vertex_offset = 0;
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &model.mesh.buffer, &vertex_offset);
		vkCmdDraw(fd.command_buffer, model.mesh.vertex_count, 1, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);
	if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to finish Vulkan command buffer\n";
	}
}

}