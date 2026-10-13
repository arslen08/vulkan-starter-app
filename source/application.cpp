#include "application.hpp"

#include <imgui.h>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace application {

namespace {

constexpr uint32_t cylinder_segments = 50;
constexpr float pi = 3.14159265358979323846f;

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

struct MaterialUniform {
	float tint[4];
};

struct Matrix {
	std::array<float, 16> values{};
};

struct PushConstants {
	Matrix mvp;
	Matrix model;
};

enum class Projection {
	perspective,
	orthographic,
};

VkPipeline graphics_pipeline;
VkPipelineLayout pipeline_layout;
VkBuffer vertex_buffer;
VmaAllocation vertex_buffer_allocation;
VkBuffer material_buffer;
VmaAllocation material_buffer_allocation;
void* material_buffer_data;
VkDescriptorSetLayout material_set_layout;
VkDescriptorPool material_descriptor_pool;
VkDescriptorSet material_descriptor_set;
uint32_t vertex_count;

Projection projection = Projection::perspective;
float translation[3] = { 0.0f, 0.0f, 0.0f };
float rotation[3] = { 20.0f, -25.0f, 0.0f };
float scale[3] = { 1.0f, 1.0f, 1.0f };
float tint[3] = { 1.0f, 1.0f, 1.0f };
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

Matrix perspectiveMatrix(float aspect_ratio) {
	constexpr float near_plane = 0.1f;
	constexpr float far_plane = 100.0f;
	constexpr float vertical_fov = 60.0f * pi / 180.0f;
	const float focal_length = 1.0f / std::tan(vertical_fov * 0.5f);

	Matrix result{};
	result.values[0] = focal_length / aspect_ratio;
	result.values[5] = -focal_length;
	result.values[10] = far_plane / (near_plane - far_plane);
	result.values[11] = -1.0f;
	result.values[14] = (far_plane * near_plane) / (near_plane - far_plane);
	return result;
}

Matrix orthographicMatrix(float aspect_ratio) {
	constexpr float half_height = 1.7f;
	constexpr float near_plane = 0.1f;
	constexpr float far_plane = 100.0f;

	Matrix result = identityMatrix();
	result.values[0] = 1.0f / (half_height * aspect_ratio);
	result.values[5] = -1.0f / half_height;
	result.values[10] = 1.0f / (near_plane - far_plane);
	result.values[14] = near_plane / (near_plane - far_plane);
	return result;
}

void addVertex(std::vector<Vertex>& vertices, Vec3 position, Vec3 normal) {
	const Vec3 color{
		0.5f * (position.x + 1.0f),
		0.5f * (position.y + 1.0f),
		0.5f * (position.z + 1.0f),
	};
	vertices.push_back({ position, normal, color });
}

void addTriangle(std::vector<Vertex>& vertices,
                 Vec3 a, Vec3 b, Vec3 c,
                 Vec3 normal_a, Vec3 normal_b, Vec3 normal_c) {
	addVertex(vertices, a, normal_a);
	addVertex(vertices, b, normal_b);
	addVertex(vertices, c, normal_c);
}

std::vector<Vertex> createCylinder() {
	std::vector<Vertex> vertices;
	vertices.reserve(cylinder_segments * 18);

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

bool createBuffer(const std::vector<Vertex>& vertices) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = vertices.size() * sizeof(Vertex),
		.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	VmaAllocationInfo allocation_result{};

	if (vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &allocation_info,
	                    &vertex_buffer, &vertex_buffer_allocation, &allocation_result) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan vertex buffer for cylinder\n";
		return false;
	}

	std::memcpy(allocation_result.pMappedData, vertices.data(), vertices.size() * sizeof(Vertex));
	vertex_count = static_cast<uint32_t>(vertices.size());
	return true;
}

bool createMaterialResources() {
	auto& context = graphics::internal::context;

	const VkDescriptorSetLayoutBinding material_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &material_binding,
	};
	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr,
	                                &material_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create material descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};
	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
	                           &material_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create material descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = material_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &material_set_layout,
	};
	if (vkAllocateDescriptorSets(context.device, &set_info, &material_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate material descriptor set\n";
		return false;
	}

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(MaterialUniform),
		.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	VmaAllocationInfo allocation_result{};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info, &material_buffer,
	                    &material_buffer_allocation, &allocation_result) != VK_SUCCESS) {
		std::cerr << "Failed to create material uniform buffer\n";
		return false;
	}
	material_buffer_data = allocation_result.pMappedData;

	const VkDescriptorBufferInfo descriptor_buffer = {
		.buffer = material_buffer,
		.offset = 0,
		.range = sizeof(MaterialUniform),
	};
	const VkWriteDescriptorSet descriptor_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = material_descriptor_set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &descriptor_buffer,
	};
	vkUpdateDescriptorSets(context.device, 1, &descriptor_write, 0, nullptr);
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
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
		{
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, color),
		},
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
	const VkPushConstantRange push_constant_range = {
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		.offset = 0,
		.size = sizeof(PushConstants),
	};
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &material_set_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constant_range,
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
	if (vertex_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_buffer_allocation);
		vertex_buffer = VK_NULL_HANDLE;
		vertex_buffer_allocation = VK_NULL_HANDLE;
	}
	if (material_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, material_buffer, material_buffer_allocation);
		material_buffer = VK_NULL_HANDLE;
		material_buffer_allocation = VK_NULL_HANDLE;
		material_buffer_data = nullptr;
	}
	if (material_descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, material_descriptor_pool, nullptr);
		material_descriptor_pool = VK_NULL_HANDLE;
		material_descriptor_set = VK_NULL_HANDLE;
	}
	if (material_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, material_set_layout, nullptr);
		material_set_layout = VK_NULL_HANDLE;
	}
}

} // namespace

bool initialize() {
	const std::vector<Vertex> vertices = createCylinder();
	if (!createBuffer(vertices) || !createMaterialResources() || !createPipeline()) {
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
	ImGui::Begin("Cylinder - variant 8");
	ImGui::Text("Right circular cylinder, 50 segments per base");

	int selected_projection = projection == Projection::perspective ? 0 : 1;
	const char* projection_modes[] = { "Perspective", "Orthographic" };
	if (ImGui::Combo("Projection", &selected_projection, projection_modes, 2)) {
		projection = selected_projection == 0 ? Projection::perspective : Projection::orthographic;
	}

	ImGui::SeparatorText("Translation");
	ImGui::SliderFloat3("X / Y / Z", translation, -2.0f, 2.0f);
	ImGui::SeparatorText("Rotation (degrees)");
	ImGui::SliderFloat3("X / Y / Z##rotation", rotation, -180.0f, 180.0f);
	ImGui::SeparatorText("Scale");
	ImGui::SliderFloat3("X / Y / Z##scale", scale, 0.2f, 2.0f);

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

	ImGui::SeparatorText("Color");
	ImGui::ColorEdit3("Tint", tint);
	if (ImGui::Button("Reset transformations")) {
		translation[0] = translation[1] = translation[2] = 0.0f;
		rotation[0] = 20.0f;
		rotation[1] = -25.0f;
		rotation[2] = 0.0f;
		scale[0] = scale[1] = scale[2] = 1.0f;
	}

	if (has_previous_update_time && animation_playing) {
		const float delta_time = std::clamp(static_cast<float>(time - previous_update_time), 0.0f, 0.1f);
		animation_phase += delta_time * animation_speed;
	}
	previous_update_time = time;
	has_previous_update_time = true;
	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;
	if (fd.command_buffer == VK_NULL_HANDLE || fd.framebuffer == VK_NULL_HANDLE ||
	    context.swapchain_extent.height == 0) {
		return;
	}

	const MaterialUniform material = {
		.tint = { tint[0], tint[1], tint[2], 1.0f },
	};
	std::memcpy(material_buffer_data, &material, sizeof(material));
	if (vmaFlushAllocation(context.allocator, material_buffer_allocation, 0,
	                       sizeof(material)) != VK_SUCCESS) {
		std::cerr << "Failed to flush material uniform buffer\n";
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

	const float animated_x = trajectory_radius * std::cos(animation_phase);
	const float animated_y = trajectory_height * std::sin(trajectory_frequency * animation_phase);
	const float animated_z = trajectory_depth * std::sin(2.0f * animation_phase);
	const Matrix model = multiply(
		translationMatrix(translation[0] + animated_x,
		                  translation[1] + animated_y,
		                  translation[2] + animated_z),
		multiply(rotationZ(rotation[2]),
		         multiply(rotationY(rotation[1] + animation_phase * animation_rotation_speed),
		                  multiply(rotationX(rotation[0]), scaleMatrix(scale[0], scale[1], scale[2])))));
	const Matrix view = translationMatrix(0.0f, 0.0f, -4.0f);
	const float aspect_ratio = static_cast<float>(context.swapchain_extent.width) /
	                           static_cast<float>(context.swapchain_extent.height);
	const Matrix projection_matrix = projection == Projection::perspective
		? perspectiveMatrix(aspect_ratio)
		: orthographicMatrix(aspect_ratio);
	const PushConstants push_constants{
		.mvp = multiply(projection_matrix, multiply(view, model)),
		.model = model,
	};
	vkCmdPushConstants(fd.command_buffer, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0,
	                   sizeof(push_constants), &push_constants);
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        0, 1, &material_descriptor_set, 0, nullptr);

	const VkDeviceSize vertex_offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, &vertex_offset);
	vkCmdDraw(fd.command_buffer, vertex_count, 1, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);
	if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to finish Vulkan command buffer\n";
	}
}

} // namespace application
