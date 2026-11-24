#include "application.hpp"

#include <imgui.h>

#include <vulkan/vulkan.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace application {

namespace {

constexpr uint32_t cylinder_segments = 50;
constexpr float pi = 3.14159265358979323846f;

struct Vec2 {
	float u;
	float v;
};

struct Vec3 {
	float x;
	float y;
	float z;
};

struct Vertex {
	Vec3 position;
	Vec3 normal;
	Vec2 uv;
};

struct Matrix {
	std::array<float, 16> values{};
};

struct SceneUniform {
	Matrix view_projection;
	Matrix shadow_projection;
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
	float parameters[4];
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

struct Texture {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VkImageView view = VK_NULL_HANDLE;
	VkSampler sampler = VK_NULL_HANDLE;
	uint32_t width = 0;
	uint32_t height = 0;
};

struct Model {
	Mesh mesh;
};

constexpr size_t model_count = 2;
constexpr size_t render_object_count = model_count + 1;

VkPipeline graphics_pipeline = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
VkDescriptorSet shadow_descriptor_set = VK_NULL_HANDLE;

HostBuffer scene_buffer;
HostBuffer shadow_scene_buffer;
HostBuffer model_uniforms_buffer;
std::array<Model, model_count> models;
Mesh floor_mesh;
Texture albedo_texture;

constexpr uint32_t shadow_map_size = 2048;
VkFormat shadow_depth_format = VK_FORMAT_UNDEFINED;
VkImage shadow_depth_image = VK_NULL_HANDLE;
VmaAllocation shadow_depth_allocation = VK_NULL_HANDLE;
VkImageView shadow_depth_view = VK_NULL_HANDLE;
VkSampler shadow_sampler = VK_NULL_HANDLE;
VkPipeline shadow_pipeline = VK_NULL_HANDLE;
PFN_vkCmdBeginRenderingKHR cmd_begin_rendering = nullptr;
PFN_vkCmdEndRenderingKHR cmd_end_rendering = nullptr;

VkDeviceSize model_uniform_alignment = 256;
VkDeviceSize model_uniform_size = 0;

Projection projection = Projection::perspective;

float translation0[3] = { -1.5f, 0.0f, 0.0f };
float rotation0[3] = { 20.0f, -25.0f, 0.0f };
float scale0[3] = { 1.0f, 1.0f, 1.0f };

float translation1[3] = { 1.5f, 0.0f, 0.0f };
float rotation1[3] = { -15.0f, 40.0f, 10.0f };
float scale1[3] = { 0.8f, 0.8f, 0.8f };

std::array<Material, model_count> materials = {{
	{ { 1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f }, 64.0f },
	{ { 1.0f, 1.0f, 1.0f }, { 0.6f, 0.6f, 0.6f }, 32.0f },
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
float shadow_projection_half_size = 6.0f;
float shadow_depth_bias_constant = 1.25f;
float shadow_depth_bias_slope = 1.75f;

bool animation_playing = true;
float animation_phase = 0.0f;
float animation_speed = 1.0f;
float trajectory_radius = 0.5f;
float trajectory_depth = 0.3f;
float trajectory_height = 0.25f;
float trajectory_frequency = 2.0f;
float animation_rotation_speed = 45.0f;
double previous_update_time = 0.0;
bool has_previous_update_time = false;

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

Matrix lookAtMatrix(const float eye[3], const float center[3], const float up[3]) {
	float f[3] = { center[0] - eye[0], center[1] - eye[1], center[2] - eye[2] };
	float f_len = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
	if (f_len > 1e-6f) {
		f[0] /= f_len; f[1] /= f_len; f[2] /= f_len;
	} else {
		f[0] = 0.0f; f[1] = 0.0f; f[2] = -1.0f;
	}

	float s[3] = {
		f[1] * up[2] - f[2] * up[1],
		f[2] * up[0] - f[0] * up[2],
		f[0] * up[1] - f[1] * up[0]
	};
	float s_len = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
	if (s_len > 1e-6f) {
		s[0] /= s_len; s[1] /= s_len; s[2] /= s_len;
	} else {
		s[0] = 1.0f; s[1] = 0.0f; s[2] = 0.0f;
	}

	float u[3] = {
		s[1] * f[2] - s[2] * f[1],
		s[2] * f[0] - s[0] * f[2],
		s[0] * f[1] - s[1] * f[0]
	};

	Matrix result = identityMatrix();
	result.values[0] = s[0];
	result.values[1] = u[0];
	result.values[2] = -f[0];
	result.values[4] = s[1];
	result.values[5] = u[1];
	result.values[6] = -f[1];
	result.values[8] = s[2];
	result.values[9] = u[2];
	result.values[10] = -f[2];
	result.values[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
	result.values[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
	result.values[14] = (f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2]);
	return result;
}

Matrix shadowOrthographicMatrix(float half_size, float near_plane, float far_plane) {
	Matrix result = identityMatrix();
	result.values[0] = 1.0f / half_size;
	result.values[5] = -1.0f / half_size; 
	result.values[10] = 1.0f / (near_plane - far_plane);
	result.values[14] = near_plane / (near_plane - far_plane);
	return result;
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

void addVertex(std::vector<Vertex>& vertices, Vec3 position, Vec3 normal, Vec2 uv) {
	vertices.push_back({ position, normal, uv });
}

void addTriangle(std::vector<Vertex>& vertices,
                 Vec3 a, Vec3 b, Vec3 c,
                 Vec3 normal_a, Vec3 normal_b, Vec3 normal_c,
                 Vec2 uv_a, Vec2 uv_b, Vec2 uv_c) {
	addVertex(vertices, a, normal_a, uv_a);
	addVertex(vertices, b, normal_b, uv_b);
	addVertex(vertices, c, normal_c, uv_c);
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

		const float u0 = static_cast<float>(i) / cylinder_segments;
		const float u1 = static_cast<float>(i + 1) / cylinder_segments;

		const Vec3 bottom0{ x0, -half_height, z0 };
		const Vec3 top0{ x0, half_height, z0 };
		const Vec3 bottom1{ x1, -half_height, z1 };
		const Vec3 top1{ x1, half_height, z1 };
		const Vec3 normal0{ std::cos(angle0), 0.0f, std::sin(angle0) };
		const Vec3 normal1{ std::cos(angle1), 0.0f, std::sin(angle1) };

		addTriangle(vertices, bottom0, top0, top1, normal0, normal0, normal1,
		            { u0, 0.0f }, { u0, 1.0f }, { u1, 1.0f });
		addTriangle(vertices, bottom0, top1, bottom1, normal0, normal1, normal1,
		            { u0, 0.0f }, { u1, 1.0f }, { u1, 0.0f });

		const Vec3 top_center{ 0.0f, half_height, 0.0f };
		const Vec3 bottom_center{ 0.0f, -half_height, 0.0f };
		const Vec3 top_normal{ 0.0f, 1.0f, 0.0f };
		const Vec3 bottom_normal{ 0.0f, -1.0f, 0.0f };

		const Vec2 uv_tc{ 0.5f, 0.5f };
		const Vec2 uv_t0{ 0.5f + 0.5f * x0, 0.5f + 0.5f * z0 };
		const Vec2 uv_t1{ 0.5f + 0.5f * x1, 0.5f + 0.5f * z1 };
		addTriangle(vertices, top_center, top1, top0, top_normal, top_normal, top_normal,
		            uv_tc, uv_t1, uv_t0);

		const Vec2 uv_bc{ 0.5f, 0.5f };
		const Vec2 uv_b0{ 0.5f + 0.5f * x0, 0.5f + 0.5f * z0 };
		const Vec2 uv_b1{ 0.5f + 0.5f * x1, 0.5f + 0.5f * z1 };
		addTriangle(vertices, bottom_center, bottom0, bottom1,
		            bottom_normal, bottom_normal, bottom_normal,
		            uv_bc, uv_b0, uv_b1);
	}

	return vertices;
}

std::vector<Vertex> createFloor() {
	constexpr float half_size = 6.0f;
	constexpr float height = -1.5f;
	const Vec3 normal{ 0.0f, 1.0f, 0.0f };
	const Vec3 a{ -half_size, height, -half_size };
	const Vec3 b{ half_size, height, -half_size };
	const Vec3 c{ half_size, height, half_size };
	const Vec3 d{ -half_size, height, half_size };

	std::vector<Vertex> vertices;
	vertices.reserve(6);
	addTriangle(vertices, a, d, b, normal, normal, normal,
	            { 0.0f, 0.0f }, { 0.0f, half_size }, { half_size, 0.0f });
	addTriangle(vertices, b, d, c, normal, normal, normal,
	            { half_size, 0.0f }, { 0.0f, half_size }, { half_size, half_size });
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

bool createDeviceLocalBuffer(VkBuffer& buffer, VmaAllocation& allocation, VkDeviceSize size,
                             VkBufferUsageFlags usage, const void* data, const char* name) {
	auto& context = graphics::internal::context;

	HostBuffer staging{};
	if (!createHostBuffer(staging, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "staging")) {
		return false;
	}
	std::memcpy(staging.data, data, static_cast<size_t>(size));
	vmaFlushAllocation(context.allocator, staging.allocation, 0, size);

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo alloc_info = {
		.flags = 0,
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &alloc_info, &buffer, &allocation,
	                    nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create device-local buffer: " << name << '\n';
		vmaDestroyBuffer(context.allocator, staging.buffer, staging.allocation);
		return false;
	}

	VkCommandBufferAllocateInfo cmd_alloc = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = context.command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	if (vkAllocateCommandBuffers(context.device, &cmd_alloc, &cmd) != VK_SUCCESS) {
		std::cerr << "Failed to allocate command buffer for transfer\n";
		vmaDestroyBuffer(context.allocator, buffer, allocation);
		vmaDestroyBuffer(context.allocator, staging.buffer, staging.allocation);
		return false;
	}

	VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin_info);

	VkBufferCopy copy_region = { .srcOffset = 0, .dstOffset = 0, .size = size };
	vkCmdCopyBuffer(cmd, staging.buffer, buffer, 1, &copy_region);

	vkEndCommandBuffer(cmd);

	VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
	};
	vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE);
	vkQueueWaitIdle(context.graphics_queue);

	vkFreeCommandBuffers(context.device, context.command_pool, 1, &cmd);
	vmaDestroyBuffer(context.allocator, staging.buffer, staging.allocation);
	return true;
}

bool createMesh(Mesh& mesh, const std::vector<Vertex>& vertices, const char* name) {
	const VkDeviceSize size = vertices.size() * sizeof(Vertex);
	if (!createDeviceLocalBuffer(mesh.buffer, mesh.allocation, size,
	                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertices.data(), name)) {
		return false;
	}
	mesh.vertex_count = static_cast<uint32_t>(vertices.size());
	return true;
}

bool createTextureFromFile(Texture& texture, const char* path) {
	auto& context = graphics::internal::context;

	int width, height, channels;
	stbi_uc* pixels = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
	if (!pixels) {
		std::cerr << "Failed to load texture: " << path << " (" << stbi_failure_reason() << ")\n";
		width = 1;
		height = 1;
		pixels = static_cast<stbi_uc*>(std::malloc(4));
		pixels[0] = pixels[1] = pixels[2] = pixels[3] = 255;
	}

	const VkDeviceSize image_size = static_cast<VkDeviceSize>(width) * height * 4;

	HostBuffer staging{};
	if (!createHostBuffer(staging, image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, "texture staging")) {
		stbi_image_free(pixels);
		return false;
	}
	std::memcpy(staging.data, pixels, static_cast<size_t>(image_size));
	vmaFlushAllocation(context.allocator, staging.allocation, 0, image_size);
	stbi_image_free(pixels);

	const VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = VK_FORMAT_R8G8B8A8_SRGB,
		.extent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	const VmaAllocationCreateInfo alloc_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};
	if (vmaCreateImage(context.allocator, &image_info, &alloc_info, &texture.image,
	                   &texture.allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create texture image\n";
		vmaDestroyBuffer(context.allocator, staging.buffer, staging.allocation);
		return false;
	}

	VkCommandBufferAllocateInfo cmd_alloc = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = context.command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	vkAllocateCommandBuffers(context.device, &cmd_alloc, &cmd);

	VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(cmd, &begin_info);

	VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = 0,
		.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
		.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = texture.image,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
	};
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     0, 0, nullptr, 0, nullptr, 1, &barrier);

	VkBufferImageCopy region = {
		.bufferOffset = 0,
		.bufferRowLength = 0,
		.bufferImageHeight = 0,
		.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
		.imageOffset = { 0, 0, 0 },
		.imageExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 },
	};
	vkCmdCopyBufferToImage(cmd, staging.buffer, texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
	                       1, &region);

	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
	                     0, 0, nullptr, 0, nullptr, 1, &barrier);

	vkEndCommandBuffer(cmd);

	VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.commandBufferCount = 1,
		.pCommandBuffers = &cmd,
	};
	vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE);
	vkQueueWaitIdle(context.graphics_queue);
	vkFreeCommandBuffers(context.device, context.command_pool, 1, &cmd);
	vmaDestroyBuffer(context.allocator, staging.buffer, staging.allocation);

	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = texture.image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = VK_FORMAT_R8G8B8A8_SRGB,
		.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
	};
	if (vkCreateImageView(context.device, &view_info, nullptr, &texture.view) != VK_SUCCESS) {
		std::cerr << "Failed to create texture image view\n";
		return false;
	}

	const VkSamplerCreateInfo sampler_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.anisotropyEnable = VK_FALSE,
		.maxAnisotropy = 1.0f,
		.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
		.unnormalizedCoordinates = VK_FALSE,
	};
	if (vkCreateSampler(context.device, &sampler_info, nullptr, &texture.sampler) != VK_SUCCESS) {
		std::cerr << "Failed to create texture sampler\n";
		return false;
	}

	texture.width = static_cast<uint32_t>(width);
	texture.height = static_cast<uint32_t>(height);
	return true;
}

bool createShadowResources() {
	auto& context = graphics::internal::context;

	const VkFormat candidates[] = {
		VK_FORMAT_D32_SFLOAT,
		VK_FORMAT_D32_SFLOAT_S8_UINT,
		VK_FORMAT_D24_UNORM_S8_UINT,
	};
	shadow_depth_format = VK_FORMAT_UNDEFINED;
	for (const auto& f : candidates) {
		VkFormatProperties properties{};
		vkGetPhysicalDeviceFormatProperties(context.physical_device, f, &properties);
		if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
			shadow_depth_format = f;
			break;
		}
	}
	if (shadow_depth_format == VK_FORMAT_UNDEFINED) {
		std::cerr << "Failed to find a suitable depth format for the shadow map\n";
		return false;
	}

	const VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = shadow_depth_format,
		.extent = { shadow_map_size, shadow_map_size, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	const VmaAllocationCreateInfo alloc_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	};
	if (vmaCreateImage(context.allocator, &image_info, &alloc_info,
	                   &shadow_depth_image, &shadow_depth_allocation, nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create shadow depth image\n";
		return false;
	}

	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = shadow_depth_image,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = shadow_depth_format,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};
	if (vkCreateImageView(context.device, &view_info, nullptr, &shadow_depth_view) != VK_SUCCESS) {
		std::cerr << "Failed to create shadow depth image view\n";
		return false;
	}

	const VkSamplerCreateInfo sampler_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR,
		.minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
		.anisotropyEnable = VK_FALSE,
		.maxAnisotropy = 1.0f,
		.compareEnable = VK_TRUE,
		.compareOp = VK_COMPARE_OP_LESS,
		.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE, 
		.unnormalizedCoordinates = VK_FALSE,
	};
	if (vkCreateSampler(context.device, &sampler_info, nullptr, &shadow_sampler) != VK_SUCCESS) {
		std::cerr << "Failed to create shadow sampler\n";
		return false;
	}

	return true;
}

bool createDescriptors() {
	auto& context = graphics::internal::context;

	VkPhysicalDeviceProperties props{};
	vkGetPhysicalDeviceProperties(context.physical_device, &props);
	model_uniform_alignment = props.limits.minUniformBufferOffsetAlignment;
	model_uniform_size = (sizeof(ModelUniform) + model_uniform_alignment - 1) /
	                     model_uniform_alignment * model_uniform_alignment;

	const VkDescriptorSetLayoutBinding bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 3,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};
	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 4,
		.pBindings = bindings,
	};
	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr,
	                                &descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_sizes[] = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4 },
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 2,
		.poolSizeCount = 3,
		.pPoolSizes = pool_sizes,
	};
	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) !=
	    VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetLayout set_layouts[] = {
		descriptor_set_layout,
		descriptor_set_layout,
	};
	const VkDescriptorSetAllocateInfo set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 2,
		.pSetLayouts = set_layouts,
	};
	VkDescriptorSet descriptor_sets[2] = {};
	if (vkAllocateDescriptorSets(context.device, &set_info, descriptor_sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set\n";
		return false;
	}
	descriptor_set = descriptor_sets[0];
	shadow_descriptor_set = descriptor_sets[1];

	if (!createHostBuffer(scene_buffer, sizeof(SceneUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                      "scene uniform")) {
		return false;
	}
	if (!createHostBuffer(shadow_scene_buffer, sizeof(SceneUniform),
	                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, "shadow scene uniform")) {
		return false;
	}

	const VkDeviceSize total_model_size = model_uniform_size * render_object_count;
	if (!createHostBuffer(model_uniforms_buffer, total_model_size,
	                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, "model uniforms dynamic")) {
		return false;
	}

	if (!createTextureFromFile(albedo_texture, "textures/texture.png")) {
		if (!createTextureFromFile(albedo_texture, "/home/workdir/artifacts/textures/texture.png")) {
			std::cerr << "Warning: using fallback texture\n";
		}
	}

	if (!createShadowResources()) {
		return false;
	}

	const VkDescriptorBufferInfo scene_info = {
		.buffer = scene_buffer.buffer,
		.offset = 0,
		.range = sizeof(SceneUniform),
	};
	const VkDescriptorBufferInfo shadow_scene_info = {
		.buffer = shadow_scene_buffer.buffer,
		.offset = 0,
		.range = sizeof(SceneUniform),
	};
	const VkDescriptorBufferInfo model_info = {
		.buffer = model_uniforms_buffer.buffer,
		.offset = 0,
		.range = sizeof(ModelUniform),
	};
	const VkDescriptorImageInfo image_info = {
		.sampler = albedo_texture.sampler,
		.imageView = albedo_texture.view,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};
	const VkDescriptorImageInfo shadow_image_info = {
		.sampler = shadow_sampler,
		.imageView = shadow_depth_view,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};

	const VkWriteDescriptorSet writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &scene_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = shadow_descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &shadow_scene_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = descriptor_set,
			.dstBinding = 1,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.pBufferInfo = &model_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = shadow_descriptor_set,
			.dstBinding = 1,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.pBufferInfo = &model_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = descriptor_set,
			.dstBinding = 2,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &image_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = shadow_descriptor_set,
			.dstBinding = 2,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &image_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = descriptor_set,
			.dstBinding = 3,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &shadow_image_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = shadow_descriptor_set,
			.dstBinding = 3,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &shadow_image_info,
		},
	};
	vkUpdateDescriptorSets(context.device, 8, writes, 0, nullptr);
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
		{ .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,
		  .offset = offsetof(Vertex, uv) },
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
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};
	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) !=
	    VK_SUCCESS) {
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

	VkShaderModule shadow_vert = VK_NULL_HANDLE;
	if (!loadShader("shaders/shadow.vert.spv", shadow_vert)) {
		std::cerr << "Failed to load shadow vertex shader\n";
		return false;
	}

	const VkPipelineShaderStageCreateInfo shadow_stage = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.stage = VK_SHADER_STAGE_VERTEX_BIT,
		.module = shadow_vert,
		.pName = "main",
	};

	const VkVertexInputAttributeDescription shadow_attributes[] = {
		{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT,
		  .offset = offsetof(Vertex, position) },
	};
	const VkPipelineVertexInputStateCreateInfo shadow_vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 1,
		.pVertexAttributeDescriptions = shadow_attributes,
	};

	const VkPipelineRasterizationStateCreateInfo shadow_rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_TRUE, 
		.lineWidth = 1.0f,
	};

	const VkPipelineColorBlendStateCreateInfo shadow_color_blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 0, 
		.pAttachments = nullptr,
	};

	const VkDynamicState shadow_dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
		VK_DYNAMIC_STATE_DEPTH_BIAS,
	};
	const VkPipelineDynamicStateCreateInfo shadow_dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 3,
		.pDynamicStates = shadow_dynamic_states,
	};

	const VkPipelineRenderingCreateInfo shadow_rendering_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
		.colorAttachmentCount = 0,
		.pColorAttachmentFormats = nullptr,
		.depthAttachmentFormat = shadow_depth_format,
	};

	const VkGraphicsPipelineCreateInfo shadow_pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.pNext = &shadow_rendering_info,
		.stageCount = 1,
		.pStages = &shadow_stage,
		.pVertexInputState = &shadow_vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &shadow_rasterization,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &shadow_color_blending,
		.pDynamicState = &shadow_dynamic_state,
		.layout = pipeline_layout,
		.renderPass = VK_NULL_HANDLE, 
		.subpass = 0,
	};

	const VkResult shadow_result = vkCreateGraphicsPipelines(
		context.device, VK_NULL_HANDLE, 1, &shadow_pipeline_info, nullptr, &shadow_pipeline);
	vkDestroyShaderModule(context.device, shadow_vert, nullptr);
	if (shadow_result != VK_SUCCESS) {
		std::cerr << "Failed to create shadow graphics pipeline\n";
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

void destroyTexture(Texture& texture) {
	auto& context = graphics::internal::context;
	if (texture.sampler != VK_NULL_HANDLE) {
		vkDestroySampler(context.device, texture.sampler, nullptr);
	}
	if (texture.view != VK_NULL_HANDLE) {
		vkDestroyImageView(context.device, texture.view, nullptr);
	}
	if (texture.image != VK_NULL_HANDLE) {
		vmaDestroyImage(context.allocator, texture.image, texture.allocation);
	}
	texture = Texture{};
}

void destroyResources() {
	auto& context = graphics::internal::context;
	if (shadow_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, shadow_pipeline, nullptr);
		shadow_pipeline = VK_NULL_HANDLE;
	}
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
	}
	if (floor_mesh.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, floor_mesh.buffer, floor_mesh.allocation);
		floor_mesh = Mesh{};
	}
	destroyHostBuffer(scene_buffer);
	destroyHostBuffer(shadow_scene_buffer);
	destroyHostBuffer(model_uniforms_buffer);
	destroyTexture(albedo_texture);

	if (shadow_sampler != VK_NULL_HANDLE) {
		vkDestroySampler(context.device, shadow_sampler, nullptr);
		shadow_sampler = VK_NULL_HANDLE;
	}
	if (shadow_depth_view != VK_NULL_HANDLE) {
		vkDestroyImageView(context.device, shadow_depth_view, nullptr);
		shadow_depth_view = VK_NULL_HANDLE;
	}
	if (shadow_depth_image != VK_NULL_HANDLE) {
		vmaDestroyImage(context.allocator, shadow_depth_image, shadow_depth_allocation);
		shadow_depth_image = VK_NULL_HANDLE;
		shadow_depth_allocation = VK_NULL_HANDLE;
	}

	if (descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
		descriptor_pool = VK_NULL_HANDLE;
		descriptor_set = VK_NULL_HANDLE;
		shadow_descriptor_set = VK_NULL_HANDLE;
	}
	if (descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);
		descriptor_set_layout = VK_NULL_HANDLE;
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

	ImGui::SeparatorText("Shadow map");
	ImGui::SliderFloat("Projection half-size", &shadow_projection_half_size, 2.0f, 12.0f);
	ImGui::Text("Coverage: %.1f x %.1f units",
	            shadow_projection_half_size * 2.0f, shadow_projection_half_size * 2.0f);
	ImGui::SliderFloat("Depth bias constant", &shadow_depth_bias_constant, 0.0f, 5.0f, "%.2f");
	ImGui::SliderFloat("Depth bias slope", &shadow_depth_bias_slope, 0.0f, 5.0f, "%.2f");

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
	ImGui::Begin("Cylinders");

	ImGui::SeparatorText("Cylinder 1");
	ImGui::SliderFloat3("Translation##0", translation0, -3.0f, 3.0f);
	ImGui::SliderFloat3("Rotation##0", rotation0, -180.0f, 180.0f);
	ImGui::SliderFloat3("Scale##0", scale0, 0.2f, 2.0f);
	drawMaterialEditor("Material 1", materials[0]);

	ImGui::SeparatorText("Cylinder 2");
	ImGui::SliderFloat3("Translation##1", translation1, -3.0f, 3.0f);
	ImGui::SliderFloat3("Rotation##1", rotation1, -180.0f, 180.0f);
	ImGui::SliderFloat3("Scale##1", scale1, 0.2f, 2.0f);
	drawMaterialEditor("Material 2", materials[1]);

	ImGui::SeparatorText("Animation");
	if (ImGui::Button(animation_playing ? "Pause" : "Play")) {
		animation_playing = !animation_playing;
	}
	ImGui::SameLine();
	if (ImGui::Button("Restart")) {
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
	for (int i = 0; i < 4; ++i) uniform.parameters[i] = 0.0f;
}

} 

bool initialize() {
	auto& context = graphics::internal::context;
	cmd_begin_rendering = reinterpret_cast<PFN_vkCmdBeginRenderingKHR>(
		vkGetDeviceProcAddr(context.device, "vkCmdBeginRenderingKHR"));
	cmd_end_rendering = reinterpret_cast<PFN_vkCmdEndRenderingKHR>(
		vkGetDeviceProcAddr(context.device, "vkCmdEndRenderingKHR"));
	if (cmd_begin_rendering == nullptr || cmd_end_rendering == nullptr) {
		std::cerr << "Failed to load Vulkan dynamic rendering commands\n";
		return false;
	}

	std::vector<Vertex> cylinder_vertices = createCylinder();
	std::vector<Vertex> floor_vertices = createFloor();
	if (!createMesh(models[0].mesh, cylinder_vertices, "cylinder 0") ||
	    !createMesh(models[1].mesh, cylinder_vertices, "cylinder 1") ||
	    !createMesh(floor_mesh, floor_vertices, "floor") ||
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
	float light_dir[3];
	normalize3(directional_direction, light_dir, default_direction);

	const float light_distance = 15.0f;
	const float light_eye[3] = {
		-light_dir[0] * light_distance,
		-light_dir[1] * light_distance,
		-light_dir[2] * light_distance
	};
	const float light_center[3] = { 0.0f, 0.0f, 0.0f };
	const float light_up[3] = { 0.0f, 1.0f, 0.0f };
	const Matrix light_view = lookAtMatrix(light_eye, light_center, light_up);
	const Matrix light_proj =
		shadowOrthographicMatrix(shadow_projection_half_size, 0.5f, 40.0f);
	const Matrix shadow_vp = multiply(light_proj, light_view);

	SceneUniform scene_data{};
	scene_data.view_projection = multiply(projection_matrix, viewMatrix());
	scene_data.shadow_projection = shadow_vp;
	for (int i = 0; i < 3; ++i) {
		scene_data.camera_position[i] = camera_position[i];
		scene_data.ambient[i] = ambient_color[i] * ambient_intensity;
		scene_data.directional_color[i] = directional_color[i];
		scene_data.directional_direction[i] = light_dir[i];
	}
	scene_data.directional_color[3] = directional_enabled ? directional_intensity : 0.0f;

	const float animated_x = trajectory_radius * std::cos(animation_phase);
	const float animated_y = trajectory_height * std::sin(trajectory_frequency * animation_phase);
	const float animated_z = trajectory_depth * std::sin(2.0f * animation_phase);

	const Matrix rot0 = multiply(
		rotationZ(rotation0[2]),
		multiply(rotationY(rotation0[1] + animation_phase * animation_rotation_speed),
		         rotationX(rotation0[0])));
	const Matrix model0 = multiply(
		translationMatrix(translation0[0] + animated_x, translation0[1] + animated_y, translation0[2] + animated_z),
		multiply(rot0, scaleMatrix(scale0[0], scale0[1], scale0[2])));
	const Matrix normal0 = multiply(rot0, scaleMatrix(1.0f / scale0[0], 1.0f / scale0[1], 1.0f / scale0[2]));

	const Matrix rot1 = multiply(
		rotationZ(rotation1[2]),
		multiply(rotationY(rotation1[1] - animation_phase * animation_rotation_speed * 0.7f),
		         rotationX(rotation1[0])));
	const Matrix model1 = multiply(
		translationMatrix(translation1[0] - animated_x * 0.6f, translation1[1] + animated_y * 0.8f, translation1[2] - animated_z),
		multiply(rot1, scaleMatrix(scale1[0], scale1[1], scale1[2])));
	const Matrix normal1 = multiply(rot1, scaleMatrix(1.0f / scale1[0], 1.0f / scale1[1], 1.0f / scale1[2]));

	ModelUniform uniforms[render_object_count];
	fillModelUniform(uniforms[0], model0, normal0, materials[0]);
	fillModelUniform(uniforms[1], model1, normal1, materials[1]);
	const Material floor_material = {
		{ 0.68f, 0.70f, 0.72f },
		{ 0.05f, 0.05f, 0.05f },
		8.0f,
	};
	fillModelUniform(uniforms[model_count], identityMatrix(), identityMatrix(), floor_material);
	uniforms[model_count].parameters[0] = 1.0f;

	for (size_t i = 0; i < render_object_count; ++i) {
		std::memcpy(static_cast<char*>(model_uniforms_buffer.data) + i * model_uniform_size,
		            &uniforms[i], sizeof(ModelUniform));
	}
	vmaFlushAllocation(context.allocator, model_uniforms_buffer.allocation, 0,
	                   model_uniform_size * render_object_count);

	const VkCommandBufferBeginInfo command_buffer_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	if (vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin) != VK_SUCCESS) {
		std::cerr << "Failed to begin Vulkan command buffer\n";
		return;
	}

	{
		VkImageMemoryBarrier barrier = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = 0,
			.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = shadow_depth_image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		vkCmdPipelineBarrier(fd.command_buffer,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			0, 0, nullptr, 0, nullptr, 1, &barrier);
	}

	SceneUniform shadow_scene = scene_data;
	shadow_scene.view_projection = shadow_vp;
	if (!uploadToBuffer(shadow_scene_buffer, &shadow_scene, sizeof(shadow_scene))) {
		return;
	}

	VkRenderingAttachmentInfo depth_attachment = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
		.imageView = shadow_depth_view,
		.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.clearValue = { .depthStencil = { 1.0f, 0 } },
	};
	VkRenderingInfo rendering_info = {
		.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
		.renderArea = { { 0, 0 }, { shadow_map_size, shadow_map_size } },
		.layerCount = 1,
		.colorAttachmentCount = 0,
		.pColorAttachments = nullptr,
		.pDepthAttachment = &depth_attachment,
	};

	cmd_begin_rendering(fd.command_buffer, &rendering_info);

	const VkViewport shadow_viewport = {
		.x = 0.0f, .y = 0.0f,
		.width = static_cast<float>(shadow_map_size),
		.height = static_cast<float>(shadow_map_size),
		.minDepth = 0.0f, .maxDepth = 1.0f,
	};
	const VkRect2D shadow_scissor = { { 0, 0 }, { shadow_map_size, shadow_map_size } };
	vkCmdSetViewport(fd.command_buffer, 0, 1, &shadow_viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &shadow_scissor);
	vkCmdSetDepthBias(fd.command_buffer, shadow_depth_bias_constant, 0.0f,
	                  shadow_depth_bias_slope);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);

	for (size_t i = 0; i < model_count; ++i) {
		uint32_t dynamic_offset = static_cast<uint32_t>(i * model_uniform_size);
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        0, 1, &shadow_descriptor_set, 1, &dynamic_offset);

		const VkDeviceSize vertex_offset = 0;
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &models[i].mesh.buffer, &vertex_offset);
		vkCmdDraw(fd.command_buffer, models[i].mesh.vertex_count, 1, 0, 0);
	}

	cmd_end_rendering(fd.command_buffer);

	{
		VkImageMemoryBarrier barrier = {
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = shadow_depth_image,
			.subresourceRange = {
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1,
			},
		};
		vkCmdPipelineBarrier(fd.command_buffer,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &barrier);
	}

	if (!uploadToBuffer(scene_buffer, &scene_data, sizeof(scene_data))) {
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

	for (size_t i = 0; i < model_count; ++i) {
		uint32_t dynamic_offset = static_cast<uint32_t>(i * model_uniform_size);
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                        0, 1, &descriptor_set, 1, &dynamic_offset);

		const VkDeviceSize vertex_offset = 0;
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &models[i].mesh.buffer, &vertex_offset);
		vkCmdDraw(fd.command_buffer, models[i].mesh.vertex_count, 1, 0, 0);
	}

	const uint32_t floor_dynamic_offset = static_cast<uint32_t>(model_count * model_uniform_size);
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
	                        0, 1, &descriptor_set, 1, &floor_dynamic_offset);
	const VkDeviceSize floor_vertex_offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &floor_mesh.buffer, &floor_vertex_offset);
	vkCmdDraw(fd.command_buffer, floor_mesh.vertex_count, 1, 0, 0);

	vkCmdEndRenderPass(fd.command_buffer);
	if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to finish Vulkan command buffer\n";
	}
}