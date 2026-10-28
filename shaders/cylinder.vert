#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(set = 0, binding = 0) uniform Scene {
	mat4 viewProjection;
	vec4 cameraPosition;
	vec4 ambient;
	vec4 directionalDirection;
	vec4 directionalColor;
} scene;

layout(set = 1, binding = 0) uniform Model {
	mat4 model;
	mat4 normalMatrix;
	vec4 albedo;
	vec4 specular;
} object;

layout(location = 0) out vec3 normal;
layout(location = 1) out vec3 vertexColor;
layout(location = 2) out vec3 worldPosition;

void main() {
	vec4 world = object.model * vec4(inPosition, 1.0);
	gl_Position = scene.viewProjection * world;
	normal = normalize(mat3(object.normalMatrix) * inNormal);
	vertexColor = inColor;
	worldPosition = world.xyz;
}