#version 450

layout(location = 0) in vec3 inPosition;

layout(set = 0, binding = 0) uniform Scene {
	mat4 viewProjection;   
	mat4 shadowProjection;
	vec4 cameraPosition;
	vec4 ambient;
	vec4 directionalDirection;
	vec4 directionalColor;
} scene;

layout(set = 0, binding = 1) uniform Model {
	mat4 model;
	mat4 normalMatrix;
	vec4 albedo;
	vec4 specular;
} object;

void main() {
	vec4 world = object.model * vec4(inPosition, 1.0);
	gl_Position = scene.viewProjection * world;
}