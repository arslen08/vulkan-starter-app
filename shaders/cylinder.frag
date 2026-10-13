#version 450

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 vertexColor;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform Material {
	vec4 tint;
} material;

void main() {
	vec3 lightDirection = normalize(vec3(-0.45, 0.8, 0.55));
	float diffuse = max(dot(normalize(normal), lightDirection), 0.0);
	float lighting = 0.25 + 0.75 * diffuse;
	outColor = vec4(vertexColor * material.tint.rgb * lighting, 1.0);
}
