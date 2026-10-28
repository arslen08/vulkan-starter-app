#version 450

layout(location = 0) in vec3 normal;
layout(location = 1) in vec3 vertexColor;
layout(location = 2) in vec3 worldPosition;
layout(location = 0) out vec4 outColor;

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

vec3 safeNormalize(vec3 value) {
	float lengthSquared = dot(value, value);
	return lengthSquared > 1.0e-12 ? value * inversesqrt(lengthSquared) : vec3(0.0);
}

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 baseColor) {
	float diffuseFactor = max(dot(N, L), 0.0);
	float specularFactor = 0.0;
	if (diffuseFactor > 0.0) {
		vec3 H = safeNormalize(L + V);
		specularFactor = pow(max(dot(N, H), 0.0), max(object.specular.w, 1.0));
	}
	return radiance * (baseColor * diffuseFactor + object.specular.rgb * specularFactor);
}

void main() {
	vec3 N = safeNormalize(normal);
	vec3 V = safeNormalize(scene.cameraPosition.xyz - worldPosition);
	vec3 baseColor = vertexColor * object.albedo.rgb;
	vec3 lighting = scene.ambient.rgb * baseColor;

	if (scene.directionalColor.w > 0.0) {
		vec3 L = safeNormalize(-scene.directionalDirection.xyz);
		lighting += blinnPhong(
			N, V, L,
			scene.directionalColor.rgb * scene.directionalColor.w,
			baseColor
		);
	}

	outColor = vec4(lighting, object.albedo.a);
}