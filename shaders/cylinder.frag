#version 450

layout(location = 0) in vec3 normal;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 worldPosition;
layout(location = 3) in vec4 shadowPosition;

layout(location = 0) out vec4 outColor;

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
	vec4 parameters;
} object;

layout(set = 0, binding = 2) uniform sampler2D albedoTexture;
layout(set = 0, binding = 3) uniform sampler2DShadow shadowMap;

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

float sampleShadow(vec4 shadowPos) {
	vec3 proj = shadowPos.xyz / shadowPos.w;
	proj.xy = proj.xy * 0.5 + 0.5;
	if (proj.z < 0.0 || proj.z > 1.0 || proj.x < 0.0 || proj.x > 1.0 || proj.y < 0.0 || proj.y > 1.0) {
		return 1.0;
	}
	return texture(shadowMap, proj);
}

void main() {
	vec3 N = safeNormalize(normal);
	vec3 V = safeNormalize(scene.cameraPosition.xyz - worldPosition);
	vec3 baseColor;
	if (object.parameters.x > 0.5) {
		vec2 tileUv = uv * 0.75;
		vec2 tilePosition = fract(tileUv);
		float edgeDistance = min(
			min(tilePosition.x, 1.0 - tilePosition.x),
			min(tilePosition.y, 1.0 - tilePosition.y)
		);
		float grout = 1.0 - smoothstep(0.0, 0.035, edgeDistance);
		float tileVariation = mod(floor(tileUv.x) + floor(tileUv.y), 2.0) * 0.015;
		float surfaceTone = 0.985 - tileVariation - grout * 0.055;
		baseColor = object.albedo.rgb * surfaceTone;
	} else {
		baseColor = texture(albedoTexture, uv).rgb * object.albedo.rgb;
	}
	vec3 lighting = scene.ambient.rgb * baseColor;

	float shadow = 1.0;
	if (scene.directionalColor.w > 0.0) {
		shadow = sampleShadow(shadowPosition);
		vec3 L = safeNormalize(-scene.directionalDirection.xyz);
		lighting += blinnPhong(
			N, V, L,
			scene.directionalColor.rgb * scene.directionalColor.w * shadow,
			baseColor
		);
	}

	outColor = vec4(lighting, object.albedo.a);
}