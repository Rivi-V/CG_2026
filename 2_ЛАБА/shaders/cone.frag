#version 450

layout(location = 0) in vec3 fragWorldPosition;
layout(location = 1) in vec3 fragWorldNormal;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform SceneData {
    mat4 model;
    mat4 view;
    mat4 projection;

    vec4 cameraPosition;
    vec4 lightDirection;
    vec4 lightColorIntensity;
    vec4 ambientColorIntensity;
    vec4 materialAlbedo;
    vec4 materialSpecularShininess;
} scene;

vec3 safeNormalize(vec3 value, vec3 fallbackValue) {
    float valueLength = length(value);
    if (valueLength < 0.0001) {
        return fallbackValue;
    }
    return value / valueLength;
}

void main() {
    vec3 N = safeNormalize(fragWorldNormal, vec3(0.0, 1.0, 0.0));
    vec3 L = safeNormalize(-scene.lightDirection.xyz, vec3(0.0, 1.0, 0.0));
    vec3 V = safeNormalize(
        scene.cameraPosition.xyz - fragWorldPosition,
        vec3(0.0, 0.0, 1.0)
    );
    vec3 H = safeNormalize(L + V, N);

    float diffuseFactor = max(dot(N, L), 0.0);

    float specularFactor = 0.0;
    if (diffuseFactor > 0.0) {
        specularFactor = pow(
            max(dot(N, H), 0.0),
            scene.materialSpecularShininess.w
        );
    }

    vec3 ambient =
        scene.ambientColorIntensity.rgb *
        scene.ambientColorIntensity.w *
        scene.materialAlbedo.rgb;

    vec3 diffuse =
        scene.lightColorIntensity.rgb *
        scene.lightColorIntensity.w *
        diffuseFactor *
        scene.materialAlbedo.rgb;

    vec3 specular =
        scene.lightColorIntensity.rgb *
        scene.lightColorIntensity.w *
        specularFactor *
        scene.materialSpecularShininess.rgb;

    outColor = vec4(ambient + diffuse + specular, 1.0);
}
