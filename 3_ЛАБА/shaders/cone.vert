#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
// UV пришли из vertex buffer.
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragWorldPosition;
layout(location = 1) out vec3 fragWorldNormal;
layout(location = 2) out vec2 fragUV;

layout(set = 0, binding = 0)
uniform SceneData {
    mat4 model;
    mat4 view;
    mat4 projection;

    vec4 cameraPosition;
    vec4 lightDirection;
    vec4 lightColorIntensity;
    vec4 ambientColorIntensity;
    vec4 materialSpecularShininess;
} scene;

void main() {
    vec4 worldPosition =
        scene.model *
        vec4(inPosition, 1.0);

    fragWorldPosition =
        worldPosition.xyz;

    fragWorldNormal =
        normalize(
            transpose(
                inverse(
                    mat3(scene.model)
                )
            ) *
            inNormal
        );

    // ???????? UV ?? fragment shader.
    fragUV = inUV;

    gl_Position =
        scene.projection *
        scene.view *
        worldPosition;
}

