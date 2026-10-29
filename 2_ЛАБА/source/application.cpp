#include "application.hpp"

#include <imgui.h>
#include <vulkan/vulkan.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace application {
namespace {

constexpr float PI = 3.14159265358979323846f;
constexpr uint32_t SEGMENTS = 32;

struct Vec3 { float x, y, z; };
struct alignas(16) Vec4 { float x, y, z, w; };
struct alignas(16) Mat4 { float m[16]{}; };

struct Vertex {
    Vec3 position; 
    Vec3 normal; 
};

struct alignas(16) SceneData {
    Mat4 model;
    Mat4 view;
    Mat4 projection;
    Vec4 cameraPosition;
    Vec4 lightDirection;
    Vec4 lightColorIntensity;
    Vec4 ambientColorIntensity;
    Vec4 materialAlbedo;
    Vec4 materialSpecularShininess;
};

VkBuffer vertexBuffer = VK_NULL_HANDLE;
VmaAllocation vertexAllocation = VK_NULL_HANDLE;
uint32_t vertexCount = 0;

VkBuffer uniformBuffer = VK_NULL_HANDLE;
VmaAllocation uniformAllocation = VK_NULL_HANDLE;
void* uniformMapped = nullptr;

VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

// изменяемые параметры камеры
float cameraPosition[3] = {0.0f, 0.0f, 4.0f};
float cameraRotation[3] = {0.0f, 0.0f, 0.0f};
float cameraFov = 60.0f;

float lightDirection[3] = {-1.0f, -1.0f, -1.0f};
float lightColor[3] = {1.0f, 1.0f, 1.0f};
float lightIntensity = 1.0f;

float ambientColor[3] = {1.0f, 1.0f, 1.0f};
float ambientIntensity = 0.12f;

float albedo[3] = {0.2f, 0.65f, 1.0f};
float specularColor[3] = {1.0f, 1.0f, 1.0f};
float shininess = 32.0f;

Vec3 sub(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

Vec3 normalize(Vec3 v) {
    const float length = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return length > 0.000001f
        ? Vec3{v.x / length, v.y / length, v.z / length}
        : Vec3{0.0f, 1.0f, 0.0f};
}

Mat4 identity() {
    Mat4 r{};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            for (int k = 0; k < 4; ++k) {
                r.m[column * 4 + row] +=
                    a.m[k * 4 + row] * b.m[column * 4 + k];
            }
        }
    }
    return r;
}

Mat4 translation(float x, float y, float z) {
    Mat4 r = identity();
    r.m[12] = x;
    r.m[13] = y;
    r.m[14] = z;
    return r;
}

Mat4 rotationX(float angle) {
    Mat4 r = identity();
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    r.m[5] = c;
    r.m[6] = s;
    r.m[9] = -s;
    r.m[10] = c;
    return r;
}

Mat4 rotationY(float angle) {
    Mat4 r = identity();
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    r.m[0] = c;
    r.m[2] = -s;
    r.m[8] = s;
    r.m[10] = c;
    return r;
}

Mat4 rotationZ(float angle) {
    Mat4 r = identity();
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    r.m[0] = c;
    r.m[1] = s;
    r.m[4] = -s;
    r.m[5] = c;
    return r;
}

Mat4 perspective(float fov, float aspect, float nearPlane, float farPlane) {
    Mat4 r{};
    const float f = 1.0f / std::tan(fov * 0.5f);
    r.m[0] = f / aspect;
    r.m[5] = -f;
    r.m[10] = farPlane / (nearPlane - farPlane);
    r.m[11] = -1.0f;
    r.m[14] = farPlane * nearPlane / (nearPlane - farPlane);
    return r;
}

bool createMappedBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    const void* data,
    VkBuffer& buffer,
    VmaAllocation& allocation,
    void** mapped = nullptr
) {
    auto& context = graphics::internal::context;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo resultInfo{};
    if (vmaCreateBuffer(
            context.allocator,
            &bufferInfo,
            &allocationInfo,
            &buffer,
            &allocation,
            &resultInfo
        ) != VK_SUCCESS) {
        std::cerr << "Failed to create buffer\n";
        return false;
    }

    if (data) {
        std::memcpy(resultInfo.pMappedData, data, static_cast<size_t>(size));
        vmaFlushAllocation(context.allocator, allocation, 0, size);
    }
    if (mapped) {
        *mapped = resultInfo.pMappedData;
    }
    return true;
}

bool createCone() {
    std::vector<Vertex> vertices;
    vertices.reserve(SEGMENTS * 6);

    const Vec3 apex{0.0f, 1.0f, 0.0f};
    const Vec3 center{0.0f, -1.0f, 0.0f};
    const Vec3 baseNormal{0.0f, -1.0f, 0.0f};

    for (uint32_t i = 0; i < SEGMENTS; ++i) {
        const float a0 = 2.0f * PI * static_cast<float>(i) / SEGMENTS;
        const float a1 = 2.0f * PI * static_cast<float>(i + 1) / SEGMENTS;

        const Vec3 current{std::cos(a0), -1.0f, std::sin(a0)};
        const Vec3 next{std::cos(a1), -1.0f, std::sin(a1)};
        const Vec3 sideNormal = normalize(cross(sub(next, apex), sub(current, apex)));

        vertices.push_back({apex, sideNormal});
        vertices.push_back({current, sideNormal});
        vertices.push_back({next, sideNormal});

        vertices.push_back({center, baseNormal});
        vertices.push_back({current, baseNormal});
        vertices.push_back({next, baseNormal});
    }

    vertexCount = static_cast<uint32_t>(vertices.size());
    return createMappedBuffer(
        static_cast<VkDeviceSize>(sizeof(Vertex) * vertices.size()),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        vertices.data(),
        vertexBuffer,
        vertexAllocation
    );
}

bool createDescriptors() {
    auto& context = graphics::internal::context;

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(
            context.device, &layoutInfo, nullptr, &descriptorSetLayout
        ) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout\n";
        return false;
    }

    if (!createMappedBuffer(
            sizeof(SceneData),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            nullptr,
            uniformBuffer,
            uniformAllocation,
            &uniformMapped
        )) {
        return false;
    }

    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool\n";
        return false;
    }

    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = descriptorPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &descriptorSetLayout;
    if (vkAllocateDescriptorSets(context.device, &allocateInfo, &descriptorSet) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor set\n";
        return false;
    }

    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = uniformBuffer;
    bufferInfo.range = sizeof(SceneData);

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.pBufferInfo = &bufferInfo;
    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
    return true;
}

VkShaderModule loadShader(const char* path) {
    auto& context = graphics::internal::context;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open shader: " << path << '\n';
        return VK_NULL_HANDLE;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0 || size % 4 != 0) {
        std::cerr << "Invalid shader: " << path << '\n';
        return VK_NULL_HANDLE;
    }

    file.seekg(0);
    std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(code.data()), size);

    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = static_cast<size_t>(size);
    info.pCode = code.data();

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(context.device, &info, nullptr, &module) != VK_SUCCESS) {
        std::cerr << "Failed to create shader module: " << path << '\n';
        return VK_NULL_HANDLE;
    }
    return module;
}

bool createPipeline() {
    auto& context = graphics::internal::context;
    const VkShaderModule vert = loadShader("shaders/cone.vert.spv");
    const VkShaderModule frag = loadShader("shaders/cone.frag.spv");
    if (!vert || !frag) {
        if (vert) vkDestroyShaderModule(context.device, vert, nullptr);
        if (frag) vkDestroyShaderModule(context.device, frag, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = static_cast<uint32_t>(sizeof(Vertex));
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributes[2]{};
    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset = static_cast<uint32_t>(offsetof(Vertex, position));
    attributes[1].location = 1;
    attributes[1].binding = 0;
    attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset = static_cast<uint32_t>(offsetof(Vertex, normal));

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 2;
    vertexInput.pVertexAttributeDescriptions = attributes;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &blendAttachment;

    const VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamicStates;

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descriptorSetLayout;
    if (vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout\n";
        vkDestroyShaderModule(context.device, vert, nullptr);
        vkDestroyShaderModule(context.device, frag, nullptr);
        return false;
    }

    VkGraphicsPipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &inputAssembly;
    info.pViewportState = &viewportState;
    info.pRasterizationState = &rasterizer;
    info.pMultisampleState = &multisampling;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = pipelineLayout;
    info.renderPass = context.render_pass;

    const VkResult result = vkCreateGraphicsPipelines(
        context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline
    );
    vkDestroyShaderModule(context.device, vert, nullptr);
    vkDestroyShaderModule(context.device, frag, nullptr);

    if (result != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline\n";
        return false;
    }
    return true;
}

// Матрица вида
Mat4 makeView() {
    // углы из градусов в радианы
    const float pitch = cameraRotation[0] * PI / 180.0f;
    const float yaw = cameraRotation[1] * PI / 180.0f;
    const float roll = cameraRotation[2] * PI / 180.0f;

    return multiply(
        rotationZ(-roll),
        multiply(
            rotationX(-pitch),
            multiply(
                rotationY(-yaw),
                translation(-cameraPosition[0], -cameraPosition[1], -cameraPosition[2])
            )
        )
    );
}

void updateUniform() {
    auto& context = graphics::internal::context;
    const float aspect =
        static_cast<float>(context.swapchain_extent.width) /
        static_cast<float>(context.swapchain_extent.height);

    SceneData data{};
    data.model = rotationX(-0.30f);
    data.view = makeView();
    data.projection = perspective(cameraFov * PI / 180.0f, aspect, 0.1f, 100.0f);
    data.cameraPosition = {cameraPosition[0], cameraPosition[1], cameraPosition[2], 1.0f};
    data.lightDirection = {lightDirection[0], lightDirection[1], lightDirection[2], 0.0f};
    data.lightColorIntensity = {lightColor[0], lightColor[1], lightColor[2], lightIntensity};
    data.ambientColorIntensity = {ambientColor[0], ambientColor[1], ambientColor[2], ambientIntensity};
    data.materialAlbedo = {albedo[0], albedo[1], albedo[2], 1.0f};
    data.materialSpecularShininess = {
        specularColor[0], specularColor[1], specularColor[2], shininess
    };

    std::memcpy(uniformMapped, &data, sizeof(data));
    vmaFlushAllocation(context.allocator, uniformAllocation, 0, sizeof(data));
}

} // namespace

bool initialize() {
    if (!createCone() || !createDescriptors() || !createPipeline()) {
        shutdown();
        return false;
    }
    return true;
}

void shutdown() {
    auto& context = graphics::internal::context;
    vkQueueWaitIdle(context.graphics_queue);

    if (pipeline) vkDestroyPipeline(context.device, pipeline, nullptr);
    if (pipelineLayout) vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
    if (descriptorPool) vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
    if (descriptorSetLayout) vkDestroyDescriptorSetLayout(context.device, descriptorSetLayout, nullptr);
    if (uniformBuffer) vmaDestroyBuffer(context.allocator, uniformBuffer, uniformAllocation);
    if (vertexBuffer) vmaDestroyBuffer(context.allocator, vertexBuffer, vertexAllocation);
}

void update([[maybe_unused]] double time) {
    ImGui::Begin("Lab 2");

    ImGui::TextUnformatted("Camera");
    ImGui::DragFloat3("Position", cameraPosition, 0.05f);
    ImGui::DragFloat3("Rotation", cameraRotation, 1.0f);
    ImGui::SliderFloat("FOV", &cameraFov, 20.0f, 120.0f);

    ImGui::Separator();
    ImGui::TextUnformatted("Directional light");
    ImGui::DragFloat3("Direction", lightDirection, 0.05f);
    ImGui::ColorEdit3("Light color", lightColor);
    ImGui::SliderFloat("Light intensity", &lightIntensity, 0.0f, 10.0f);

    ImGui::Separator();
    ImGui::TextUnformatted("Ambient light");
    ImGui::ColorEdit3("Ambient color", ambientColor);
    ImGui::SliderFloat("Ambient intensity", &ambientIntensity, 0.0f, 1.0f);

    ImGui::Separator();
    ImGui::TextUnformatted("Material");
    ImGui::ColorEdit3("Albedo", albedo);
    ImGui::ColorEdit3("Specular color", specularColor);
    ImGui::SliderFloat("Shininess", &shininess, 1.0f, 128.0f);

    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    auto& context = graphics::internal::context;
    updateUniform();

    vkResetCommandBuffer(fd.command_buffer, 0);

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(fd.command_buffer, &begin);

    VkClearValue clears[2]{};
    clears[0].color.float32[0] = 0.04f;
    clears[0].color.float32[1] = 0.05f;
    clears[0].color.float32[2] = 0.08f;
    clears[0].color.float32[3] = 1.0f;
    clears[1].depthStencil.depth = 1.0f;
    clears[1].depthStencil.stencil = 0;

    VkRenderPassBeginInfo renderPass{};
    renderPass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPass.renderPass = context.render_pass;
    renderPass.framebuffer = fd.framebuffer;
    renderPass.renderArea.offset = {0, 0};
    renderPass.renderArea.extent = context.swapchain_extent;
    renderPass.clearValueCount = 2;
    renderPass.pClearValues = clears;
    vkCmdBeginRenderPass(fd.command_buffer, &renderPass, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.width = static_cast<float>(context.swapchain_extent.width);
    viewport.height = static_cast<float>(context.swapchain_extent.height);
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = context.swapchain_extent;
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(
        fd.command_buffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipelineLayout,
        0,
        1,
        &descriptorSet,
        0,
        nullptr
    );

    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertexBuffer, &offset);
    vkCmdDraw(fd.command_buffer, vertexCount, 1, 0, 0);

    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
