#include "application.hpp"

#include <imgui.h>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace application {

namespace {

struct Vertex {
    float position[3];
    float color[3];
};

struct Mat4 {
    float m[16]{};
};

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_allocation = VK_NULL_HANDLE;

VkBuffer index_buffer = VK_NULL_HANDLE;
VmaAllocation index_allocation = VK_NULL_HANDLE;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

uint32_t index_count = 0;
Mat4 identity() {
    Mat4 r{};
    r.m[0] = 1.0f;
    r.m[5] = 1.0f;
    r.m[10] = 1.0f;
    r.m[15] = 1.0f;
    return r;
}

Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 r{};

    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            for (int k = 0; k < 4; ++k) {
                r.m[column * 4 + row] +=
                    a.m[k * 4 + row] *
                    b.m[column * 4 + k];
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

Mat4 perspective(
    float fov_y,
    float aspect,
    float near_plane,
    float far_plane
) {
    Mat4 r{};

    const float f = 1.0f / std::tan(fov_y * 0.5f);

    r.m[0] = f / aspect;

    // Vulkan: ?????????????? Y.
    r.m[5] = -f;

    r.m[10] =
        far_plane /
        (near_plane - far_plane);

    r.m[11] = -1.0f;

    r.m[14] =
        (far_plane * near_plane) /
        (near_plane - far_plane);

    return r;
}

bool createBuffer( // Эта функция создаёт память на GPU/для Vulkan buffer и копирует туда данные
    const void* data,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkBuffer& buffer,
    VmaAllocation& allocation
) {
    auto& context = graphics::internal::context;

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType =
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode =
        VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage =
        VMA_MEMORY_USAGE_AUTO;

    allocation_info.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo mapped_info{};

    if (vmaCreateBuffer(
            context.allocator,
            &buffer_info,
            &allocation_info,
            &buffer,
            &allocation,
            &mapped_info
        ) != VK_SUCCESS) {
        std::cerr << "Failed to create buffer\n";
        return false;
    }

    std::memcpy(
        mapped_info.pMappedData,
        data,
        static_cast<size_t>(size)
    );

    vmaFlushAllocation(
        context.allocator,
        allocation,
        0,
        size
    );

    return true;
}

VkShaderModule loadShader(const char* filename) {
    auto& context = graphics::internal::context;

    std::ifstream file(
        filename,
        std::ios::binary | std::ios::ate
    );

    if (!file) {
        std::cerr
            << "Cannot open shader: "
            << filename
            << '\n';

        return VK_NULL_HANDLE;
    }

    const std::streamsize size = file.tellg();
    file.seekg(0);

    std::vector<uint32_t> code(
        static_cast<size_t>(size + 3) / 4
    );

    file.read(
        reinterpret_cast<char*>(code.data()),
        size
    );

    VkShaderModuleCreateInfo info{};
    info.sType =
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize =
        static_cast<size_t>(size);
    info.pCode = code.data();

    VkShaderModule module = VK_NULL_HANDLE;

    if (vkCreateShaderModule(
            context.device,
            &info,
            nullptr,
            &module
        ) != VK_SUCCESS) {

        std::cerr
            << "Failed to create shader module\n";

        return VK_NULL_HANDLE;
    }

    return module;
}

bool createCone() {
    constexpr uint32_t segments = 32;
    constexpr float pi =
        3.14159265358979323846f;

    std::vector<Vertex> vertices; // хранит сами вершины
    std::vector<uint32_t> indices; // хранит номера вергшин, из которых собираем треугольники

    // 0 вершина конуса.
    vertices.push_back({
        {0.0f, 1.0f, 0.0f},
        {0.2f, 0.7f, 1.0f}
    });

    // 1 центр основания.
    vertices.push_back({
        {0.0f, -1.0f, 0.0f},
        {0.2f, 0.7f, 1.0f}
    });

    for (uint32_t i = 0; i < segments; ++i) {
        const float angle =
            2.0f * pi *
            static_cast<float>(i) /
            static_cast<float>(segments);

        // x, z - по углу в этой плоскости окружность, радиус равен 1
        const float x = std::cos(angle);
        const float z = std::sin(angle);
        // y - постоянная
        vertices.push_back({
            {x, -1.0f, z},
            {0.2f, 0.7f, 1.0f}
        });
    }

    // тут из вершины появляются треугольники
    for (uint32_t i = 0; i < segments; ++i) {
        const uint32_t current = 2 + i; // текщая вершина окружности
        const uint32_t next =
            2 + ((i + 1) % segments); // следующая

        // боковая часть
        indices.push_back(0); // вершина конус - индекс 0
        indices.push_back(current);
        indices.push_back(next);
        // основание
        indices.push_back(1);
        indices.push_back(next);
        indices.push_back(current);
    }

    // итого 64 оснвоания
    // 64 * 3 индекса

    index_count =
        static_cast<uint32_t>(indices.size());

    if (!createBuffer(
            vertices.data(),
            sizeof(Vertex) * vertices.size(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            vertex_buffer,
            vertex_allocation
        )) {
        return false;
    }

    if (!createBuffer(
            indices.data(),
            sizeof(uint32_t) * indices.size(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            index_buffer,
            index_allocation
        )) {
        return false;
    }

    return true;
}

bool createPipeline() {
    auto& context = graphics::internal::context;

    VkShaderModule vert =
        loadShader("shaders/cone.vert.spv");

    VkShaderModule frag =
        loadShader("shaders/cone.frag.spv");

    if (vert == VK_NULL_HANDLE ||
        frag == VK_NULL_HANDLE) {
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};

    stages[0].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage =
        VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName = "main";

    stages[1].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage =
        VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(Vertex);
    binding.inputRate =
        VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription attributes[2]{};

    attributes[0].location = 0;
    attributes[0].binding = 0;
    attributes[0].format =
        VK_FORMAT_R32G32B32_SFLOAT;
    attributes[0].offset =
        offsetof(Vertex, position);

    attributes[1].location = 1;
    attributes[1].binding = 0;
    attributes[1].format =
        VK_FORMAT_R32G32B32_SFLOAT;
    attributes[1].offset =
        offsetof(Vertex, color);

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions =
        &binding;

    vertex_input.vertexAttributeDescriptionCount = 2;
    vertex_input.pVertexAttributeDescriptions =
        attributes;

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology =
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode =
        VK_POLYGON_MODE_FILL;
    rasterizer.cullMode =
        VK_CULL_MODE_NONE;
    rasterizer.frontFace =
        VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples =
        VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable = VK_TRUE;
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp =
        VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments =
        &blend_attachment;

    VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates =
        dynamic_states;

    VkPushConstantRange push{};
    push.stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT;
    push.offset = 0;
    push.size = sizeof(Mat4);

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges =
        &push;

    if (vkCreatePipelineLayout(
            context.device,
            &layout_info,
            nullptr,
            &pipeline_layout
        ) != VK_SUCCESS) {

        std::cerr
            << "Failed to create pipeline layout\n";

        return false;
    }

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType =
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

    pipeline_info.stageCount = 2;
    pipeline_info.pStages = stages;
    pipeline_info.pVertexInputState =
        &vertex_input;
    pipeline_info.pInputAssemblyState =
        &input_assembly;
    pipeline_info.pViewportState =
        &viewport_state;
    pipeline_info.pRasterizationState =
        &rasterizer;
    pipeline_info.pMultisampleState =
        &multisampling;
    pipeline_info.pDepthStencilState =
        &depth;
    pipeline_info.pColorBlendState =
        &blend;
    pipeline_info.pDynamicState =
        &dynamic;

    pipeline_info.layout =
        pipeline_layout;
    pipeline_info.renderPass =
        context.render_pass;
    pipeline_info.subpass = 0;

    const VkResult result =
        vkCreateGraphicsPipelines(
            context.device,
            VK_NULL_HANDLE,
            1,
            &pipeline_info,
            nullptr,
            &pipeline
        );

    vkDestroyShaderModule(
        context.device,
        vert,
        nullptr
    );

    vkDestroyShaderModule(
        context.device,
        frag,
        nullptr
    );

    if (result != VK_SUCCESS) {
        std::cerr
            << "Failed to create graphics pipeline\n";
        return false;
    }

    return true;
}

} // namespace

bool initialize() {
    if (!createCone()) {
        return false;
    }

    if (!createPipeline()) {
        return false;
    }

    return true;
}

void shutdown() {
    auto& context = graphics::internal::context;

    vkQueueWaitIdle(context.graphics_queue);

    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(
            context.device,
            pipeline,
            nullptr
        );
    }

    if (pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(
            context.device,
            pipeline_layout,
            nullptr
        );
    }

    if (vertex_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            vertex_buffer,
            vertex_allocation
        );
    }

    if (index_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            index_buffer,
            index_allocation
        );
    }
}

void update(double time) {
    (void)time;

    ImGui::Begin("Lab 1");

    ImGui::Text("Variant 9: Cone");
    ImGui::Text("Vulkan");
    ImGui::Text("Perspective projection");

    ImGui::End();
}

void render(
    const graphics::internal::FrameData& fd
) {
    auto& context =
        graphics::internal::context;

    vkResetCommandBuffer(
        fd.command_buffer,
        0
    );

    VkCommandBufferBeginInfo begin{};
    begin.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags =
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(
        fd.command_buffer,
        &begin
    );

    VkClearValue clears[2]{};

    clears[0].color.float32[0] = 0.05f;
    clears[0].color.float32[1] = 0.07f;
    clears[0].color.float32[2] = 0.10f;
    clears[0].color.float32[3] = 1.0f;

    clears[1].depthStencil.depth = 1.0f;
    clears[1].depthStencil.stencil = 0;

    VkRenderPassBeginInfo render_pass{};
    render_pass.sType =
        VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass.renderPass =
        context.render_pass;
    render_pass.framebuffer =
        fd.framebuffer;

    render_pass.renderArea.offset =
        {0, 0};

    render_pass.renderArea.extent =
        context.swapchain_extent;

    render_pass.clearValueCount = 2;
    render_pass.pClearValues = clears;

    vkCmdBeginRenderPass(
        fd.command_buffer,
        &render_pass,
        VK_SUBPASS_CONTENTS_INLINE
    );

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width =
        static_cast<float>(
            context.swapchain_extent.width
        );
    viewport.height =
        static_cast<float>(
            context.swapchain_extent.height
        );
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    vkCmdSetViewport(
        fd.command_buffer,
        0,
        1,
        &viewport
    );

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent =
        context.swapchain_extent;

    vkCmdSetScissor(
        fd.command_buffer,
        0,
        1,
        &scissor
    );

    vkCmdBindPipeline(
        fd.command_buffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipeline
    );

    const VkDeviceSize offset = 0;

    vkCmdBindVertexBuffers(
        fd.command_buffer,
        0,
        1,
        &vertex_buffer,
        &offset
    );

    vkCmdBindIndexBuffer(
        fd.command_buffer,
        index_buffer,
        0,
        VK_INDEX_TYPE_UINT32
    );

    const float aspect =
        static_cast<float>(
            context.swapchain_extent.width
        ) /
        static_cast<float>(
            context.swapchain_extent.height
        );

    constexpr float pi =
        3.14159265358979323846f;

    const Mat4 model = rotationX(-0.30f);

    const Mat4 view =
        translation(
            0.0f,
            0.0f,
            -4.0f
        );

    const Mat4 projection =
        perspective(
            60.0f * pi / 180.0f,
            aspect,
            0.1f,
            100.0f
        );

    const Mat4 mvp =
        multiply(
            projection,
            multiply(view, model)
        );

    vkCmdPushConstants(
        fd.command_buffer,
        pipeline_layout,
        VK_SHADER_STAGE_VERTEX_BIT,
        0,
        sizeof(Mat4),
        &mvp
    );

    vkCmdDrawIndexed( // тут рисуется индексированная геометрия
        fd.command_buffer,
        index_count,
        1,
        0,
        0,
        0
    );

    vkCmdEndRenderPass(
        fd.command_buffer
    );

    vkEndCommandBuffer(
        fd.command_buffer
    );
}

} // namespace application

