#include "application.hpp"

#include <imgui.h>
#include <vulkan/vulkan.h>

#define STB_IMAGE_IMPLEMENTATION // библиотека для декодирование png
#include "stb_image.h"

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

struct Vec2 {
    float x, y;
};

struct Vec3 {
    float x, y, z;
};

struct alignas(16) Vec4 {
    float x, y, z, w;
};

struct alignas(16) Mat4 {
    float m[16]{};
};

// В лабе 3 добавились UV.
struct Vertex {
    // Где вершина находится в 3D.
    Vec3 position;
     // Нормаль для Blinn-Phong освещения.
    Vec3 normal;
    // координата на двумерной текстуре.
    Vec2 uv;
    // u - горизонталь, v - вертикаль
};

struct alignas(16) SceneData {
    Mat4 model;
    Mat4 view;
    Mat4 projection;

    Vec4 cameraPosition;
    Vec4 lightDirection;
    Vec4 lightColorIntensity;
    Vec4 ambientColorIntensity;
    Vec4 materialSpecularShininess;
};

// Vertex -данные вершин/index -номера вершин buffers теперь будут в VRAM.
VkBuffer vertexBuffer = VK_NULL_HANDLE;
VmaAllocation vertexAllocation = VK_NULL_HANDLE;

VkBuffer indexBuffer = VK_NULL_HANDLE;
VmaAllocation indexAllocation = VK_NULL_HANDLE;

uint32_t indexCount = 0;

// Dynamic uniform buffer.
VkBuffer uniformBuffer = VK_NULL_HANDLE;
VmaAllocation uniformAllocation = VK_NULL_HANDLE;
void* uniformMapped = nullptr;

VkDeviceSize uniformStride = 0;

// Текстура. непосредственно графическое изображение в памяти GPU.
VkImage textureImage = VK_NULL_HANDLE;
VmaAllocation textureAllocation = VK_NULL_HANDLE;

// как интерпретировать это изображение. память изображения.
VkImageView textureImageView = VK_NULL_HANDLE;

// как читать изображение по UV.
VkSampler textureSampler = VK_NULL_HANDLE;

// Descriptor set.
VkDescriptorSetLayout descriptorSetLayout =
    VK_NULL_HANDLE;

VkDescriptorPool descriptorPool =
    VK_NULL_HANDLE;

VkDescriptorSet descriptorSet =
    VK_NULL_HANDLE;

VkPipelineLayout pipelineLayout =
    VK_NULL_HANDLE;

VkPipeline pipeline =
    VK_NULL_HANDLE;

// Command pool для staging-копирования.
VkCommandPool uploadCommandPool =
    VK_NULL_HANDLE;

// Камера из лабы 2.
float cameraPosition[3] = {
    0.0f,
    0.0f,
    4.0f
};

float cameraRotation[3] = {
    0.0f,
    0.0f,
    0.0f
};

float cameraFov = 60.0f;

// Directional light.
float lightDirection[3] = {
    -1.0f,
    -1.0f,
    -1.0f
};

float lightColor[3] = {
    1.0f,
    1.0f,
    1.0f
};

float lightIntensity = 1.0f;

// Ambient.
float ambientColor[3] = {
    1.0f,
    1.0f,
    1.0f
};

float ambientIntensity = 0.12f;

// Материал.
// Albedo теперь берётся из texture.
float specularColor[3] = {
    1.0f,
    1.0f,
    1.0f
};

float shininess = 32.0f;

Vec3 sub(Vec3 a, Vec3 b) {
    return {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    };
}

Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

Vec3 normalize(Vec3 v) {
    const float length =
        std::sqrt(
            v.x * v.x +
            v.y * v.y +
            v.z * v.z
        );

    if (length < 0.000001f) {
        return {
            0.0f,
            1.0f,
            0.0f
        };
    }

    return {
        v.x / length,
        v.y / length,
        v.z / length
    };
}

Mat4 identity() {
    Mat4 r{};

    r.m[0] = 1.0f;
    r.m[5] = 1.0f;
    r.m[10] = 1.0f;
    r.m[15] = 1.0f;

    return r;
}

Mat4 multiply(
    const Mat4& a,
    const Mat4& b
) {
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

Mat4 translation(
    float x,
    float y,
    float z
) {
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

Mat4 perspective(
    float fov,
    float aspect,
    float nearPlane,
    float farPlane
) {
    Mat4 r{};

    const float f =
        1.0f /
        std::tan(fov * 0.5f);

    r.m[0] = f / aspect;
    r.m[5] = -f;

    r.m[10] =
        farPlane /
        (nearPlane - farPlane);

    r.m[11] = -1.0f;

    r.m[14] =
        farPlane *
        nearPlane /
        (nearPlane - farPlane);

    return r;
}

// Host-visible buffer.
// Используется для staging и UBO.
bool createMappedBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    const void* data,
    VkBuffer& buffer,
    VmaAllocation& allocation,
    void** mapped = nullptr
) {
    auto& context =
        graphics::internal::context;

    VkBufferCreateInfo bufferInfo{};

    bufferInfo.sType =
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

    bufferInfo.size = size;
    bufferInfo.usage = usage;

    bufferInfo.sharingMode =
        VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationInfo{};

    allocationInfo.usage =
        VMA_MEMORY_USAGE_AUTO;

    allocationInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | // Staging buffer — временный промежуточный буфер (доступен cpu)
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

        std::cerr
            << "Failed to create mapped buffer\n";

        return false;
    }

    if (data) {
        std::memcpy(
            resultInfo.pMappedData,
            data,
            static_cast<size_t>(size)
        );

        vmaFlushAllocation(
            context.allocator,
            allocation,
            0,
            size
        );
    }

    if (mapped) {
        *mapped =
            resultInfo.pMappedData;
    }

    return true;
}

// Буфер в VRAM.
bool createDeviceBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkBuffer& buffer,
    VmaAllocation& allocation
) {
    auto& context =
        graphics::internal::context;

    VkBufferCreateInfo bufferInfo{};

    bufferInfo.sType =
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

    bufferInfo.size = size;

    bufferInfo.usage =
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        usage;

    bufferInfo.sharingMode =
        VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationInfo{};

    allocationInfo.usage =
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    allocationInfo.requiredFlags =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (vmaCreateBuffer(
            context.allocator,
            &bufferInfo,
            &allocationInfo,
            &buffer,
            &allocation,
            nullptr
        ) != VK_SUCCESS) {

        std::cerr
            << "Failed to create VRAM buffer\n";

        return false;
    }

    return true;
}

bool createUploadCommandPool() {
    auto& context =
        graphics::internal::context;

    VkCommandPoolCreateInfo info{};

    info.sType =
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;

    info.flags =
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

    info.queueFamilyIndex =
        context.graphics_queue_index;

    if (vkCreateCommandPool(
            context.device,
            &info,
            nullptr,
            &uploadCommandPool
        ) != VK_SUCCESS) {

        std::cerr
            << "Failed to create upload command pool\n";

        return false;
    }

    return true;
}

VkCommandBuffer beginUploadCommands() {
    auto& context =
        graphics::internal::context;

    VkCommandBufferAllocateInfo info{};

    info.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;

    info.commandPool =
        uploadCommandPool;

    info.level =
        VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    info.commandBufferCount = 1;

    VkCommandBuffer cmd =
        VK_NULL_HANDLE;

    if (vkAllocateCommandBuffers(
            context.device,
            &info,
            &cmd
        ) != VK_SUCCESS) {

        return VK_NULL_HANDLE;
    }

    VkCommandBufferBeginInfo begin{};

    begin.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    begin.flags =
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(
            cmd,
            &begin
        ) != VK_SUCCESS) {

        vkFreeCommandBuffers(
            context.device,
            uploadCommandPool,
            1,
            &cmd
        );

        return VK_NULL_HANDLE;
    }

    return cmd;
}

bool endUploadCommands(
    VkCommandBuffer cmd
) {
    auto& context =
        graphics::internal::context;

    if (vkEndCommandBuffer(cmd) !=
        VK_SUCCESS) {

        return false;
    }

    VkSubmitInfo submit{};

    submit.sType =
        VK_STRUCTURE_TYPE_SUBMIT_INFO;

    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    if (vkQueueSubmit(
            context.graphics_queue,
            1,
            &submit,
            VK_NULL_HANDLE
        ) != VK_SUCCESS) {

        return false;
    }

    vkQueueWaitIdle(
        context.graphics_queue
    );

    vkFreeCommandBuffers(
        context.device,
        uploadCommandPool,
        1,
        &cmd
    );

    return true;
}

bool copyBuffer(
    VkBuffer source,
    VkBuffer destination,
    VkDeviceSize size
) {
    VkCommandBuffer cmd =
        beginUploadCommands();

    if (!cmd) {
        return false;
    }

    VkBufferCopy copy{};

    copy.size = size;

    vkCmdCopyBuffer( // GPU сам копирует данные ИЗ staging buffer
        cmd,
        source,
        destination,
        1,
        &copy
    );

    return endUploadCommands(cmd);
}

// RAM -> staging -> VRAM.
bool uploadToDeviceBuffer(
    const void* data,
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkBuffer& buffer,
    VmaAllocation& allocation
) {
    auto& context =
        graphics::internal::context;

    VkBuffer stagingBuffer =
        VK_NULL_HANDLE;

    VmaAllocation stagingAllocation =
        VK_NULL_HANDLE;

    if (!createMappedBuffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            data,
            stagingBuffer,
            stagingAllocation
        )) {

        return false;
    }

    if (!createDeviceBuffer(
            size,
            usage,
            buffer,
            allocation
        )) {

        vmaDestroyBuffer(
            context.allocator,
            stagingBuffer,
            stagingAllocation
        );

        return false;
    }

    const bool result =
        copyBuffer(
            stagingBuffer,
            buffer,
            size
        );

    vmaDestroyBuffer(
        context.allocator,
        stagingBuffer,
        stagingAllocation
    );

    return result;
}

// Генерация конуса + UV.
bool createCone() {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    vertices.reserve(
        SEGMENTS * 6
    );

    indices.reserve(
        SEGMENTS * 6
    );

    const Vec3 apex{
        0.0f,
        1.0f,
        0.0f
    };

    const Vec3 center{
        0.0f,
        -1.0f,
        0.0f
    };

    const Vec3 baseNormal{
        0.0f,
        -1.0f,
        0.0f
    };

    for (
        uint32_t i = 0;
        i < SEGMENTS; 
        ++i
    ) {
        const float u0 =
            static_cast<float>(i) /
            static_cast<float>(
                SEGMENTS
            ); // uo от 0 до 1 меняется (координата начала)

        const float u1 =
            static_cast<float>(
                i + 1
            ) /
            static_cast<float>(
                SEGMENTS
            ); // (координата конца) - чтобы обернуть текстуру вокргу конуса
            // нормализованное положение по окрудности 0...1

        const float a0 = // углы в радианах для двух соседний точек
            2.0f * PI * u0;

        const float a1 =
            2.0f * PI * u1;
        // реално 3д основание
        const Vec3 current{
            std::cos(a0),
            -1.0f,
            std::sin(a0)
        };

        const Vec3 next{
            std::cos(a1),
            -1.0f,
            std::sin(a1)
        };

        const Vec3 sideNormal =
            normalize(
                cross(
                    sub(next, apex),
                    sub(current, apex)
                )
            );

        const uint32_t first =
            static_cast<uint32_t>(
                vertices.size()
            );

        // Боковая поверхность.
        vertices.push_back({
            apex, // вершина конуса
            sideNormal, // нормаль грани
            {
                (u0 + u1) * 0.5f, // u - примерно посередине сегмента
                0.0f // v - верх текстуры
            }
        });

        vertices.push_back({
            current, // левая нижняя точка
            sideNormal,
            {
                u0, // текущая горизонтальная позиция текстуры
                1.0f  // низ текстуры
            }
        });

        vertices.push_back({
            next, // правая нижняя точка
            sideNormal,
            {
                u1,
                1.0f
            }
        });

        // Основание.
        vertices.push_back({
            center,
            baseNormal,
            {
                0.5f, // середина текстуры
                0.5f
            }
        });

        // чтобы помещалас окружност в квадрат текстуры
        // * 0.5 превращает диапазон:
        // [-1, +1] -> [-0.5, +0.5]
        //
        // + 0.5 превращает его:
        // [-0.5, +0.5] -> [0, 1]
        vertices.push_back({
            current,
            baseNormal,
            {
                0.5f +
                    0.5f *
                    std::cos(a0),

                0.5f +
                    0.5f *
                    std::sin(a0)
            }
        });

        vertices.push_back({
            next,
            baseNormal,
            {
                0.5f +
                    0.5f *
                    std::cos(a1),

                0.5f +
                    0.5f *
                    std::sin(a1)
            }
        });

        for (
            uint32_t j = 0;
            j < 6;
            ++j
        ) {
            indices.push_back(
                first + j
            );
        }
    }

    indexCount =
        static_cast<uint32_t>(
            indices.size()
        );

    // Vertex data -> VRAM.
    if (!uploadToDeviceBuffer(
            vertices.data(),
            sizeof(Vertex) *
                vertices.size(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            vertexBuffer,
            vertexAllocation
        )) {

        return false;
    }

    // Index data -> VRAM.
    if (!uploadToDeviceBuffer(
            indices.data(),
            sizeof(uint32_t) *
                indices.size(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            indexBuffer,
            indexAllocation
        )) {

        return false;
    }

    return true;
}

bool transitionImage(
    VkImage image,
    VkImageLayout oldLayout,
    VkImageLayout newLayout
) {
    VkCommandBuffer cmd =
        beginUploadCommands();

    if (!cmd) {
        return false;
    }

    VkImageMemoryBarrier barrier{};

    barrier.sType =
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;

    barrier.oldLayout =
        oldLayout;

    barrier.newLayout =
        newLayout;

    barrier.srcQueueFamilyIndex =
        VK_QUEUE_FAMILY_IGNORED;

    barrier.dstQueueFamilyIndex =
        VK_QUEUE_FAMILY_IGNORED;

    barrier.image =
        image;

    barrier.subresourceRange.aspectMask =
        VK_IMAGE_ASPECT_COLOR_BIT;

    barrier.subresourceRange.baseMipLevel =
        0;

    barrier.subresourceRange.levelCount =
        1;

    barrier.subresourceRange.baseArrayLayer =
        0;

    barrier.subresourceRange.layerCount =
        1;

    VkPipelineStageFlags srcStage;
    VkPipelineStageFlags dstStage;

    if (
        oldLayout ==
            VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout ==
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    ) {
        barrier.srcAccessMask = 0;

        barrier.dstAccessMask =
            VK_ACCESS_TRANSFER_WRITE_BIT;

        srcStage =
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

        dstStage =
            VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (
        oldLayout ==
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
        newLayout ==
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    ) {
        barrier.srcAccessMask =
            VK_ACCESS_TRANSFER_WRITE_BIT;

        barrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT;

        srcStage =
            VK_PIPELINE_STAGE_TRANSFER_BIT;

        dstStage =
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else {
        return false;
    }

    vkCmdPipelineBarrier(
        cmd,
        srcStage,
        dstStage,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &barrier
    );

    return endUploadCommands(cmd);
}

bool copyBufferToImage(
    VkBuffer buffer,
    VkImage image,
    uint32_t width,
    uint32_t height
) {
    VkCommandBuffer cmd =
        beginUploadCommands();

    if (!cmd) {
        return false;
    }

    VkBufferImageCopy region{};

    region.imageSubresource.aspectMask =
        VK_IMAGE_ASPECT_COLOR_BIT;

    region.imageSubresource.mipLevel =
        0;

    region.imageSubresource.baseArrayLayer =
        0;

    region.imageSubresource.layerCount =
        1;

    region.imageExtent = {
        width,
        height,
        1
    };

    vkCmdCopyBufferToImage( // для текстуры гпу
        cmd,
        buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
    );

    return endUploadCommands(cmd);
}

bool createTexture() {
    auto& context =
        graphics::internal::context;

// чтение
    int width = 0;
    int height = 0;
    int channels = 0;

    // Переворачиваем изображение по вертикали чтобы ориентация совпала с UV
    stbi_set_flip_vertically_on_load(
        1
    );

    // Текстура загружается с диска.
    stbi_uc* pixels =
        stbi_load(
            "textures/cone.png",
            &width,
            &height,
            &channels,
            STBI_rgb_alpha
        );

    if (!pixels) {
        std::cerr
            << "Cannot load textures/cone.png\n";

        return false;
    }

    const VkDeviceSize imageSize =
        static_cast<VkDeviceSize>(
            width
        ) *
        static_cast<VkDeviceSize>(
            height
        ) *
        4; // четыре RGBA

    VkBuffer stagingBuffer =
        VK_NULL_HANDLE;

    VmaAllocation stagingAllocation =
        VK_NULL_HANDLE;

    if (!createMappedBuffer(
            imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            pixels,
            stagingBuffer,
            stagingAllocation
        )) {

        stbi_image_free(pixels);

        return false;
    }

    stbi_image_free(pixels);

    VkImageCreateInfo imageInfo{};

    imageInfo.sType =
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;

    imageInfo.imageType =
        VK_IMAGE_TYPE_2D;

    imageInfo.extent = {
        static_cast<uint32_t>(
            width
        ),

        static_cast<uint32_t>(
            height
        ),

        1
    };

    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;

    imageInfo.format =
        VK_FORMAT_R8G8B8A8_SRGB; // для албедо картинки

    imageInfo.tiling =
        VK_IMAGE_TILING_OPTIMAL;

    imageInfo.initialLayout =
        VK_IMAGE_LAYOUT_UNDEFINED;

    imageInfo.usage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT;

    imageInfo.sharingMode =
        VK_SHARING_MODE_EXCLUSIVE;

    imageInfo.samples =
        VK_SAMPLE_COUNT_1_BIT;

    VmaAllocationCreateInfo allocationInfo{};

    allocationInfo.usage =
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    allocationInfo.requiredFlags =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (vmaCreateImage(
            context.allocator,
            &imageInfo,
            &allocationInfo,
            &textureImage,
            &textureAllocation,
            nullptr
        ) != VK_SUCCESS) {

        vmaDestroyBuffer(
            context.allocator,
            stagingBuffer,
            stagingAllocation
        );

        return false;
    }

    bool success =
        transitionImage(
            textureImage,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        );

    if (success) {
        success =
            copyBufferToImage(
                stagingBuffer,
                textureImage,
                static_cast<uint32_t>(
                    width
                ),
                static_cast<uint32_t>(
                    height
                )
            );
    }

    if (success) {
        success =
            transitionImage(
                textureImage,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            );
    }

    vmaDestroyBuffer(
        context.allocator,
        stagingBuffer,
        stagingAllocation
    );

    if (!success) {
        return false;
    }

    // ImageView позволяет shader читать VkImage.
    VkImageViewCreateInfo viewInfo{};

    viewInfo.sType =
        VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;

    viewInfo.image =
        textureImage;

    viewInfo.viewType =
        VK_IMAGE_VIEW_TYPE_2D;

    viewInfo.format =
        VK_FORMAT_R8G8B8A8_SRGB;

    viewInfo.subresourceRange.aspectMask =
        VK_IMAGE_ASPECT_COLOR_BIT;

    viewInfo.subresourceRange.baseMipLevel =
        0;

    viewInfo.subresourceRange.levelCount =
        1;

    viewInfo.subresourceRange.baseArrayLayer =
        0;

    viewInfo.subresourceRange.layerCount =
        1;

    if (vkCreateImageView(
            context.device,
            &viewInfo,
            nullptr,
            &textureImageView
        ) != VK_SUCCESS) {

        return false;
    }

    // Sampler определяет способ чтения текстуры.
    VkSamplerCreateInfo samplerInfo{};

    samplerInfo.sType =
        VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;

    samplerInfo.magFilter =
        VK_FILTER_LINEAR; // фильтрация 

    samplerInfo.minFilter =
        VK_FILTER_LINEAR;

    samplerInfo.addressModeU =
        VK_SAMPLER_ADDRESS_MODE_REPEAT; // чтобы за пределами координат было 0-1-0-1-0-1

    samplerInfo.addressModeV =
        VK_SAMPLER_ADDRESS_MODE_REPEAT;

    samplerInfo.addressModeW =
        VK_SAMPLER_ADDRESS_MODE_REPEAT;

    samplerInfo.anisotropyEnable =
        VK_FALSE;

    samplerInfo.maxAnisotropy =
        1.0f;

    samplerInfo.unnormalizedCoordinates =
        VK_FALSE;

    samplerInfo.compareEnable =
        VK_FALSE;

    samplerInfo.mipmapMode =
        VK_SAMPLER_MIPMAP_MODE_LINEAR;

    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(
            context.device,
            &samplerInfo,
            nullptr,
            &textureSampler
        ) != VK_SUCCESS) {

        return false;
    }

    return true;
}

VkDeviceSize alignSize(
    VkDeviceSize size,
    VkDeviceSize alignment
) {
    if (alignment == 0) {
        return size;
    }

    return
        (size + alignment - 1) &
        ~(alignment - 1);
}

// binding 0 = dynamic UBO
// binding 1 = image + sampler
bool createDescriptors() {
    auto& context =
        graphics::internal::context;

    VkDescriptorSetLayoutBinding bindings[2]{};

    bindings[0].binding = 0;

    bindings[0].descriptorType =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;

    bindings[0].descriptorCount = 1;

    bindings[0].stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT |
        VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[1].binding = 1;

    bindings[1].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

    bindings[1].descriptorCount = 1;

    bindings[1].stageFlags =
        VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};

    layoutInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;

    layoutInfo.bindingCount = 2;

    layoutInfo.pBindings =
        bindings;

    if (vkCreateDescriptorSetLayout(
            context.device,
            &layoutInfo,
            nullptr,
            &descriptorSetLayout
        ) != VK_SUCCESS) {

        return false;
    }

    // Dynamic UBO должен учитывать alignment GPU.
    VkPhysicalDeviceProperties properties{};

    vkGetPhysicalDeviceProperties(
        context.physical_device,
        &properties
    );

    uniformStride =
        alignSize(
            sizeof(SceneData),
            properties.limits
                .minUniformBufferOffsetAlignment
        );

    if (!createMappedBuffer(
            uniformStride,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            nullptr,
            uniformBuffer,
            uniformAllocation,
            &uniformMapped
        )) {

        return false;
    }

    VkDescriptorPoolSize poolSizes[2]{};

    poolSizes[0].type =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;

    poolSizes[0].descriptorCount =
        1;

    poolSizes[1].type =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

    poolSizes[1].descriptorCount =
        1;

    VkDescriptorPoolCreateInfo poolInfo{};

    poolInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;

    poolInfo.maxSets = 1;

    poolInfo.poolSizeCount = 2;

    poolInfo.pPoolSizes =
        poolSizes;

    if (vkCreateDescriptorPool(
            context.device,
            &poolInfo,
            nullptr,
            &descriptorPool
        ) != VK_SUCCESS) {

        return false;
    }

    VkDescriptorSetAllocateInfo allocateInfo{};

    allocateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;

    allocateInfo.descriptorPool =
        descriptorPool;

    allocateInfo.descriptorSetCount =
        1;

    allocateInfo.pSetLayouts =
        &descriptorSetLayout;

    if (vkAllocateDescriptorSets(
            context.device,
            &allocateInfo,
            &descriptorSet
        ) != VK_SUCCESS) {

        return false;
    }

    VkDescriptorBufferInfo bufferInfo{};

    bufferInfo.buffer =
        uniformBuffer;

    bufferInfo.offset = 0;

    bufferInfo.range =
        sizeof(SceneData);

    VkDescriptorImageInfo imageInfo{};

    imageInfo.sampler =
        textureSampler;

    imageInfo.imageView =
        textureImageView;

    imageInfo.imageLayout =
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet writes[2]{};

    writes[0].sType =
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

    writes[0].dstSet =
        descriptorSet;

    writes[0].dstBinding = 0;

    writes[0].descriptorCount =
        1;

    writes[0].descriptorType =
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;

    writes[0].pBufferInfo =
        &bufferInfo;

    writes[1].sType =
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

    writes[1].dstSet =
        descriptorSet;

    writes[1].dstBinding = 1;

    writes[1].descriptorCount =
        1;

    writes[1].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

    writes[1].pImageInfo =
        &imageInfo;

    vkUpdateDescriptorSets(
        context.device,
        2,
        writes,
        0,
        nullptr
    );

    return true;
}

VkShaderModule loadShader(
    const char* path
) {
    auto& context =
        graphics::internal::context;

    std::ifstream file(
        path,
        std::ios::binary |
        std::ios::ate
    );

    if (!file) {
        std::cerr
            << "Cannot open shader: "
            << path
            << '\n';

        return VK_NULL_HANDLE;
    }

    const std::streamsize size =
        file.tellg();

    file.seekg(0);

    std::vector<uint32_t> code(
        static_cast<size_t>(
            size
        ) /
        sizeof(uint32_t)
    );

    file.read(
        reinterpret_cast<char*>(
            code.data()
        ),
        size
    );

    VkShaderModuleCreateInfo info{};

    info.sType =
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

    info.codeSize =
        static_cast<size_t>(size);

    info.pCode =
        code.data();

    VkShaderModule module =
        VK_NULL_HANDLE;

    if (vkCreateShaderModule(
            context.device,
            &info,
            nullptr,
            &module
        ) != VK_SUCCESS) {

        return VK_NULL_HANDLE;
    }

    return module;
}

bool createPipeline() {
    auto& context =
        graphics::internal::context;

    VkShaderModule vert =
        loadShader(
            "shaders/cone.vert.spv"
        );

    VkShaderModule frag =
        loadShader(
            "shaders/cone.frag.spv"
        );

    if (!vert || !frag) {
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

    binding.stride =
        sizeof(Vertex);

    binding.inputRate =
        VK_VERTEX_INPUT_RATE_VERTEX;

    // position + normal + UV
    VkVertexInputAttributeDescription attributes[3]{};

    attributes[0].location = 0;
    attributes[0].binding = 0;

    attributes[0].format =
        VK_FORMAT_R32G32B32_SFLOAT;

    attributes[0].offset =
        offsetof(
            Vertex,
            position
        );

    attributes[1].location = 1;
    attributes[1].binding = 0;

    attributes[1].format =
        VK_FORMAT_R32G32B32_SFLOAT;

    attributes[1].offset =
        offsetof(
            Vertex,
            normal
        );

    attributes[2].location = 2;
    attributes[2].binding = 0;

    attributes[2].format =
        VK_FORMAT_R32G32_SFLOAT;

    attributes[2].offset =
        offsetof(
            Vertex,
            uv
        );

    VkPipelineVertexInputStateCreateInfo vertexInput{};

    vertexInput.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    vertexInput.vertexBindingDescriptionCount =
        1;

    vertexInput.pVertexBindingDescriptions =
        &binding;

    vertexInput.vertexAttributeDescriptionCount =
        3;

    vertexInput.pVertexAttributeDescriptions =
        attributes;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};

    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;

    inputAssembly.topology =
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{};

    viewportState.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;

    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};

    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;

    rasterizer.polygonMode =
        VK_POLYGON_MODE_FILL;

    rasterizer.cullMode =
        VK_CULL_MODE_NONE;

    rasterizer.frontFace =
        VK_FRONT_FACE_COUNTER_CLOCKWISE;

    rasterizer.lineWidth =
        1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};

    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;

    multisampling.rasterizationSamples =
        VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};

    depth.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    depth.depthTestEnable =
        VK_TRUE;

    depth.depthWriteEnable =
        VK_TRUE;

    depth.depthCompareOp =
        VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorAttachment{};

    colorAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blend{};

    blend.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

    blend.attachmentCount = 1;

    blend.pAttachments =
        &colorAttachment;

    const VkDynamicState states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamic{};

    dynamic.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;

    dynamic.dynamicStateCount = 2;

    dynamic.pDynamicStates =
        states;

    VkPipelineLayoutCreateInfo layoutInfo{};

    layoutInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

    layoutInfo.setLayoutCount = 1;

    layoutInfo.pSetLayouts =
        &descriptorSetLayout;

    if (vkCreatePipelineLayout(
            context.device,
            &layoutInfo,
            nullptr,
            &pipelineLayout
        ) != VK_SUCCESS) {

        return false;
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};

    pipelineInfo.sType =
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;

    pipelineInfo.pVertexInputState =
        &vertexInput;

    pipelineInfo.pInputAssemblyState =
        &inputAssembly;

    pipelineInfo.pViewportState =
        &viewportState;

    pipelineInfo.pRasterizationState =
        &rasterizer;

    pipelineInfo.pMultisampleState =
        &multisampling;

    pipelineInfo.pDepthStencilState =
        &depth;

    pipelineInfo.pColorBlendState =
        &blend;

    pipelineInfo.pDynamicState =
        &dynamic;

    pipelineInfo.layout =
        pipelineLayout;

    pipelineInfo.renderPass =
        context.render_pass;

    pipelineInfo.subpass = 0;

    const VkResult result =
        vkCreateGraphicsPipelines(
            context.device,
            VK_NULL_HANDLE,
            1,
            &pipelineInfo,
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

    return result ==
        VK_SUCCESS;
}

Mat4 makeView() {
    const float pitch =
        cameraRotation[0] *
        PI / 180.0f;

    const float yaw =
        cameraRotation[1] *
        PI / 180.0f;

    const float roll =
        cameraRotation[2] *
        PI / 180.0f;

    return multiply(
        rotationZ(-roll),
        multiply(
            rotationX(-pitch),
            multiply(
                rotationY(-yaw),
                translation(
                    -cameraPosition[0],
                    -cameraPosition[1],
                    -cameraPosition[2]
                )
            )
        )
    );
}

void updateUniform() {
    auto& context =
        graphics::internal::context;

    const float aspect =
        static_cast<float>(
            context.swapchain_extent.width
        ) /
        static_cast<float>(
            context.swapchain_extent.height
        );

    SceneData data{};

    data.model =
        rotationX(-0.30f);

    data.view =
        makeView();

    data.projection =
        perspective(
            cameraFov *
                PI / 180.0f,
            aspect,
            0.1f,
            100.0f
        );

    data.cameraPosition = {
        cameraPosition[0],
        cameraPosition[1],
        cameraPosition[2],
        1.0f
    };

    data.lightDirection = {
        lightDirection[0],
        lightDirection[1],
        lightDirection[2],
        0.0f
    };

    data.lightColorIntensity = {
        lightColor[0],
        lightColor[1],
        lightColor[2],
        lightIntensity
    };

    data.ambientColorIntensity = {
        ambientColor[0],
        ambientColor[1],
        ambientColor[2],
        ambientIntensity
    };

    data.materialSpecularShininess = {
        specularColor[0],
        specularColor[1],
        specularColor[2],
        shininess
    };

    std::memcpy(
        uniformMapped,
        &data,
        sizeof(data)
    );

    vmaFlushAllocation(
        context.allocator,
        uniformAllocation,
        0,
        uniformStride
    );
}

} // namespace

bool initialize() {
    if (
        !createUploadCommandPool() ||
        !createCone() ||
        !createTexture() ||
        !createDescriptors() ||
        !createPipeline()
    ) {
        shutdown();
        return false;
    }

    return true;
}

void shutdown() {
    auto& context =
        graphics::internal::context;

    vkQueueWaitIdle(
        context.graphics_queue
    );

    if (pipeline) {
        vkDestroyPipeline(
            context.device,
            pipeline,
            nullptr
        );
    }

    if (pipelineLayout) {
        vkDestroyPipelineLayout(
            context.device,
            pipelineLayout,
            nullptr
        );
    }

    if (descriptorPool) {
        vkDestroyDescriptorPool(
            context.device,
            descriptorPool,
            nullptr
        );
    }

    if (descriptorSetLayout) {
        vkDestroyDescriptorSetLayout(
            context.device,
            descriptorSetLayout,
            nullptr
        );
    }

    if (textureSampler) {
        vkDestroySampler(
            context.device,
            textureSampler,
            nullptr
        );
    }

    if (textureImageView) {
        vkDestroyImageView(
            context.device,
            textureImageView,
            nullptr
        );
    }

    if (textureImage) {
        vmaDestroyImage(
            context.allocator,
            textureImage,
            textureAllocation
        );
    }

    if (uniformBuffer) {
        vmaDestroyBuffer(
            context.allocator,
            uniformBuffer,
            uniformAllocation
        );
    }

    if (indexBuffer) {
        vmaDestroyBuffer(
            context.allocator,
            indexBuffer,
            indexAllocation
        );
    }

    if (vertexBuffer) {
        vmaDestroyBuffer(
            context.allocator,
            vertexBuffer,
            vertexAllocation
        );
    }

    if (uploadCommandPool) {
        vkDestroyCommandPool(
            context.device,
            uploadCommandPool,
            nullptr
        );
    }
}

void update(
    [[maybe_unused]] double time
) {
    ImGui::Begin("Lab 3");

    ImGui::TextUnformatted(
        "Albedo: textures/cone.png"
    );

    ImGui::Separator();

    ImGui::TextUnformatted("Camera");

    ImGui::DragFloat3(
        "Position",
        cameraPosition,
        0.05f
    );

    ImGui::DragFloat3(
        "Rotation",
        cameraRotation,
        1.0f
    );

    ImGui::SliderFloat(
        "FOV",
        &cameraFov,
        20.0f,
        120.0f
    );

    ImGui::Separator();

    ImGui::TextUnformatted(
        "Directional light"
    );

    ImGui::DragFloat3(
        "Direction",
        lightDirection,
        0.05f
    );

    ImGui::ColorEdit3(
        "Light color",
        lightColor
    );

    ImGui::SliderFloat(
        "Light intensity",
        &lightIntensity,
        0.0f,
        10.0f
    );

    ImGui::Separator();

    ImGui::TextUnformatted(
        "Ambient"
    );

    ImGui::ColorEdit3(
        "Ambient color",
        ambientColor
    );

    ImGui::SliderFloat(
        "Ambient intensity",
        &ambientIntensity,
        0.0f,
        1.0f
    );

    ImGui::Separator();

    ImGui::TextUnformatted(
        "Material"
    );

    ImGui::ColorEdit3(
        "Specular color",
        specularColor
    );

    ImGui::SliderFloat(
        "Shininess",
        &shininess,
        1.0f,
        128.0f
    );

    ImGui::End();
}

void render(
    const graphics::internal::FrameData& fd
) {
    auto& context =
        graphics::internal::context;

    updateUniform();

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

    clears[0].color.float32[0] =
        0.04f;

    clears[0].color.float32[1] =
        0.05f;

    clears[0].color.float32[2] =
        0.08f;

    clears[0].color.float32[3] =
        1.0f;

    clears[1].depthStencil.depth =
        1.0f;

    VkRenderPassBeginInfo renderPass{};

    renderPass.sType =
        VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;

    renderPass.renderPass =
        context.render_pass;

    renderPass.framebuffer =
        fd.framebuffer;

    renderPass.renderArea.offset =
        {0, 0};

    renderPass.renderArea.extent =
        context.swapchain_extent;

    renderPass.clearValueCount =
        2;

    renderPass.pClearValues =
        clears;

    vkCmdBeginRenderPass(
        fd.command_buffer,
        &renderPass,
        VK_SUBPASS_CONTENTS_INLINE
    );

    VkViewport viewport{};

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

    scissor.offset =
        {0, 0};

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

    // Объект один => первый элемент dynamic UBO.
    const uint32_t dynamicOffset = 0;

    vkCmdBindDescriptorSets(
        fd.command_buffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipelineLayout,
        0,
        1,
        &descriptorSet,
        1,
        &dynamicOffset
    );

    const VkDeviceSize offset =
        0;

    vkCmdBindVertexBuffers(
        fd.command_buffer,
        0,
        1,
        &vertexBuffer,
        &offset
    );

    vkCmdBindIndexBuffer(
        fd.command_buffer,
        indexBuffer,
        0,
        VK_INDEX_TYPE_UINT32
    );

    vkCmdDrawIndexed(
        fd.command_buffer,
        indexCount,
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
