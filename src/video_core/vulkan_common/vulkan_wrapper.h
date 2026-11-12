// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <string>
#include <vector>
#include <fmt/format.h>
#include <vulkan/vulkan_core.h>

#include "common/common_types.h"
#include "video_core/vulkan_common/vulkan.h"
#include "video_core/gpu_logging/gpu_logging.h"

#ifdef _MSC_VER
#pragma warning(disable : 26812) // Disable prefer enum class over enum
#endif

VK_DEFINE_HANDLE(VmaAllocator)
VK_DEFINE_HANDLE(VmaAllocation)

namespace Vulkan::vk {

namespace diagnostic {
template <typename Handle>
[[nodiscard]] inline u64 HandleBits(Handle handle) noexcept {
    if constexpr (std::is_pointer_v<Handle>) {
        return static_cast<u64>(reinterpret_cast<std::uintptr_t>(handle));
    } else {
        return static_cast<u64>(handle);
    }
}

template <typename... Args>
inline void Trace(const char* category, fmt::format_string<Args...> format,
                  Args&&... args) noexcept {
    if (!GPU::Logging::IsDiagnosticTraceEnabled()) {
        return;
    }
    try {
        GPU::Logging::LogDiagnosticTrace(category,
                                         fmt::format(format, std::forward<Args>(args)...));
    } catch (...) {
        // Optional diagnostics must never terminate a noexcept Vulkan wrapper.
    }
}
} // namespace diagnostic

/**
 * Span for Vulkan arrays.
 * Based on std::span but optimized for array access instead of iterators.
 * Size returns uint32_t instead of size_t to ease interaction with Vulkan functions.
 */
template <typename T>
class Span {
public:
    using value_type = T;
    using size_type = u32;
    using difference_type = std::ptrdiff_t;
    using reference = const T&;
    using const_reference = const T&;
    using pointer = const T*;
    using const_pointer = const T*;
    using iterator = const T*;
    using const_iterator = const T*;

    /// Construct an empty span.
    constexpr Span() noexcept = default;

    /// Construct an empty span
    constexpr Span(std::nullptr_t) noexcept {}

    /// Construct a span from a single element.
    constexpr Span(const T& value) noexcept : ptr{&value}, num{1} {}

    /// Construct a span from a range.
    template <typename Range>
    // requires std::data(const Range&)
    // requires std::size(const Range&)
    constexpr Span(const Range& range) : ptr{std::data(range)}, num{std::size(range)} {}

    /// Construct a span from a pointer and a size.
    /// This is intended for subranges.
    constexpr Span(const T* ptr_, std::size_t num_) noexcept : ptr{ptr_}, num{num_} {}

    /// Returns the data pointer by the span.
    constexpr const T* data() const noexcept {
        return ptr;
    }

    /// Returns the number of elements in the span.
    /// @note Returns a 32 bits integer because most Vulkan functions expect this type.
    constexpr u32 size() const noexcept {
        return static_cast<u32>(num);
    }

    /// Returns true when the span is empty.
    constexpr bool empty() const noexcept {
        return num == 0;
    }

    /// Returns a reference to the element in the passed index.
    /// @pre: index < size()
    constexpr const T& operator[](std::size_t index) const noexcept {
        return ptr[index];
    }

    /// Returns an iterator to the beginning of the span.
    constexpr const T* begin() const noexcept {
        return ptr;
    }

    /// Returns an iterator to the end of the span.
    constexpr const T* end() const noexcept {
        return ptr + num;
    }

    /// Returns an iterator to the beginning of the span.
    constexpr const T* cbegin() const noexcept {
        return ptr;
    }

    /// Returns an iterator to the end of the span.
    constexpr const T* cend() const noexcept {
        return ptr + num;
    }

private:
    const T* ptr = nullptr;
    std::size_t num = 0;
};

/// Vulkan exception generated from a VkResult.
class Exception final : public std::exception {
public:
    /// Construct the exception with a result.
    /// @pre result != VK_SUCCESS
    explicit Exception(VkResult result_) : result{result_} {}
    virtual ~Exception() = default;

    const char* what() const noexcept override;
    VkResult GetResult() const noexcept {
        return result;
    }

private:
    VkResult result;
};

/// Throws a Vulkan exception if result is not success.
inline void Check(VkResult result) {
    if (result != VK_SUCCESS) {
        throw Exception(result);
    }
}

/// Throws a Vulkan exception if result is an error.
/// @return result
inline VkResult Filter(VkResult result) {
    if (result < 0) {
        throw Exception(result);
    }
    return result;
}

inline constexpr VkPipelineStageFlags PIPELINE_STAGE_GRAPHICS_COMPUTE =
    VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

inline constexpr VkPipelineStageFlags PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER =
    PIPELINE_STAGE_GRAPHICS_COMPUTE | VK_PIPELINE_STAGE_TRANSFER_BIT;

inline constexpr VkPipelineStageFlags PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER_HOST =
    PIPELINE_STAGE_GRAPHICS_COMPUTE_TRANSFER | VK_PIPELINE_STAGE_HOST_BIT;

inline constexpr VkPipelineStageFlags PIPELINE_STAGE_HOST = VK_PIPELINE_STAGE_HOST_BIT;


/// Table holding Vulkan instance function pointers.
struct InstanceDispatch {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr{};

    PFN_vkCreateInstance vkCreateInstance{};
    PFN_vkDestroyInstance vkDestroyInstance{};
    PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties{};
    PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties{};

    PFN_vkCreateDebugUtilsMessengerEXT vkCreateDebugUtilsMessengerEXT{};
    PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT{};
    PFN_vkCreateDevice vkCreateDevice{};
    PFN_vkDestroyDebugUtilsMessengerEXT vkDestroyDebugUtilsMessengerEXT{};
    PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT{};
    PFN_vkDestroyDevice vkDestroyDevice{};
    PFN_vkDestroySurfaceKHR vkDestroySurfaceKHR{};
    PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties{};
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices{};
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr{};
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR
        vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR{};
    PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2{};
    PFN_vkGetPhysicalDeviceFormatProperties vkGetPhysicalDeviceFormatProperties{};
    PFN_vkGetPhysicalDeviceFormatProperties2 vkGetPhysicalDeviceFormatProperties2{};
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties{};
    PFN_vkGetPhysicalDeviceMemoryProperties2 vkGetPhysicalDeviceMemoryProperties2{};
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties{};
    PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2{};
    PFN_vkGetPhysicalDeviceToolProperties vkGetPhysicalDeviceToolProperties{};
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties{};
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR vkGetPhysicalDeviceSurfaceCapabilitiesKHR{};
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR vkGetPhysicalDeviceSurfaceFormatsKHR{};
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR vkGetPhysicalDeviceSurfacePresentModesKHR{};
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR vkGetPhysicalDeviceSurfaceSupportKHR{};
    PFN_vkGetSwapchainImagesKHR vkGetSwapchainImagesKHR{};
    PFN_vkQueuePresentKHR vkQueuePresentKHR{};
};

/// Table holding Vulkan device function pointers.
struct DeviceDispatch : InstanceDispatch {
    PFN_vkAcquireNextImageKHR vkAcquireNextImageKHR{};
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers{};
    PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets{};
    PFN_vkAllocateMemory vkAllocateMemory{};
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer{};
    PFN_vkBindBufferMemory vkBindBufferMemory{};
    PFN_vkBindImageMemory vkBindImageMemory{};
    PFN_vkCmdBeginConditionalRenderingEXT vkCmdBeginConditionalRenderingEXT{};
    PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT{};
    PFN_vkCmdBeginQuery vkCmdBeginQuery{};
    PFN_vkCmdBeginRenderPass vkCmdBeginRenderPass{};
    PFN_vkCmdBeginTransformFeedbackEXT vkCmdBeginTransformFeedbackEXT{};
    PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets{};
    PFN_vkCmdBindIndexBuffer vkCmdBindIndexBuffer{};
    PFN_vkCmdBindPipeline vkCmdBindPipeline{};
    PFN_vkCmdBindTransformFeedbackBuffersEXT vkCmdBindTransformFeedbackBuffersEXT{};
    PFN_vkCmdBindVertexBuffers vkCmdBindVertexBuffers{};
    PFN_vkCmdBindVertexBuffers2EXT vkCmdBindVertexBuffers2EXT{};
    PFN_vkCmdBlitImage vkCmdBlitImage{};
    PFN_vkCmdClearAttachments vkCmdClearAttachments{};
    PFN_vkCmdClearColorImage vkCmdClearColorImage{};
    PFN_vkCmdCopyBuffer vkCmdCopyBuffer{};
    PFN_vkCmdCopyBufferToImage vkCmdCopyBufferToImage{};
    PFN_vkCmdCopyImage vkCmdCopyImage{};
    PFN_vkCmdCopyImageToBuffer vkCmdCopyImageToBuffer{};
    PFN_vkCmdCopyQueryPoolResults vkCmdCopyQueryPoolResults{};
    PFN_vkCmdDispatch vkCmdDispatch{};
    PFN_vkCmdDispatchIndirect vkCmdDispatchIndirect{};
    PFN_vkCmdDraw vkCmdDraw{};
    PFN_vkCmdDrawIndexed vkCmdDrawIndexed{};
    PFN_vkCmdDrawIndirect vkCmdDrawIndirect{};
    PFN_vkCmdDrawIndexedIndirect vkCmdDrawIndexedIndirect{};
    PFN_vkCmdDrawIndirectCount vkCmdDrawIndirectCount{};
    PFN_vkCmdDrawIndexedIndirectCount vkCmdDrawIndexedIndirectCount{};
    PFN_vkCmdDrawIndirectByteCountEXT vkCmdDrawIndirectByteCountEXT{};
    PFN_vkCmdDrawMeshTasksEXT vkCmdDrawMeshTasksEXT{};
    PFN_vkCmdEndConditionalRenderingEXT vkCmdEndConditionalRenderingEXT{};
    PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT{};
    PFN_vkCmdEndQuery vkCmdEndQuery{};
    PFN_vkCmdEndRenderPass vkCmdEndRenderPass{};
    PFN_vkCmdEndTransformFeedbackEXT vkCmdEndTransformFeedbackEXT{};
    PFN_vkCmdFillBuffer vkCmdFillBuffer{};
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier{};
    PFN_vkCmdPipelineBarrier2 vkCmdPipelineBarrier2{};
    PFN_vkCmdPushConstants vkCmdPushConstants{};
    PFN_vkCmdPushDescriptorSetWithTemplateKHR vkCmdPushDescriptorSetWithTemplateKHR{};
    PFN_vkCmdResetQueryPool vkCmdResetQueryPool{};
    PFN_vkCmdWriteTimestamp vkCmdWriteTimestamp{};
    PFN_vkCmdResolveImage vkCmdResolveImage{};
    PFN_vkCmdSetBlendConstants vkCmdSetBlendConstants{};
    PFN_vkCmdSetCullModeEXT vkCmdSetCullModeEXT{};
    PFN_vkCmdSetDepthBias vkCmdSetDepthBias{};
    PFN_vkCmdSetDepthBias2EXT vkCmdSetDepthBias2EXT{};
    PFN_vkCmdSetDepthBounds vkCmdSetDepthBounds{};
    PFN_vkCmdSetDepthBoundsTestEnableEXT vkCmdSetDepthBoundsTestEnableEXT{};
    PFN_vkCmdSetDepthCompareOpEXT vkCmdSetDepthCompareOpEXT{};
    PFN_vkCmdSetDepthTestEnableEXT vkCmdSetDepthTestEnableEXT{};
    PFN_vkCmdSetDepthWriteEnableEXT vkCmdSetDepthWriteEnableEXT{};
    PFN_vkCmdSetPrimitiveRestartEnableEXT vkCmdSetPrimitiveRestartEnableEXT{};
    PFN_vkCmdSetRasterizerDiscardEnableEXT vkCmdSetRasterizerDiscardEnableEXT{};
    PFN_vkCmdSetAlphaToCoverageEnableEXT vkCmdSetAlphaToCoverageEnableEXT{};
    PFN_vkCmdSetAlphaToOneEnableEXT vkCmdSetAlphaToOneEnableEXT{};
    PFN_vkCmdSetConservativeRasterizationModeEXT vkCmdSetConservativeRasterizationModeEXT{};
    PFN_vkCmdSetLineRasterizationModeEXT vkCmdSetLineRasterizationModeEXT{};
    PFN_vkCmdSetLineStippleEnableEXT vkCmdSetLineStippleEnableEXT{};
    PFN_vkCmdSetLineStippleEXT vkCmdSetLineStippleEXT{};
    PFN_vkCmdSetDepthBiasEnableEXT vkCmdSetDepthBiasEnableEXT{};
    PFN_vkCmdSetLogicOpEnableEXT vkCmdSetLogicOpEnableEXT{};
    PFN_vkCmdSetDepthClampEnableEXT vkCmdSetDepthClampEnableEXT{};
    PFN_vkCmdSetEvent vkCmdSetEvent{};
    PFN_vkCmdSetFrontFaceEXT vkCmdSetFrontFaceEXT{};
    PFN_vkCmdSetPatchControlPointsEXT vkCmdSetPatchControlPointsEXT{};
    PFN_vkCmdSetLogicOpEXT vkCmdSetLogicOpEXT{};
    PFN_vkCmdSetLineWidth vkCmdSetLineWidth{};
    PFN_vkCmdSetPrimitiveTopologyEXT vkCmdSetPrimitiveTopologyEXT{};
    PFN_vkCmdSetScissor vkCmdSetScissor{};
    PFN_vkCmdSetStencilCompareMask vkCmdSetStencilCompareMask{};
    PFN_vkCmdSetStencilOpEXT vkCmdSetStencilOpEXT{};
    PFN_vkCmdSetStencilReference vkCmdSetStencilReference{};
    PFN_vkCmdSetStencilTestEnableEXT vkCmdSetStencilTestEnableEXT{};
    PFN_vkCmdSetStencilWriteMask vkCmdSetStencilWriteMask{};
    PFN_vkCmdSetVertexInputEXT vkCmdSetVertexInputEXT{};
    PFN_vkCmdSetViewport vkCmdSetViewport{};
    PFN_vkCmdSetColorWriteMaskEXT vkCmdSetColorWriteMaskEXT{};
    PFN_vkCmdSetColorWriteEnableEXT vkCmdSetColorWriteEnableEXT{};
    PFN_vkCmdSetColorBlendEnableEXT vkCmdSetColorBlendEnableEXT{};
    PFN_vkCmdSetColorBlendEquationEXT vkCmdSetColorBlendEquationEXT{};
    PFN_vkCmdWaitEvents vkCmdWaitEvents{};
    PFN_vkCreateBuffer vkCreateBuffer{};
    PFN_vkCreateBufferView vkCreateBufferView{};
    PFN_vkCreateCommandPool vkCreateCommandPool{};
    PFN_vkCreateComputePipelines vkCreateComputePipelines{};
    PFN_vkCreateDescriptorPool vkCreateDescriptorPool{};
    PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout{};
    PFN_vkCreateDescriptorUpdateTemplate vkCreateDescriptorUpdateTemplate{};
    PFN_vkCreateEvent vkCreateEvent{};
    PFN_vkCreateFence vkCreateFence{};
    PFN_vkCreateFramebuffer vkCreateFramebuffer{};
    PFN_vkCreateGraphicsPipelines vkCreateGraphicsPipelines{};
    PFN_vkCreateImage vkCreateImage{};
    PFN_vkCreateImageView vkCreateImageView{};
    PFN_vkCreatePipelineCache vkCreatePipelineCache{};
    PFN_vkCreatePipelineLayout vkCreatePipelineLayout{};
    PFN_vkCreateQueryPool vkCreateQueryPool{};
    PFN_vkCreateRenderPass vkCreateRenderPass{};
    PFN_vkCreateRenderPass2 vkCreateRenderPass2{};
    PFN_vkCreateSampler vkCreateSampler{};
    PFN_vkCreateSemaphore vkCreateSemaphore{};
    PFN_vkCreateShaderModule vkCreateShaderModule{};
    PFN_vkCreateSwapchainKHR vkCreateSwapchainKHR{};
    PFN_vkDestroyBuffer vkDestroyBuffer{};
    PFN_vkDestroyBufferView vkDestroyBufferView{};
    PFN_vkDestroyCommandPool vkDestroyCommandPool{};
    PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool{};
    PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout{};
    PFN_vkDestroyDescriptorUpdateTemplate vkDestroyDescriptorUpdateTemplate{};
    PFN_vkDestroyEvent vkDestroyEvent{};
    PFN_vkDestroyFence vkDestroyFence{};
    PFN_vkDestroyFramebuffer vkDestroyFramebuffer{};
    PFN_vkDestroyImage vkDestroyImage{};
    PFN_vkDestroyImageView vkDestroyImageView{};
    PFN_vkDestroyPipeline vkDestroyPipeline{};
    PFN_vkDestroyPipelineCache vkDestroyPipelineCache{};
    PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout{};
    PFN_vkDestroyQueryPool vkDestroyQueryPool{};
    PFN_vkDestroyRenderPass vkDestroyRenderPass{};
    PFN_vkDestroySampler vkDestroySampler{};
    PFN_vkDestroySemaphore vkDestroySemaphore{};
    PFN_vkDestroyShaderModule vkDestroyShaderModule{};
    PFN_vkDestroySwapchainKHR vkDestroySwapchainKHR{};
    PFN_vkDeviceWaitIdle vkDeviceWaitIdle{};
    PFN_vkEndCommandBuffer vkEndCommandBuffer{};
    PFN_vkFreeCommandBuffers vkFreeCommandBuffers{};
    PFN_vkFreeDescriptorSets vkFreeDescriptorSets{};
    PFN_vkFreeMemory vkFreeMemory{};
    PFN_vkGetBufferMemoryRequirements2 vkGetBufferMemoryRequirements2{};
    PFN_vkGetDeviceQueue vkGetDeviceQueue{};
    PFN_vkGetEventStatus vkGetEventStatus{};
    PFN_vkGetFenceStatus vkGetFenceStatus{};
    PFN_vkGetImageMemoryRequirements vkGetImageMemoryRequirements{};
    PFN_vkGetImageSparseMemoryRequirements vkGetImageSparseMemoryRequirements{};
    PFN_vkGetPipelineCacheData vkGetPipelineCacheData{};
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR{};
#ifdef _WIN32
    PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR{};
#endif
    PFN_vkGetDeviceFaultInfoEXT vkGetDeviceFaultInfoEXT{};
    PFN_vkGetPipelineExecutablePropertiesKHR vkGetPipelineExecutablePropertiesKHR{};
    PFN_vkGetPipelineExecutableStatisticsKHR vkGetPipelineExecutableStatisticsKHR{};
    PFN_vkGetQueryPoolResults vkGetQueryPoolResults{};
    PFN_vkGetSemaphoreCounterValue vkGetSemaphoreCounterValue{};
    PFN_vkMapMemory vkMapMemory{};
    PFN_vkQueueBindSparse vkQueueBindSparse{};
    PFN_vkQueueSubmit vkQueueSubmit{};
    PFN_vkQueueSubmit2 vkQueueSubmit2{};
    PFN_vkResetFences vkResetFences{};
    PFN_vkResetQueryPool vkResetQueryPool{};
    PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT{};
    PFN_vkSetDebugUtilsObjectTagEXT vkSetDebugUtilsObjectTagEXT{};
    PFN_vkUnmapMemory vkUnmapMemory{};
    PFN_vkUpdateDescriptorSetWithTemplate vkUpdateDescriptorSetWithTemplate{};
    PFN_vkGetBufferDeviceAddress vkGetBufferDeviceAddress{};
    PFN_vkGetDescriptorSetLayoutSizeEXT vkGetDescriptorSetLayoutSizeEXT{};
    PFN_vkGetDescriptorSetLayoutBindingOffsetEXT vkGetDescriptorSetLayoutBindingOffsetEXT{};
    PFN_vkGetDescriptorEXT vkGetDescriptorEXT{};
    PFN_vkCmdBindDescriptorBuffersEXT vkCmdBindDescriptorBuffersEXT{};
    PFN_vkCmdSetDescriptorBufferOffsetsEXT vkCmdSetDescriptorBufferOffsetsEXT{};
    PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets{};
    PFN_vkWaitForFences vkWaitForFences{};
    PFN_vkWaitSemaphores vkWaitSemaphores{};
};

/// Loads instance agnostic function pointers.
/// @return True on success, false on error.
bool Load(InstanceDispatch&) noexcept;

/// Loads instance function pointers.
/// @return True on success, false on error.
bool Load(VkInstance, InstanceDispatch&) noexcept;

void Destroy(VkInstance, const InstanceDispatch&) noexcept;
void Destroy(VkDevice, const InstanceDispatch&) noexcept;

void Destroy(VkDevice, VkBuffer, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkBufferView, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkCommandPool, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkDescriptorPool, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkDescriptorSetLayout, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkDescriptorUpdateTemplate, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkDeviceMemory, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkEvent, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkFence, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkFramebuffer, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkImage, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkImageView, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkPipeline, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkPipelineCache, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkPipelineLayout, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkQueryPool, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkRenderPass, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkSampler, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkSwapchainKHR, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkSemaphore, const DeviceDispatch&) noexcept;
void Destroy(VkDevice, VkShaderModule, const DeviceDispatch&) noexcept;
void Destroy(VkInstance, VkDebugUtilsMessengerEXT, const InstanceDispatch&) noexcept;
void Destroy(VkInstance, VkDebugReportCallbackEXT, const InstanceDispatch&) noexcept;
void Destroy(VkInstance, VkSurfaceKHR, const InstanceDispatch&) noexcept;

VkResult Free(VkDevice, VkDescriptorPool, Span<VkDescriptorSet>, const DeviceDispatch&) noexcept;
VkResult Free(VkDevice, VkCommandPool, Span<VkCommandBuffer>, const DeviceDispatch&) noexcept;

template <typename Type, typename OwnerType, typename Dispatch>
class Handle;

/// Handle with an owning type.
/// Analogue to std::unique_ptr.
template <typename Type, typename OwnerType, typename Dispatch>
class Handle {
public:
    /// Construct a handle and hold it's ownership.
    explicit Handle(Type handle_, OwnerType owner_, const Dispatch& dld_) noexcept
        : handle{handle_}, owner{owner_}, dld{&dld_} {}

    /// Construct an empty handle.
    Handle() = default;

    /// Construct an empty handle.
    Handle(std::nullptr_t) {}

    /// Copying Vulkan objects is not supported and will never be.
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    /// Construct a handle transferring the ownership from another handle.
    Handle(Handle&& rhs) noexcept
        : handle{std::exchange(rhs.handle, Type{})}, owner{rhs.owner}, dld{rhs.dld} {}

    /// Assign the current handle transferring the ownership from another handle.
    /// Destroys any previously held object.
    Handle& operator=(Handle&& rhs) noexcept {
        Release();
        handle = std::exchange(rhs.handle, Type{});
        owner = rhs.owner;
        dld = rhs.dld;
        return *this;
    }

    /// Destroys the current handle if it existed.
    ~Handle() noexcept {
        Release();
    }

    /// Destroys any held object.
    void reset() noexcept {
        Release();
        handle = Type{};
    }

    /// Returns the address of the held object.
    /// Intended for Vulkan structures that expect a pointer to an array.
    const Type* address() const noexcept {
        return std::addressof(handle);
    }

    /// Returns the held Vulkan handle.
    Type operator*() const noexcept {
        return handle;
    }

    /// Returns true when there's a held object.
    explicit operator bool() const noexcept {
        return handle != Type{};
    }

#ifndef __ANDROID__
    /**
     * Releases ownership of the managed handle.
     * The caller is responsible for managing the lifetime of the returned handle.
     * The Handle object becomes invalid after this call.
     */
    Type release() noexcept {
        return std::exchange(handle, nullptr);
    }
#endif

protected:
    Type handle{};
    OwnerType owner = nullptr;
    const Dispatch* dld = nullptr;

private:
    /// Destroys the held object if it exists.
    void Release() noexcept {
        if (handle) {
            Destroy(OwnerType(owner), Type(handle), *dld);
        }
    }
};

/// Dummy type used to specify a handle has no owner.
struct NoOwner {};

/// Handle without an owning type.
/// Analogue to std::unique_ptr
template <typename Type, typename Dispatch>
class Handle<Type, NoOwner, Dispatch> {
public:
    /// Construct a handle and hold it's ownership.
    explicit Handle(Type handle_, const Dispatch& dld_) noexcept : handle{handle_}, dld{&dld_} {}

    /// Construct an empty handle.
    Handle() noexcept = default;

    /// Copying Vulkan objects is not supported and will never be.
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;

    /// Construct a handle transferring ownership from another handle.
    Handle(Handle&& rhs) noexcept : handle{std::exchange(rhs.handle, nullptr)}, dld{rhs.dld} {}

    /// Assign the current handle transferring the ownership from another handle.
    /// Destroys any previously held object.
    Handle& operator=(Handle&& rhs) noexcept {
        Release();
        handle = std::exchange(rhs.handle, nullptr);
        dld = rhs.dld;
        return *this;
    }

    /// Destroys the current handle if it existed.
    ~Handle() noexcept {
        Release();
    }

    /// Destroys any held object.
    void reset() noexcept {
        Release();
        handle = {};
    }

    /// Returns the address of the held object.
    /// Intended for Vulkan structures that expect a pointer to an array.
    const Type* address() const noexcept {
        return std::addressof(handle);
    }

    /// Returns the held Vulkan handle.
    Type operator*() const noexcept {
        return handle;
    }

    /// Returns true when there's a held object.
    explicit operator bool() const noexcept {
        return handle != Type{};
    }

#ifndef __ANDROID__
    /**
     * Releases ownership of the managed handle.
     * The caller is responsible for managing the lifetime of the returned handle.
     * The Handle object becomes invalid after this call.
     */
    Type release() noexcept {
        return std::exchange(handle, nullptr);
    }
#endif

protected:
    Type handle{};
    const Dispatch* dld = nullptr;

private:
    /// Destroys the held object if it exists.
    void Release() noexcept {
        if (handle) {
            Destroy(handle, *dld);
        }
    }
};

/// Array of a pool allocation.
/// Analogue to std::vector
template <typename AllocationType, typename PoolType>
class PoolAllocations {
public:
    /// Construct an empty allocation.
    PoolAllocations() = default;

    /// Construct an allocation. Errors are reported through IsOutOfPoolMemory().
    explicit PoolAllocations(std::unique_ptr<AllocationType[]> allocations_, std::size_t num_,
                             VkDevice device_, PoolType pool_, const DeviceDispatch& dld_) noexcept
        : allocations{std::move(allocations_)}, num{num_}, device{device_}, pool{pool_},
          dld{&dld_} {}

    /// Copying Vulkan allocations is not supported and will never be.
    PoolAllocations(const PoolAllocations&) = delete;
    PoolAllocations& operator=(const PoolAllocations&) = delete;

    /// Construct an allocation transferring ownership from another allocation.
    PoolAllocations(PoolAllocations&& rhs) noexcept
        : allocations{std::move(rhs.allocations)}, num{rhs.num}, device{rhs.device}, pool{rhs.pool},
          dld{rhs.dld} {}

    /// Assign an allocation transferring ownership from another allocation.
    PoolAllocations& operator=(PoolAllocations&& rhs) noexcept {
        allocations = std::move(rhs.allocations);
        num = rhs.num;
        device = rhs.device;
        pool = rhs.pool;
        dld = rhs.dld;
        return *this;
    }

    /// Returns the number of allocations.
    std::size_t size() const noexcept {
        return num;
    }

    /// Returns a pointer to the array of allocations.
    AllocationType const* data() const noexcept {
        return allocations.get();
    }

    /// Returns the allocation in the specified index.
    /// @pre index < size()
    AllocationType operator[](std::size_t index) const noexcept {
        return allocations[index];
    }

    /// True when a pool fails to construct.
    bool IsOutOfPoolMemory() const noexcept {
        return !device;
    }

private:
    std::unique_ptr<AllocationType[]> allocations;
    std::size_t num = 0;
    VkDevice device = nullptr;
    PoolType pool{};
    const DeviceDispatch* dld = nullptr;
};

using DebugUtilsMessenger = Handle<VkDebugUtilsMessengerEXT, VkInstance, InstanceDispatch>;
using DebugReportCallback = Handle<VkDebugReportCallbackEXT, VkInstance, InstanceDispatch>;
using DescriptorSetLayout = Handle<VkDescriptorSetLayout, VkDevice, DeviceDispatch>;
using DescriptorUpdateTemplate = Handle<VkDescriptorUpdateTemplate, VkDevice, DeviceDispatch>;
using Pipeline = Handle<VkPipeline, VkDevice, DeviceDispatch>;
using PipelineLayout = Handle<VkPipelineLayout, VkDevice, DeviceDispatch>;
using QueryPool = Handle<VkQueryPool, VkDevice, DeviceDispatch>;
using RenderPass = Handle<VkRenderPass, VkDevice, DeviceDispatch>;
using Sampler = Handle<VkSampler, VkDevice, DeviceDispatch>;
using SurfaceKHR = Handle<VkSurfaceKHR, VkInstance, InstanceDispatch>;

using DescriptorSets = PoolAllocations<VkDescriptorSet, VkDescriptorPool>;
using CommandBuffers = PoolAllocations<VkCommandBuffer, VkCommandPool>;

/// Vulkan instance owning handle.
class Instance : public Handle<VkInstance, NoOwner, InstanceDispatch> {
    using Handle<VkInstance, NoOwner, InstanceDispatch>::Handle;

public:
    /// Creates a Vulkan instance.
    /// @throw Exception on initialization error.
    [[nodiscard]] static Instance Create(u32 version, Span<const char*> layers, Span<const char*> extensions,
                           InstanceDispatch& dispatch);

    /// Enumerates physical devices.
    /// @return Physical devices and an empty handle on failure.
    /// @throw Exception on Vulkan error.
    std::vector<VkPhysicalDevice> EnumeratePhysicalDevices() const;

    /// Creates a debug callback messenger.
    /// @throw Exception on creation failure.
    [[nodiscard]] DebugUtilsMessenger CreateDebugUtilsMessenger(
        const VkDebugUtilsMessengerCreateInfoEXT& create_info) const;

    /// Creates a debug report callback.
    /// @throw Exception on creation failure.
    [[nodiscard]] DebugReportCallback CreateDebugReportCallback(
        const VkDebugReportCallbackCreateInfoEXT& create_info) const;

    /// Returns dispatch table.
    const InstanceDispatch& Dispatch() const noexcept {
        return *dld;
    }
};

class Image {
public:
    explicit Image(VkImage handle_, VkImageUsageFlags usage_, VkDevice owner_,
                   VmaAllocator allocator_, VmaAllocation allocation_,
                   const DeviceDispatch& dld_) noexcept
        : handle{handle_}, usage{usage_}, owner{owner_}, allocator{allocator_},
          allocation{allocation_}, dld{&dld_} {}
    Image() = default;

    /// True for an image created without memory (an unbacked sparse image).
    bool IsUnbacked() const noexcept {
        return handle != VK_NULL_HANDLE && allocation == VK_NULL_HANDLE;
    }

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&& rhs) noexcept
        : handle{std::exchange(rhs.handle, VkImage{})}, usage{rhs.usage}, owner{rhs.owner},
          allocator{rhs.allocator}, allocation{rhs.allocation}, dld{rhs.dld} {}

    Image& operator=(Image&& rhs) noexcept {
        Release();
        handle = std::exchange(rhs.handle, VkImage{});
        usage = rhs.usage;
        owner = rhs.owner;
        allocator = rhs.allocator;
        allocation = rhs.allocation;
        dld = rhs.dld;
        return *this;
    }

    ~Image() noexcept {
        Release();
    }

    VkImage operator*() const noexcept {
        return handle;
    }

    void reset() noexcept {
        Release();
        handle = VkImage{};
    }

    explicit operator bool() const noexcept {
        return handle != VkImage{};
    }

    void SetObjectNameEXT(const char* name) const;

    [[nodiscard]] VkImageUsageFlags UsageFlags() const noexcept {
        return usage;
    }

    /// Where the image's memory lies: the block (0 when the image has no memory or a block of
    /// its own), the block's size and the image's part of it.
    struct MemoryBlock {
        u64 block;
        u64 block_size;
        u64 size;
    };
    [[nodiscard]] MemoryBlock GetMemoryBlock() const noexcept;

private:
    void Release() const noexcept;

    VkImage handle{};
    VkImageUsageFlags usage{};
    VkDevice owner = nullptr;
    VmaAllocator allocator = nullptr;
    VmaAllocation allocation = nullptr;
    const DeviceDispatch* dld = nullptr;
};

struct MemoryLocation {
    VkDeviceMemory memory{};
    VkDeviceSize offset{};
    u32 memory_type{};
};

class Buffer {
public:
    explicit Buffer(VkBuffer handle_, VkDevice owner_, VmaAllocator allocator_,
                    VmaAllocation allocation_, std::span<u8> mapped_, bool is_coherent_,
                    MemoryLocation location_, const DeviceDispatch& dld_) noexcept
        : handle{handle_}, owner{owner_}, allocator{allocator_},
          allocation{allocation_}, mapped{mapped_}, location{location_},
          is_coherent{is_coherent_}, dld{&dld_} {}
    Buffer() = default;

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& rhs) noexcept
        : handle{std::exchange(rhs.handle, VkBuffer{})}, owner{rhs.owner}, allocator{rhs.allocator},
          allocation{rhs.allocation}, mapped{rhs.mapped}, location{rhs.location},
          is_coherent{rhs.is_coherent}, dld{rhs.dld} {}

    Buffer& operator=(Buffer&& rhs) noexcept {
        Release();
        handle = std::exchange(rhs.handle, VkBuffer{});
        owner = rhs.owner;
        allocator = rhs.allocator;
        allocation = rhs.allocation;
        mapped = rhs.mapped;
        location = rhs.location;
        is_coherent = rhs.is_coherent;
        dld = rhs.dld;
        return *this;
    }

    ~Buffer() noexcept {
        Release();
    }

    VkBuffer operator*() const noexcept {
        return handle;
    }

    void reset() noexcept {
        Release();
        handle = VkBuffer{};
    }

    explicit operator bool() const noexcept {
        return handle != VkBuffer{};
    }

    /// Returns the host mapped memory, an empty span otherwise.
    std::span<u8> Mapped() noexcept {
        return mapped;
    }

    std::span<const u8> Mapped() const noexcept {
        return mapped;
    }

    /// Returns true if the buffer is mapped to the host.
    bool IsHostVisible() const noexcept {
        return !mapped.empty();
    }

    /// Returns true if host writes are visible to the device.
    bool IsHostCoherent() const noexcept {
        return is_coherent;
    }

    void Flush() const;

    void Invalidate() const;

    void SetObjectNameEXT(const char* name) const;

    MemoryLocation Location() const noexcept {
        return location;
    }

private:
    void Release() const noexcept;

    VkBuffer handle{};
    VkDevice owner = nullptr;
    VmaAllocator allocator = nullptr;
    VmaAllocation allocation = nullptr;
    std::span<u8> mapped = {};
    MemoryLocation location{};
    bool is_coherent = false;
    const DeviceDispatch* dld = nullptr;
};

class Queue {
public:
    /// Construct an empty queue handle.
    constexpr Queue() noexcept = default;

    /// Construct a queue handle.
    constexpr Queue(VkQueue queue_, const DeviceDispatch& dld_) noexcept
        : queue{queue_}, dld{&dld_} {}

    VkResult Submit(Span<VkSubmitInfo> submit_infos,
                    VkFence fence = VK_NULL_HANDLE) const noexcept {
        diagnostic::Trace("QUEUE_SUBMIT",
                          "queue={:#x} submit_count={} fence={:#x} api=vkQueueSubmit",
                          diagnostic::HandleBits(queue), submit_infos.size(),
                          diagnostic::HandleBits(fence));
        for (u32 i = 0; i < submit_infos.size(); ++i) {
            const auto& info = submit_infos[i];
            diagnostic::Trace("QUEUE_SUBMIT_INFO",
                              "queue={:#x} submit_index={} waits={} command_buffers={} signals={}",
                              diagnostic::HandleBits(queue), i, info.waitSemaphoreCount,
                              info.commandBufferCount, info.signalSemaphoreCount);
            for (auto* next = static_cast<const VkBaseInStructure*>(info.pNext); next;
                 next = next->pNext) {
                diagnostic::Trace("QUEUE_SUBMIT_PNEXT",
                                  "queue={:#x} submit_index={} structure_type={}",
                                  diagnostic::HandleBits(queue), i, static_cast<u32>(next->sType));
                if (next->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) {
                    const auto& timeline = *reinterpret_cast<const VkTimelineSemaphoreSubmitInfo*>(next);
                    diagnostic::Trace("QUEUE_SUBMIT_TIMELINE_INFO",
                                      "queue={:#x} submit_index={} wait_value_count={} signal_value_count={}",
                                      diagnostic::HandleBits(queue), i, timeline.waitSemaphoreValueCount,
                                      timeline.signalSemaphoreValueCount);
                    for (u32 value_index = 0; timeline.pWaitSemaphoreValues &&
                                               value_index < timeline.waitSemaphoreValueCount;
                         ++value_index) {
                        diagnostic::Trace("QUEUE_WAIT_TIMELINE_VALUE",
                                          "queue={:#x} submit_index={} index={} value={}",
                                          diagnostic::HandleBits(queue), i, value_index,
                                          timeline.pWaitSemaphoreValues[value_index]);
                    }
                    if (timeline.waitSemaphoreValueCount != 0 && !timeline.pWaitSemaphoreValues) {
                        diagnostic::Trace("QUEUE_WAIT_TIMELINE_VALUES_NULL",
                                          "queue={:#x} submit_index={} count={}",
                                          diagnostic::HandleBits(queue), i,
                                          timeline.waitSemaphoreValueCount);
                    }
                    for (u32 value_index = 0; timeline.pSignalSemaphoreValues &&
                                               value_index < timeline.signalSemaphoreValueCount;
                         ++value_index) {
                        diagnostic::Trace("QUEUE_SIGNAL_TIMELINE_VALUE",
                                          "queue={:#x} submit_index={} index={} value={}",
                                          diagnostic::HandleBits(queue), i, value_index,
                                          timeline.pSignalSemaphoreValues[value_index]);
                    }
                    if (timeline.signalSemaphoreValueCount != 0 && !timeline.pSignalSemaphoreValues) {
                        diagnostic::Trace("QUEUE_SIGNAL_TIMELINE_VALUES_NULL",
                                          "queue={:#x} submit_index={} count={}",
                                          diagnostic::HandleBits(queue), i,
                                          timeline.signalSemaphoreValueCount);
                    }
                }
            }
            if (info.pWaitSemaphores) {
                for (u32 j = 0; j < info.waitSemaphoreCount; ++j) {
                    diagnostic::Trace("QUEUE_WAIT_SEMAPHORE",
                                      "queue={:#x} submit_index={} index={} semaphore={:#x} dst_stage={:#x}",
                                      diagnostic::HandleBits(queue), i, j,
                                      diagnostic::HandleBits(info.pWaitSemaphores[j]),
                                      static_cast<u64>(info.pWaitDstStageMask ? info.pWaitDstStageMask[j] : 0));
                }
            } else if (info.waitSemaphoreCount != 0) {
                diagnostic::Trace("QUEUE_WAIT_SEMAPHORES_NULL",
                                  "queue={:#x} submit_index={} count={}",
                                  diagnostic::HandleBits(queue), i, info.waitSemaphoreCount);
            }
            if (info.pCommandBuffers) {
                for (u32 j = 0; j < info.commandBufferCount; ++j) {
                    diagnostic::Trace("QUEUE_COMMAND_BUFFER",
                                      "queue={:#x} submit_index={} index={} command_buffer={:#x}",
                                      diagnostic::HandleBits(queue), i, j,
                                      diagnostic::HandleBits(info.pCommandBuffers[j]));
                }
            } else if (info.commandBufferCount != 0) {
                diagnostic::Trace("QUEUE_COMMAND_BUFFERS_NULL",
                                  "queue={:#x} submit_index={} count={}",
                                  diagnostic::HandleBits(queue), i, info.commandBufferCount);
            }
            if (info.pSignalSemaphores) {
                for (u32 j = 0; j < info.signalSemaphoreCount; ++j) {
                    diagnostic::Trace("QUEUE_SIGNAL_SEMAPHORE",
                                      "queue={:#x} submit_index={} index={} semaphore={:#x}",
                                      diagnostic::HandleBits(queue), i, j,
                                      diagnostic::HandleBits(info.pSignalSemaphores[j]));
                }
            } else if (info.signalSemaphoreCount != 0) {
                diagnostic::Trace("QUEUE_SIGNAL_SEMAPHORES_NULL",
                                  "queue={:#x} submit_index={} count={}",
                                  diagnostic::HandleBits(queue), i, info.signalSemaphoreCount);
            }
        }
        const VkResult result = dld->vkQueueSubmit(queue, submit_infos.size(), submit_infos.data(), fence);
        diagnostic::Trace("QUEUE_SUBMIT_RESULT", "queue={:#x} api=vkQueueSubmit result={}",
                          diagnostic::HandleBits(queue), static_cast<s32>(result));
        return result;
    }

    /// Submits using VK_KHR_synchronization2 / Vulkan 1.3 vkQueueSubmit2.
    VkResult Submit2(Span<VkSubmitInfo2> submit_infos,
                     VkFence fence = VK_NULL_HANDLE) const noexcept {
        diagnostic::Trace("QUEUE_SUBMIT",
                          "queue={:#x} submit_count={} fence={:#x} api=vkQueueSubmit2",
                          diagnostic::HandleBits(queue), submit_infos.size(),
                          diagnostic::HandleBits(fence));
        for (u32 i = 0; i < submit_infos.size(); ++i) {
            const auto& info = submit_infos[i];
            diagnostic::Trace("QUEUE_SUBMIT_INFO",
                              "queue={:#x} submit_index={} waits={} command_buffers={} signals={}",
                              diagnostic::HandleBits(queue), i, info.waitSemaphoreInfoCount,
                              info.commandBufferInfoCount, info.signalSemaphoreInfoCount);
            for (u32 j = 0; info.pWaitSemaphoreInfos && j < info.waitSemaphoreInfoCount; ++j) {
                const auto& wait = info.pWaitSemaphoreInfos[j];
                diagnostic::Trace("QUEUE_WAIT_SEMAPHORE",
                                  "queue={:#x} submit_index={} index={} semaphore={:#x} value={} stage={:#x} device_index={}",
                                  diagnostic::HandleBits(queue), i, j,
                                  diagnostic::HandleBits(wait.semaphore), wait.value,
                                  static_cast<u64>(wait.stageMask), wait.deviceIndex);
            }
            for (u32 j = 0; info.pCommandBufferInfos && j < info.commandBufferInfoCount; ++j) {
                const auto& command = info.pCommandBufferInfos[j];
                diagnostic::Trace("QUEUE_COMMAND_BUFFER",
                                  "queue={:#x} submit_index={} index={} command_buffer={:#x} device_mask={:#x}",
                                  diagnostic::HandleBits(queue), i, j,
                                  diagnostic::HandleBits(command.commandBuffer), command.deviceMask);
            }
            for (u32 j = 0; info.pSignalSemaphoreInfos && j < info.signalSemaphoreInfoCount; ++j) {
                const auto& signal = info.pSignalSemaphoreInfos[j];
                diagnostic::Trace("QUEUE_SIGNAL_SEMAPHORE",
                                  "queue={:#x} submit_index={} index={} semaphore={:#x} value={} stage={:#x} device_index={}",
                                  diagnostic::HandleBits(queue), i, j,
                                  diagnostic::HandleBits(signal.semaphore), signal.value,
                                  static_cast<u64>(signal.stageMask), signal.deviceIndex);
            }
        }
        const VkResult result = dld->vkQueueSubmit2(queue, submit_infos.size(), submit_infos.data(), fence);
        diagnostic::Trace("QUEUE_SUBMIT_RESULT", "queue={:#x} api=vkQueueSubmit2 result={}",
                          diagnostic::HandleBits(queue), static_cast<s32>(result));
        return result;
    }

    VkResult BindSparse(Span<VkBindSparseInfo> bind_infos,
                        VkFence fence = VK_NULL_HANDLE) const noexcept {
        diagnostic::Trace("QUEUE_BIND_SPARSE", "queue={:#x} bind_info_count={} fence={:#x}",
                          diagnostic::HandleBits(queue), bind_infos.size(),
                          diagnostic::HandleBits(fence));
        const VkResult result = dld->vkQueueBindSparse(queue, bind_infos.size(), bind_infos.data(), fence);
        diagnostic::Trace("QUEUE_BIND_SPARSE_RESULT", "queue={:#x} result={}",
                          diagnostic::HandleBits(queue), static_cast<s32>(result));
        return result;
    }

    VkResult Present(const VkPresentInfoKHR& present_info) const noexcept {
        diagnostic::Trace("QUEUE_PRESENT",
                          "queue={:#x} waits={} swapchains={} result_array={}",
                          diagnostic::HandleBits(queue), present_info.waitSemaphoreCount,
                          present_info.swapchainCount, present_info.pResults != nullptr);
        if (present_info.pWaitSemaphores) {
            for (u32 i = 0; i < present_info.waitSemaphoreCount; ++i) {
                diagnostic::Trace("PRESENT_WAIT_SEMAPHORE", "queue={:#x} index={} semaphore={:#x}",
                                  diagnostic::HandleBits(queue), i,
                                  diagnostic::HandleBits(present_info.pWaitSemaphores[i]));
            }
        } else if (present_info.waitSemaphoreCount != 0) {
            diagnostic::Trace("PRESENT_WAIT_SEMAPHORES_NULL", "queue={:#x} count={}",
                              diagnostic::HandleBits(queue), present_info.waitSemaphoreCount);
        }
        for (u32 i = 0; present_info.pSwapchains && present_info.pImageIndices &&
                        i < present_info.swapchainCount; ++i) {
            diagnostic::Trace("PRESENT_SWAPCHAIN_IMAGE",
                              "queue={:#x} index={} swapchain={:#x} image_index={}",
                              diagnostic::HandleBits(queue), i,
                              diagnostic::HandleBits(present_info.pSwapchains[i]),
                              present_info.pImageIndices[i]);
        }
        const VkResult result = dld->vkQueuePresentKHR(queue, &present_info);
        diagnostic::Trace("QUEUE_PRESENT_RESULT", "queue={:#x} result={}",
                          diagnostic::HandleBits(queue), static_cast<s32>(result));
        if (present_info.pResults) {
            for (u32 i = 0; i < present_info.swapchainCount; ++i) {
                diagnostic::Trace("PRESENT_SWAPCHAIN_RESULT",
                                  "queue={:#x} index={} result={}",
                                  diagnostic::HandleBits(queue), i,
                                  static_cast<s32>(present_info.pResults[i]));
            }
        }
        return result;
    }

private:
    VkQueue queue = nullptr;
    const DeviceDispatch* dld = nullptr;
};

class BufferView : public Handle<VkBufferView, VkDevice, DeviceDispatch> {
    using Handle<VkBufferView, VkDevice, DeviceDispatch>::Handle;

public:
    /// Set object name.
    void SetObjectNameEXT(const char* name) const;
};

class ImageView : public Handle<VkImageView, VkDevice, DeviceDispatch> {
    using Handle<VkImageView, VkDevice, DeviceDispatch>::Handle;

public:
    /// Set object name.
    void SetObjectNameEXT(const char* name) const;
};

class DeviceMemory : public Handle<VkDeviceMemory, VkDevice, DeviceDispatch> {
    using Handle<VkDeviceMemory, VkDevice, DeviceDispatch>::Handle;

public:
    int GetMemoryFdKHR() const;

#ifdef _WIN32
    HANDLE GetMemoryWin32HandleKHR() const;
#endif

    /// Set object name.
    void SetObjectNameEXT(const char* name) const;

    u8* Map(VkDeviceSize offset, VkDeviceSize size) const {
        void* data;
        const VkResult result = dld->vkMapMemory(owner, handle, offset, size, 0, &data);
        diagnostic::Trace("MAP_MEMORY", "memory={:#x} offset={} size={} result={} data={:#x}",
                          diagnostic::HandleBits(handle), offset, size, static_cast<s32>(result),
                          result == VK_SUCCESS
                              ? static_cast<u64>(reinterpret_cast<std::uintptr_t>(data)) : 0);
        Check(result);
        return static_cast<u8*>(data);
    }

    void Unmap() const noexcept {
        diagnostic::Trace("UNMAP_MEMORY", "memory={:#x}", diagnostic::HandleBits(handle));
        dld->vkUnmapMemory(owner, handle);
    }
};

class Fence : public Handle<VkFence, VkDevice, DeviceDispatch> {
    using Handle<VkFence, VkDevice, DeviceDispatch>::Handle;

public:
    /// Set object name.
    void SetObjectNameEXT(const char* name) const;

    VkResult Wait(u64 timeout = (std::numeric_limits<u64>::max)()) const noexcept {
        const VkResult result = dld->vkWaitForFences(owner, 1, &handle, true, timeout);
        diagnostic::Trace("FENCE_WAIT", "fence={:#x} timeout_ns={} result={}",
                          diagnostic::HandleBits(handle), timeout, static_cast<s32>(result));
        return result;
    }

    VkResult GetStatus() const noexcept {
        const VkResult result = dld->vkGetFenceStatus(owner, handle);
        diagnostic::Trace("FENCE_STATUS", "fence={:#x} result={}",
                          diagnostic::HandleBits(handle), static_cast<s32>(result));
        return result;
    }

    void Reset() const {
        const VkResult result = dld->vkResetFences(owner, 1, &handle);
        diagnostic::Trace("FENCE_RESET", "fence={:#x} result={}",
                          diagnostic::HandleBits(handle), static_cast<s32>(result));
        Check(result);
    }
};

class Framebuffer : public Handle<VkFramebuffer, VkDevice, DeviceDispatch> {
    using Handle<VkFramebuffer, VkDevice, DeviceDispatch>::Handle;

public:
    /// Set object name.
    void SetObjectNameEXT(const char* name) const;
};

class DescriptorPool : public Handle<VkDescriptorPool, VkDevice, DeviceDispatch> {
    using Handle<VkDescriptorPool, VkDevice, DeviceDispatch>::Handle;

public:
    DescriptorSets Allocate(const VkDescriptorSetAllocateInfo& ai) const;

    /// Set object name.
    void SetObjectNameEXT(const char* name) const;
};

class CommandPool : public Handle<VkCommandPool, VkDevice, DeviceDispatch> {
    using Handle<VkCommandPool, VkDevice, DeviceDispatch>::Handle;

public:
    CommandBuffers Allocate(std::size_t num_buffers,
                            VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY) const;

    /// Set object name.
    void SetObjectNameEXT(const char* name) const;
};

class SwapchainKHR : public Handle<VkSwapchainKHR, VkDevice, DeviceDispatch> {
    using Handle<VkSwapchainKHR, VkDevice, DeviceDispatch>::Handle;

public:
    std::vector<VkImage> GetImages() const;
};

class Event : public Handle<VkEvent, VkDevice, DeviceDispatch> {
    using Handle<VkEvent, VkDevice, DeviceDispatch>::Handle;

public:
    /// Set object name.
    void SetObjectNameEXT(const char* name) const;

    VkResult GetStatus() const noexcept {
        const VkResult result = dld->vkGetEventStatus(owner, handle);
        diagnostic::Trace("EVENT_STATUS", "event={:#x} result={}",
                          diagnostic::HandleBits(handle), static_cast<s32>(result));
        return result;
    }
};

class ShaderModule : public Handle<VkShaderModule, VkDevice, DeviceDispatch> {
    using Handle<VkShaderModule, VkDevice, DeviceDispatch>::Handle;

public:
    /// Set object name.
    void SetObjectNameEXT(const char* name) const;
};

class PipelineCache : public Handle<VkPipelineCache, VkDevice, DeviceDispatch> {
    using Handle<VkPipelineCache, VkDevice, DeviceDispatch>::Handle;

public:
    /// Set object name.
    void SetObjectNameEXT(const char* name) const;

    VkResult Read(size_t* size, void* data) const noexcept {
        const VkResult result = dld->vkGetPipelineCacheData(owner, handle, size, data);
        diagnostic::Trace("PIPELINE_CACHE_READ",
                          "cache={:#x} result={} data_size={}",
                          diagnostic::HandleBits(handle), static_cast<s32>(result),
                          size ? *size : 0);
        return result;
    }
};

class Semaphore : public Handle<VkSemaphore, VkDevice, DeviceDispatch> {
    using Handle<VkSemaphore, VkDevice, DeviceDispatch>::Handle;

public:
    /// Set object name.
    void SetObjectNameEXT(const char* name) const;

    [[nodiscard]] u64 GetCounter() const {
        u64 value;
        const VkResult result = dld->vkGetSemaphoreCounterValue(owner, handle, &value);
        diagnostic::Trace("TIMELINE_SEMAPHORE_COUNTER",
                          "semaphore={:#x} result={} value={}",
                          diagnostic::HandleBits(handle), static_cast<s32>(result),
                          result == VK_SUCCESS ? value : 0);
        Check(result);
        return value;
    }

    /**
     * Waits for a timeline semaphore on the host.
     *
     * @param value   Value to wait
     * @param timeout Time in nanoseconds to timeout
     * @return        True on successful wait, false on timeout
     */
    bool Wait(u64 value, u64 timeout = (std::numeric_limits<u64>::max)()) const {
        const VkSemaphoreWaitInfo wait_info{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .pNext = nullptr,
            .flags = 0,
            .semaphoreCount = 1,
            .pSemaphores = &handle,
            .pValues = &value,
        };
        const VkResult result = dld->vkWaitSemaphores(owner, &wait_info, timeout);
        diagnostic::Trace("TIMELINE_SEMAPHORE_WAIT",
                          "semaphore={:#x} value={} timeout_ns={} result={}",
                          diagnostic::HandleBits(handle), value, timeout,
                          static_cast<s32>(result));
        switch (result) {
        case VK_SUCCESS:
            return true;
        case VK_TIMEOUT:
            return false;
        default:
            throw Exception(result);
        }
    }
};

class Device : public Handle<VkDevice, NoOwner, DeviceDispatch> {
    using Handle<VkDevice, NoOwner, DeviceDispatch>::Handle;

public:
    [[nodiscard]] static Device Create(VkPhysicalDevice physical_device,
                                       Span<VkDeviceQueueCreateInfo> queues_ci,
                                       Span<const char*> enabled_extensions, const void* next,
                                       DeviceDispatch& dispatch);

    [[nodiscard]] Queue GetQueue(u32 family_index) const noexcept;

    [[nodiscard]] BufferView CreateBufferView(const VkBufferViewCreateInfo& ci) const;

    [[nodiscard]] ImageView CreateImageView(const VkImageViewCreateInfo& ci) const;

    [[nodiscard]] Semaphore CreateSemaphore() const;

    [[nodiscard]] Semaphore CreateSemaphore(const VkSemaphoreCreateInfo& ci) const;

    [[nodiscard]] Fence CreateFence(const VkFenceCreateInfo& ci) const;

    [[nodiscard]] DescriptorPool CreateDescriptorPool(const VkDescriptorPoolCreateInfo& ci) const;

    [[nodiscard]] RenderPass CreateRenderPass(const VkRenderPassCreateInfo& ci) const;

    [[nodiscard]] RenderPass CreateRenderPass2(const VkRenderPassCreateInfo2& ci) const;

    [[nodiscard]] DescriptorSetLayout CreateDescriptorSetLayout(
        const VkDescriptorSetLayoutCreateInfo& ci) const;

    [[nodiscard]] PipelineCache CreatePipelineCache(const VkPipelineCacheCreateInfo& ci) const;

    [[nodiscard]] PipelineLayout CreatePipelineLayout(const VkPipelineLayoutCreateInfo& ci) const;

    [[nodiscard]] Pipeline CreateGraphicsPipeline(const VkGraphicsPipelineCreateInfo& ci,
                                                  VkPipelineCache cache = {}) const;

    [[nodiscard]] Pipeline CreateComputePipeline(const VkComputePipelineCreateInfo& ci,
                                                 VkPipelineCache cache = {}) const;

    [[nodiscard]] Sampler CreateSampler(const VkSamplerCreateInfo& ci) const;

    [[nodiscard]] Framebuffer CreateFramebuffer(const VkFramebufferCreateInfo& ci) const;

    [[nodiscard]] CommandPool CreateCommandPool(const VkCommandPoolCreateInfo& ci) const;

    [[nodiscard]] DescriptorUpdateTemplate CreateDescriptorUpdateTemplate(
        const VkDescriptorUpdateTemplateCreateInfo& ci) const;

    [[nodiscard]] QueryPool CreateQueryPool(const VkQueryPoolCreateInfo& ci) const;

    [[nodiscard]] ShaderModule CreateShaderModule(const VkShaderModuleCreateInfo& ci) const;

    [[nodiscard]] Event CreateEvent() const;

    [[nodiscard]] SwapchainKHR CreateSwapchainKHR(const VkSwapchainCreateInfoKHR& ci) const;

    [[nodiscard]] DeviceMemory TryAllocateMemory(const VkMemoryAllocateInfo& ai) const noexcept;

    [[nodiscard]] DeviceMemory AllocateMemory(const VkMemoryAllocateInfo& ai) const;

    VkMemoryRequirements GetBufferMemoryRequirements(VkBuffer buffer,
                                                     void* pnext = nullptr) const noexcept;

    VkMemoryRequirements GetImageMemoryRequirements(VkImage image) const noexcept;

    std::vector<VkSparseImageMemoryRequirements> GetImageSparseMemoryRequirements(
        VkImage image) const;

    std::vector<VkPipelineExecutablePropertiesKHR> GetPipelineExecutablePropertiesKHR(
        VkPipeline pipeline) const;

    std::vector<VkPipelineExecutableStatisticKHR> GetPipelineExecutableStatisticsKHR(
        VkPipeline pipeline, u32 executable_index) const;

    void UpdateDescriptorSets(Span<VkWriteDescriptorSet> writes,
                              Span<VkCopyDescriptorSet> copies) const noexcept;

    void UpdateDescriptorSet(VkDescriptorSet set, VkDescriptorUpdateTemplate update_template,
                             const void* data) const noexcept {
        diagnostic::Trace("UPDATE_DESCRIPTOR_SET_TEMPLATE",
                          "device={:#x} set={:#x} update_template={:#x} data={:#x}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(set),
                          diagnostic::HandleBits(update_template),
                          static_cast<u64>(reinterpret_cast<std::uintptr_t>(data)));
        dld->vkUpdateDescriptorSetWithTemplate(handle, set, update_template, data);
    }

    [[nodiscard]] VkDeviceAddress GetBufferDeviceAddress(VkBuffer buffer) const noexcept {
        const VkBufferDeviceAddressInfo info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .pNext = nullptr,
            .buffer = buffer,
        };
        return dld->vkGetBufferDeviceAddress(handle, &info);
    }

    [[nodiscard]] VkDeviceSize GetDescriptorSetLayoutSizeEXT(
        VkDescriptorSetLayout layout) const noexcept {
        VkDeviceSize size{};
        dld->vkGetDescriptorSetLayoutSizeEXT(handle, layout, &size);
        return size;
    }

    [[nodiscard]] VkDeviceSize GetDescriptorSetLayoutBindingOffsetEXT(
        VkDescriptorSetLayout layout, u32 binding) const noexcept {
        VkDeviceSize offset{};
        dld->vkGetDescriptorSetLayoutBindingOffsetEXT(handle, layout, binding, &offset);
        return offset;
    }

    void GetDescriptorEXT(const VkDescriptorGetInfoEXT& info, size_t size,
                          void* descriptor) const noexcept {
        dld->vkGetDescriptorEXT(handle, &info, size, descriptor);
    }

    VkResult AcquireNextImageKHR(VkSwapchainKHR swapchain, u64 timeout, VkSemaphore semaphore,
                                 VkFence fence, u32* image_index) const noexcept {
        const VkResult result = dld->vkAcquireNextImageKHR(handle, swapchain, timeout, semaphore,
                                                           fence, image_index);
        diagnostic::Trace("ACQUIRE_NEXT_IMAGE",
                          "device={:#x} swapchain={:#x} timeout_ns={} semaphore={:#x} fence={:#x} result={} image_index={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(swapchain), timeout,
                          diagnostic::HandleBits(semaphore), diagnostic::HandleBits(fence),
                          static_cast<s32>(result),
                          image_index && (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR)
                              ? *image_index : UINT32_MAX);
        return result;
    }

    VkResult WaitIdle() const noexcept {
        diagnostic::Trace("DEVICE_WAIT_IDLE_BEGIN", "device={:#x}",
                          diagnostic::HandleBits(handle));
        const VkResult result = dld->vkDeviceWaitIdle(handle);
        diagnostic::Trace("DEVICE_WAIT_IDLE_END", "device={:#x} result={}",
                          diagnostic::HandleBits(handle), static_cast<s32>(result));
        return result;
    }

    void ResetQueryPool(VkQueryPool query_pool, u32 first, u32 count) const noexcept {
        diagnostic::Trace("QUERY_POOL_RESET",
                          "device={:#x} query_pool={:#x} first={} count={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(query_pool), first,
                          count);
        dld->vkResetQueryPool(handle, query_pool, first, count);
    }

    VkResult GetQueryResults(VkQueryPool query_pool, u32 first, u32 count, std::size_t data_size,
                             void* data, VkDeviceSize stride,
                             VkQueryResultFlags flags) const noexcept {
        const VkResult result = dld->vkGetQueryPoolResults(handle, query_pool, first, count,
                                                           data_size, data, stride, flags);
        diagnostic::Trace("QUERY_POOL_RESULTS",
                          "device={:#x} query_pool={:#x} first={} count={} data_size={} stride={} flags={:#x} result={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(query_pool), first,
                          count, data_size, stride, static_cast<u32>(flags),
                          static_cast<s32>(result));
        return result;
    }
};

class PhysicalDevice {
public:
    constexpr PhysicalDevice() noexcept = default;

    constexpr PhysicalDevice(VkPhysicalDevice physical_device_,
                             const InstanceDispatch& dld_) noexcept
        : physical_device{physical_device_}, dld{&dld_} {}

    constexpr operator VkPhysicalDevice() const noexcept {
        return physical_device;
    }

    VkPhysicalDeviceProperties GetProperties() const noexcept;

    void GetProperties2(VkPhysicalDeviceProperties2&) const noexcept;

    VkPhysicalDeviceFeatures GetFeatures() const noexcept;

    void GetFeatures2(VkPhysicalDeviceFeatures2&) const noexcept;

    VkFormatProperties GetFormatProperties(VkFormat) const noexcept;

    VkFormatProperties3 GetFormatProperties3(VkFormat) const noexcept;

    std::vector<VkExtensionProperties> EnumerateDeviceExtensionProperties() const;

    std::vector<VkQueueFamilyProperties> GetQueueFamilyProperties() const;

    std::vector<VkPhysicalDeviceToolProperties> GetPhysicalDeviceToolProperties() const;

    /// Matrix shapes and types of VK_KHR_cooperative_matrix; empty when the driver lacks it.
    std::vector<VkCooperativeMatrixPropertiesKHR> GetCooperativeMatrixPropertiesKHR() const;

    bool GetSurfaceSupportKHR(u32 queue_family_index, VkSurfaceKHR) const;

    VkSurfaceCapabilitiesKHR GetSurfaceCapabilitiesKHR(VkSurfaceKHR) const;

    std::vector<VkSurfaceFormatKHR> GetSurfaceFormatsKHR(VkSurfaceKHR) const;

    std::vector<VkPresentModeKHR> GetSurfacePresentModesKHR(VkSurfaceKHR) const;

    VkPhysicalDeviceMemoryProperties2 GetMemoryProperties(
        void* next_structures = nullptr) const noexcept;

private:
    VkPhysicalDevice physical_device = nullptr;
    const InstanceDispatch* dld = nullptr;
};

class CommandBuffer {
public:
    CommandBuffer() noexcept = default;

    explicit CommandBuffer(VkCommandBuffer handle_, const DeviceDispatch& dld_) noexcept
        : handle{handle_}, dld{&dld_} {}

    const VkCommandBuffer* address() const noexcept {
        return &handle;
    }

    VkCommandBuffer operator*() const noexcept {
        return handle;
    }
    void Begin(const VkCommandBufferBeginInfo& begin_info) const {
        diagnostic::Trace("COMMAND_BUFFER_BEGIN",
                          "command_buffer={:#x} flags={:#x} inheritance={:#x}",
                          diagnostic::HandleBits(handle), static_cast<u32>(begin_info.flags),
                          diagnostic::HandleBits(begin_info.pInheritanceInfo
                              ? begin_info.pInheritanceInfo->renderPass : VK_NULL_HANDLE));
        const VkResult result = dld->vkBeginCommandBuffer(handle, &begin_info);
        diagnostic::Trace("COMMAND_BUFFER_BEGIN_RESULT", "command_buffer={:#x} result={}",
                          diagnostic::HandleBits(handle), static_cast<s32>(result));
        Check(result);
    }

    void End() const {
        const VkResult result = dld->vkEndCommandBuffer(handle);
        diagnostic::Trace("COMMAND_BUFFER_END", "command_buffer={:#x} result={}",
                          diagnostic::HandleBits(handle), static_cast<s32>(result));
        Check(result);
    }

    void BeginRenderPass(const VkRenderPassBeginInfo& renderpass_bi,
                         VkSubpassContents contents) const noexcept {
        diagnostic::Trace("RENDER_PASS_BEGIN",
                          "cmd={:#x} renderpass={:#x} framebuffer={:#x} area=({},{} {}x{}) "
                          "clear_count={} contents={}",
                          diagnostic::HandleBits(handle),
                          diagnostic::HandleBits(renderpass_bi.renderPass),
                          diagnostic::HandleBits(renderpass_bi.framebuffer),
                          renderpass_bi.renderArea.offset.x, renderpass_bi.renderArea.offset.y,
                          renderpass_bi.renderArea.extent.width, renderpass_bi.renderArea.extent.height,
                          renderpass_bi.clearValueCount, static_cast<u32>(contents));
        dld->vkCmdBeginRenderPass(handle, &renderpass_bi, contents);
    }

    void EndRenderPass() const noexcept {
        diagnostic::Trace("RENDER_PASS_END", "cmd={:#x}", diagnostic::HandleBits(handle));
        dld->vkCmdEndRenderPass(handle);
    }

    void BeginQuery(VkQueryPool query_pool, u32 query, VkQueryControlFlags flags) const noexcept {
        diagnostic::Trace("COMMAND_QUERY_BEGIN",
                          "cmd={:#x} query_pool={:#x} query={} flags={:#x}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(query_pool), query,
                          static_cast<u32>(flags));
        dld->vkCmdBeginQuery(handle, query_pool, query, flags);
    }

    void EndQuery(VkQueryPool query_pool, u32 query) const noexcept {
        diagnostic::Trace("COMMAND_QUERY_END",
                          "cmd={:#x} query_pool={:#x} query={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(query_pool), query);
        dld->vkCmdEndQuery(handle, query_pool, query);
    }

    void ResetQueryPool(VkQueryPool query_pool, u32 first, u32 count) const noexcept {
        diagnostic::Trace("COMMAND_QUERY_RESET",
                          "cmd={:#x} query_pool={:#x} first={} count={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(query_pool), first,
                          count);
        dld->vkCmdResetQueryPool(handle, query_pool, first, count);
    }

    void WriteTimestamp(VkPipelineStageFlagBits stage, VkQueryPool query_pool,
                        u32 query) const noexcept {
        diagnostic::Trace("COMMAND_TIMESTAMP",
                          "cmd={:#x} stage={:#x} query_pool={:#x} query={}",
                          diagnostic::HandleBits(handle), static_cast<u32>(stage),
                          diagnostic::HandleBits(query_pool), query);
        dld->vkCmdWriteTimestamp(handle, stage, query_pool, query);
    }

    void BindDescriptorSets(VkPipelineBindPoint bind_point, VkPipelineLayout layout, u32 first,
                            Span<VkDescriptorSet> sets, Span<u32> dynamic_offsets) const noexcept {
        diagnostic::Trace("DESCRIPTOR_BIND",
                          "cmd={:#x} bind_point={} layout={:#x} first_set={} set_count={} "
                          "dynamic_offset_count={}",
                          diagnostic::HandleBits(handle), static_cast<u32>(bind_point),
                          diagnostic::HandleBits(layout), first, sets.size(), dynamic_offsets.size());
        for (u32 i = 0; i < sets.size(); ++i) {
            diagnostic::Trace("DESCRIPTOR_SET",
                              "cmd={:#x} set_index={} set={:#x}",
                              diagnostic::HandleBits(handle), first + i,
                              diagnostic::HandleBits(sets[i]));
        }
        for (u32 i = 0; i < dynamic_offsets.size(); ++i) {
            diagnostic::Trace("DESCRIPTOR_DYNAMIC_OFFSET",
                              "cmd={:#x} offset_index={} value={}",
                              diagnostic::HandleBits(handle), i, dynamic_offsets[i]);
        }
        dld->vkCmdBindDescriptorSets(handle, bind_point, layout, first, sets.size(), sets.data(),
                                     dynamic_offsets.size(), dynamic_offsets.data());
    }

    void PushDescriptorSetWithTemplateKHR(VkDescriptorUpdateTemplate update_template,
                                          VkPipelineLayout layout, u32 set,
                                          const void* data) const noexcept {
        dld->vkCmdPushDescriptorSetWithTemplateKHR(handle, update_template, layout, set, data);
    }

    void BindPipeline(VkPipelineBindPoint bind_point, VkPipeline pipeline) const noexcept {
        diagnostic::Trace("PIPELINE_BIND",
                          "cmd={:#x} bind_point={} pipeline={:#x}",
                          diagnostic::HandleBits(handle), static_cast<u32>(bind_point),
                          diagnostic::HandleBits(pipeline));
        dld->vkCmdBindPipeline(handle, bind_point, pipeline);
    }

    void BindIndexBuffer(VkBuffer buffer, VkDeviceSize offset,
                         VkIndexType index_type) const noexcept {
        diagnostic::Trace("BIND_INDEX_BUFFER",
                          "cmd={:#x} buffer={:#x} offset={} index_type={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(buffer), offset,
                          static_cast<u32>(index_type));
        dld->vkCmdBindIndexBuffer(handle, buffer, offset, index_type);
    }

    void BindVertexBuffers(u32 first, u32 count, const VkBuffer* buffers,
                           const VkDeviceSize* offsets) const noexcept {
        diagnostic::Trace("BIND_VERTEX_BUFFERS",
                          "cmd={:#x} first_binding={} count={}",
                          diagnostic::HandleBits(handle), first, count);
        for (u32 i = 0; i < count; ++i) {
            diagnostic::Trace("BIND_VERTEX_BUFFER",
                              "cmd={:#x} binding={} buffer={:#x} offset={}",
                              diagnostic::HandleBits(handle), first + i,
                              diagnostic::HandleBits(buffers[i]), offsets ? offsets[i] : 0);
        }
        dld->vkCmdBindVertexBuffers(handle, first, count, buffers, offsets);
    }

    void BindVertexBuffer(u32 binding, VkBuffer buffer, VkDeviceSize offset) const noexcept {
        BindVertexBuffers(binding, 1, &buffer, &offset);
    }

    void Draw(u32 vertex_count, u32 instance_count, u32 first_vertex,
              u32 first_instance) const noexcept {
        diagnostic::Trace("DRAW", "cmd={:#x} vertices={} instances={} first_vertex={} first_instance={}",
                          diagnostic::HandleBits(handle), vertex_count, instance_count,
                          first_vertex, first_instance);
        dld->vkCmdDraw(handle, vertex_count, instance_count, first_vertex, first_instance);
    }

    void DrawIndexed(u32 index_count, u32 instance_count, u32 first_index, u32 vertex_offset,
                     u32 first_instance) const noexcept {
        diagnostic::Trace("DRAW_INDEXED",
                          "cmd={:#x} indices={} instances={} first_index={} vertex_offset={} first_instance={}",
                          diagnostic::HandleBits(handle), index_count, instance_count, first_index,
                          vertex_offset, first_instance);
        dld->vkCmdDrawIndexed(handle, index_count, instance_count, first_index, vertex_offset,
                              first_instance);
    }

    void DrawIndirect(VkBuffer src_buffer, VkDeviceSize src_offset, u32 draw_count,
                      u32 stride) const noexcept {
        diagnostic::Trace("DRAW_INDIRECT",
                          "cmd={:#x} buffer={:#x} offset={} draw_count={} stride={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(src_buffer),
                          src_offset, draw_count, stride);
        dld->vkCmdDrawIndirect(handle, src_buffer, src_offset, draw_count, stride);
    }

    void DrawIndexedIndirect(VkBuffer src_buffer, VkDeviceSize src_offset, u32 draw_count,
                             u32 stride) const noexcept {
        diagnostic::Trace("DRAW_INDEXED_INDIRECT",
                          "cmd={:#x} buffer={:#x} offset={} draw_count={} stride={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(src_buffer),
                          src_offset, draw_count, stride);
        dld->vkCmdDrawIndexedIndirect(handle, src_buffer, src_offset, draw_count, stride);
    }

    void DrawMeshTasksEXT(u32 group_count_x, u32 group_count_y, u32 group_count_z) const noexcept {
        diagnostic::Trace("DRAW_MESH_TASKS",
                          "cmd={:#x} groups=({}, {}, {})", diagnostic::HandleBits(handle),
                          group_count_x, group_count_y, group_count_z);
        dld->vkCmdDrawMeshTasksEXT(handle, group_count_x, group_count_y, group_count_z);
    }

    void DrawIndirectCount(VkBuffer src_buffer, VkDeviceSize src_offset, VkBuffer count_buffer,
                           VkDeviceSize count_offset, u32 draw_count, u32 stride) const noexcept {
        dld->vkCmdDrawIndirectCount(handle, src_buffer, src_offset, count_buffer, count_offset,
                                    draw_count, stride);
    }

    void DrawIndexedIndirectCount(VkBuffer src_buffer, VkDeviceSize src_offset,
                                  VkBuffer count_buffer, VkDeviceSize count_offset, u32 draw_count,
                                  u32 stride) const noexcept {
        dld->vkCmdDrawIndexedIndirectCount(handle, src_buffer, src_offset, count_buffer,
                                           count_offset, draw_count, stride);
    }

    void DrawIndirectByteCountEXT(u32 instance_count, u32 first_instance, VkBuffer counter_buffer,
                                  VkDeviceSize counter_buffer_offset, u32 counter_offset,
                                  u32 stride) {
        dld->vkCmdDrawIndirectByteCountEXT(handle, instance_count, first_instance, counter_buffer,
                                           counter_buffer_offset, counter_offset, stride);
    }

    void ClearAttachments(Span<VkClearAttachment> attachments,
                          Span<VkClearRect> rects) const noexcept {
        diagnostic::Trace("CLEAR_ATTACHMENTS",
                          "cmd={:#x} attachment_count={} rect_count={}",
                          diagnostic::HandleBits(handle), attachments.size(), rects.size());
        for (u32 i = 0; i < attachments.size(); ++i) {
            const auto& a = attachments[i];
            if (a.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) {
                diagnostic::Trace("CLEAR_ATTACHMENT",
                                  "cmd={:#x} index={} aspect={:#x} color_attachment={} color_bits=({:#x},{:#x},{:#x},{:#x})",
                                  diagnostic::HandleBits(handle), i,
                                  static_cast<u32>(a.aspectMask), a.colorAttachment,
                                  a.clearValue.color.uint32[0], a.clearValue.color.uint32[1],
                                  a.clearValue.color.uint32[2], a.clearValue.color.uint32[3]);
            } else {
                diagnostic::Trace("CLEAR_ATTACHMENT",
                                  "cmd={:#x} index={} aspect={:#x} depth={} stencil={}",
                                  diagnostic::HandleBits(handle), i,
                                  static_cast<u32>(a.aspectMask), a.clearValue.depthStencil.depth,
                                  a.clearValue.depthStencil.stencil);
            }
        }
        for (u32 i = 0; i < rects.size(); ++i) {
            const auto& r = rects[i];
            diagnostic::Trace("CLEAR_RECT",
                              "cmd={:#x} index={} rect=({},{} {}x{}) base_layer={} layer_count={}",
                              diagnostic::HandleBits(handle), i, r.rect.offset.x, r.rect.offset.y,
                              r.rect.extent.width, r.rect.extent.height, r.baseArrayLayer,
                              r.layerCount);
        }
        dld->vkCmdClearAttachments(handle, attachments.size(), attachments.data(), rects.size(),
                                   rects.data());
    }

    void ClearColorImage(VkImage image, VkImageLayout layout, VkClearColorValue color,
                         Span<VkImageSubresourceRange> ranges) {
        diagnostic::Trace("CLEAR_COLOR_IMAGE",
                          "cmd={:#x} image={:#x} layout={} range_count={} rgba_bits=({:#x},{:#x},{:#x},{:#x})",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(image),
                          static_cast<u32>(layout), ranges.size(), color.uint32[0], color.uint32[1],
                          color.uint32[2], color.uint32[3]);
        for (u32 i = 0; i < ranges.size(); ++i) {
            const auto& range = ranges[i];
            diagnostic::Trace("CLEAR_COLOR_RANGE",
                              "cmd={:#x} image={:#x} index={} aspect={:#x} base_mip={} mip_count={} "
                              "base_layer={} layer_count={}", diagnostic::HandleBits(handle),
                              diagnostic::HandleBits(image), i, static_cast<u32>(range.aspectMask),
                              range.baseMipLevel, range.levelCount, range.baseArrayLayer, range.layerCount);
        }
        dld->vkCmdClearColorImage(handle, image, layout, &color, ranges.size(), ranges.data());
    }

    void BlitImage(VkImage src_image, VkImageLayout src_layout, VkImage dst_image,
                   VkImageLayout dst_layout, Span<VkImageBlit> regions,
                   VkFilter filter) const noexcept {
        diagnostic::Trace("BLIT_IMAGE",
                          "cmd={:#x} src={:#x} src_layout={} dst={:#x} dst_layout={} regions={} filter={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(src_image),
                          static_cast<u32>(src_layout), diagnostic::HandleBits(dst_image),
                          static_cast<u32>(dst_layout), regions.size(), static_cast<u32>(filter));
        for (u32 i = 0; i < regions.size(); ++i) {
            const auto& region = regions[i];
            diagnostic::Trace("BLIT_REGION",
                              "cmd={:#x} index={} src_aspect={:#x} src_mip={} src_layers=({}, {}) "
                              "src0=({}, {}, {}) src1=({}, {}, {}) dst_aspect={:#x} dst_mip={} "
                              "dst_layers=({}, {}) dst0=({}, {}, {}) dst1=({}, {}, {})",
                              diagnostic::HandleBits(handle), i,
                              static_cast<u32>(region.srcSubresource.aspectMask), region.srcSubresource.mipLevel,
                              region.srcSubresource.baseArrayLayer, region.srcSubresource.layerCount,
                              region.srcOffsets[0].x, region.srcOffsets[0].y, region.srcOffsets[0].z,
                              region.srcOffsets[1].x, region.srcOffsets[1].y, region.srcOffsets[1].z,
                              static_cast<u32>(region.dstSubresource.aspectMask), region.dstSubresource.mipLevel,
                              region.dstSubresource.baseArrayLayer, region.dstSubresource.layerCount,
                              region.dstOffsets[0].x, region.dstOffsets[0].y, region.dstOffsets[0].z,
                              region.dstOffsets[1].x, region.dstOffsets[1].y, region.dstOffsets[1].z);
        }
        dld->vkCmdBlitImage(handle, src_image, src_layout, dst_image, dst_layout, regions.size(),
                            regions.data(), filter);
    }

    void ResolveImage(VkImage src_image, VkImageLayout src_layout, VkImage dst_image,
                      VkImageLayout dst_layout, Span<VkImageResolve> regions) {
        diagnostic::Trace("RESOLVE_IMAGE",
                          "cmd={:#x} src={:#x} src_layout={} dst={:#x} dst_layout={} regions={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(src_image),
                          static_cast<u32>(src_layout), diagnostic::HandleBits(dst_image),
                          static_cast<u32>(dst_layout), regions.size());
        for (u32 i = 0; i < regions.size(); ++i) {
            const auto& r = regions[i];
            diagnostic::Trace("RESOLVE_REGION",
                              "cmd={:#x} index={} src_aspect={:#x} src_mip={} src_layer={} "
                              "src_offset=({}, {}, {}) extent=({}, {}, {}) dst_aspect={:#x} "
                              "dst_mip={} dst_layer={} dst_offset=({}, {}, {})",
                              diagnostic::HandleBits(handle), i, static_cast<u32>(r.srcSubresource.aspectMask),
                              r.srcSubresource.mipLevel, r.srcSubresource.baseArrayLayer,
                              r.srcOffset.x, r.srcOffset.y, r.srcOffset.z, r.extent.width,
                              r.extent.height, r.extent.depth, static_cast<u32>(r.dstSubresource.aspectMask),
                              r.dstSubresource.mipLevel, r.dstSubresource.baseArrayLayer,
                              r.dstOffset.x, r.dstOffset.y, r.dstOffset.z);
        }
        dld->vkCmdResolveImage(handle, src_image, src_layout, dst_image, dst_layout, regions.size(),
                               regions.data());
    }

    void Dispatch(u32 x, u32 y, u32 z) const noexcept {
        diagnostic::Trace("DISPATCH", "cmd={:#x} groups=({}, {}, {})",
                          diagnostic::HandleBits(handle), x, y, z);
        dld->vkCmdDispatch(handle, x, y, z);
    }

    void DispatchIndirect(VkBuffer indirect_buffer, VkDeviceSize offset) const noexcept {
        diagnostic::Trace("DISPATCH_INDIRECT", "cmd={:#x} buffer={:#x} offset={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(indirect_buffer), offset);
        dld->vkCmdDispatchIndirect(handle, indirect_buffer, offset);
    }

    void PipelineBarrier(VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
                         VkDependencyFlags dependency_flags, Span<VkMemoryBarrier> memory_barriers,
                         Span<VkBufferMemoryBarrier> buffer_barriers,
                         Span<VkImageMemoryBarrier> image_barriers) const noexcept {
        static constexpr u32 MaxBarriers = 16;
        const bool use_sync2 = dld->vkCmdPipelineBarrier2 && memory_barriers.size() <= MaxBarriers &&
            buffer_barriers.size() <= MaxBarriers && image_barriers.size() <= MaxBarriers;
        diagnostic::Trace("PIPELINE_BARRIER",
                          "cmd={:#x} api={} src_stage={:#x} dst_stage={:#x} dep_flags={:#x} "
                          "memory_count={} buffer_count={} image_count={}",
                          diagnostic::HandleBits(handle), use_sync2 ? "vkCmdPipelineBarrier2" : "vkCmdPipelineBarrier",
                          static_cast<u64>(src_stage_mask), static_cast<u64>(dst_stage_mask),
                          static_cast<u32>(dependency_flags), memory_barriers.size(),
                          buffer_barriers.size(), image_barriers.size());
        for (u32 i = 0; i < memory_barriers.size(); ++i) {
            const auto& barrier = memory_barriers[i];
            diagnostic::Trace("BARRIER_MEMORY",
                              "cmd={:#x} index={} src_access={:#x} dst_access={:#x}",
                              diagnostic::HandleBits(handle), i,
                              static_cast<u64>(barrier.srcAccessMask),
                              static_cast<u64>(barrier.dstAccessMask));
        }
        for (u32 i = 0; i < buffer_barriers.size(); ++i) {
            const auto& barrier = buffer_barriers[i];
            diagnostic::Trace("BARRIER_BUFFER",
                              "cmd={:#x} index={} buffer={:#x} offset={} size={} src_access={:#x} "
                              "dst_access={:#x} src_queue={} dst_queue={}",
                              diagnostic::HandleBits(handle), i, diagnostic::HandleBits(barrier.buffer),
                              barrier.offset, barrier.size, static_cast<u64>(barrier.srcAccessMask),
                              static_cast<u64>(barrier.dstAccessMask), barrier.srcQueueFamilyIndex,
                              barrier.dstQueueFamilyIndex);
        }
        for (u32 i = 0; i < image_barriers.size(); ++i) {
            const auto& barrier = image_barriers[i];
            const auto& range = barrier.subresourceRange;
            diagnostic::Trace("BARRIER_IMAGE",
                              "cmd={:#x} index={} image={:#x} old_layout={} new_layout={} "
                              "src_access={:#x} dst_access={:#x} src_queue={} dst_queue={} "
                              "aspect={:#x} base_mip={} mip_count={} base_layer={} layer_count={}",
                              diagnostic::HandleBits(handle), i, diagnostic::HandleBits(barrier.image),
                              static_cast<u32>(barrier.oldLayout), static_cast<u32>(barrier.newLayout),
                              static_cast<u64>(barrier.srcAccessMask), static_cast<u64>(barrier.dstAccessMask),
                              barrier.srcQueueFamilyIndex, barrier.dstQueueFamilyIndex,
                              static_cast<u32>(range.aspectMask), range.baseMipLevel,
                              range.levelCount, range.baseArrayLayer, range.layerCount);
        }
        if (use_sync2 &&
            buffer_barriers.size() <= MaxBarriers && image_barriers.size() <= MaxBarriers) {
            const auto src_stage_mask2 = static_cast<VkPipelineStageFlags2>(src_stage_mask);
            const auto dst_stage_mask2 = static_cast<VkPipelineStageFlags2>(dst_stage_mask);

            std::array<VkMemoryBarrier2, MaxBarriers> memory_barriers2;
            for (u32 i = 0; i < memory_barriers.size(); ++i) {
                memory_barriers2[i] = VkMemoryBarrier2{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                    .pNext = nullptr,
                    .srcStageMask = src_stage_mask2,
                    .srcAccessMask = static_cast<VkAccessFlags2>(memory_barriers[i].srcAccessMask),
                    .dstStageMask = dst_stage_mask2,
                    .dstAccessMask = static_cast<VkAccessFlags2>(memory_barriers[i].dstAccessMask),
                };
            }
            std::array<VkBufferMemoryBarrier2, MaxBarriers> buffer_barriers2;
            for (u32 i = 0; i < buffer_barriers.size(); ++i) {
                const auto& barrier = buffer_barriers[i];
                buffer_barriers2[i] = VkBufferMemoryBarrier2{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                    .pNext = nullptr,
                    .srcStageMask = src_stage_mask2,
                    .srcAccessMask = static_cast<VkAccessFlags2>(barrier.srcAccessMask),
                    .dstStageMask = dst_stage_mask2,
                    .dstAccessMask = static_cast<VkAccessFlags2>(barrier.dstAccessMask),
                    .srcQueueFamilyIndex = barrier.srcQueueFamilyIndex,
                    .dstQueueFamilyIndex = barrier.dstQueueFamilyIndex,
                    .buffer = barrier.buffer,
                    .offset = barrier.offset,
                    .size = barrier.size,
                };
            }
            std::array<VkImageMemoryBarrier2, MaxBarriers> image_barriers2;
            for (u32 i = 0; i < image_barriers.size(); ++i) {
                const auto& barrier = image_barriers[i];
                image_barriers2[i] = VkImageMemoryBarrier2{
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                    .pNext = nullptr,
                    .srcStageMask = src_stage_mask2,
                    .srcAccessMask = static_cast<VkAccessFlags2>(barrier.srcAccessMask),
                    .dstStageMask = dst_stage_mask2,
                    .dstAccessMask = static_cast<VkAccessFlags2>(barrier.dstAccessMask),
                    .oldLayout = barrier.oldLayout,
                    .newLayout = barrier.newLayout,
                    .srcQueueFamilyIndex = barrier.srcQueueFamilyIndex,
                    .dstQueueFamilyIndex = barrier.dstQueueFamilyIndex,
                    .image = barrier.image,
                    .subresourceRange = barrier.subresourceRange,
                };
            }
            const VkDependencyInfo dependency_info{
                .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                .pNext = nullptr,
                .dependencyFlags = dependency_flags,
                .memoryBarrierCount = memory_barriers.size(),
                .pMemoryBarriers = memory_barriers2.data(),
                .bufferMemoryBarrierCount = buffer_barriers.size(),
                .pBufferMemoryBarriers = buffer_barriers2.data(),
                .imageMemoryBarrierCount = image_barriers.size(),
                .pImageMemoryBarriers = image_barriers2.data(),
            };
            dld->vkCmdPipelineBarrier2(handle, &dependency_info);
            return;
        }
        dld->vkCmdPipelineBarrier(handle, src_stage_mask, dst_stage_mask, dependency_flags,
                                  memory_barriers.size(), memory_barriers.data(),
                                  buffer_barriers.size(), buffer_barriers.data(),
                                  image_barriers.size(), image_barriers.data());
    }

    void PipelineBarrier(VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
                         VkDependencyFlags dependency_flags = 0) const noexcept {
        PipelineBarrier(src_stage_mask, dst_stage_mask, dependency_flags, {}, {}, {});
    }

    void PipelineBarrier(VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
                         VkDependencyFlags dependency_flags,
                         const VkMemoryBarrier& memory_barrier) const noexcept {
        PipelineBarrier(src_stage_mask, dst_stage_mask, dependency_flags, memory_barrier, {}, {});
    }

    void PipelineBarrier(VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
                         VkDependencyFlags dependency_flags,
                         const VkBufferMemoryBarrier& buffer_barrier) const noexcept {
        PipelineBarrier(src_stage_mask, dst_stage_mask, dependency_flags, {}, buffer_barrier, {});
    }

    void PipelineBarrier(VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
                         VkDependencyFlags dependency_flags,
                         const VkImageMemoryBarrier& image_barrier) const noexcept {
        PipelineBarrier(src_stage_mask, dst_stage_mask, dependency_flags, {}, {}, image_barrier);
    }

    void BindDescriptorBuffersEXT(Span<VkDescriptorBufferBindingInfoEXT> bindings) const noexcept {
        diagnostic::Trace("BIND_DESCRIPTOR_BUFFERS",
                          "cmd={:#x} count={}", diagnostic::HandleBits(handle), bindings.size());
        for (u32 i = 0; i < bindings.size(); ++i) {
            diagnostic::Trace("DESCRIPTOR_BUFFER_BINDING",
                              "cmd={:#x} index={} address={:#x} usage={:#x}",
                              diagnostic::HandleBits(handle), i, bindings[i].address,
                              static_cast<u32>(bindings[i].usage));
        }
        dld->vkCmdBindDescriptorBuffersEXT(handle, bindings.size(), bindings.data());
    }

    void SetDescriptorBufferOffsetsEXT(VkPipelineBindPoint bind_point, VkPipelineLayout layout,
                                       u32 first_set, Span<u32> buffer_indices,
                                       Span<VkDeviceSize> offsets) const noexcept {
        diagnostic::Trace("SET_DESCRIPTOR_BUFFER_OFFSETS",
                          "cmd={:#x} bind_point={} layout={:#x} first_set={} count={}",
                          diagnostic::HandleBits(handle), static_cast<u32>(bind_point),
                          diagnostic::HandleBits(layout), first_set, buffer_indices.size());
        for (u32 i = 0; i < buffer_indices.size(); ++i) {
            diagnostic::Trace("DESCRIPTOR_BUFFER_OFFSET",
                              "cmd={:#x} set={} buffer_index={} offset={}",
                              diagnostic::HandleBits(handle), first_set + i,
                              buffer_indices[i], offsets[i]);
        }
        dld->vkCmdSetDescriptorBufferOffsetsEXT(handle, bind_point, layout, first_set,
                                                buffer_indices.size(), buffer_indices.data(),
                                                offsets.data());
    }

    void CopyBufferToImage(VkBuffer src_buffer, VkImage dst_image, VkImageLayout dst_image_layout,
                           Span<VkBufferImageCopy> regions) const noexcept {
        diagnostic::Trace("COPY_BUFFER_TO_IMAGE",
                          "cmd={:#x} src_buffer={:#x} dst_image={:#x} dst_layout={} regions={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(src_buffer),
                          diagnostic::HandleBits(dst_image), static_cast<u32>(dst_image_layout), regions.size());
        for (u32 i = 0; i < regions.size(); ++i) {
            const auto& r = regions[i];
            diagnostic::Trace("COPY_BUFFER_IMAGE_REGION",
                              "cmd={:#x} index={} buffer_offset={} row_length={} image_height={} "
                              "aspect={:#x} mip={} layer={} layers={} image_offset=({}, {}, {}) extent=({}, {}, {})",
                              diagnostic::HandleBits(handle), i, r.bufferOffset, r.bufferRowLength,
                              r.bufferImageHeight, static_cast<u32>(r.imageSubresource.aspectMask),
                              r.imageSubresource.mipLevel, r.imageSubresource.baseArrayLayer,
                              r.imageSubresource.layerCount, r.imageOffset.x, r.imageOffset.y,
                              r.imageOffset.z, r.imageExtent.width, r.imageExtent.height, r.imageExtent.depth);
        }
        dld->vkCmdCopyBufferToImage(handle, src_buffer, dst_image, dst_image_layout, regions.size(),
                                    regions.data());
    }

    void CopyBuffer(VkBuffer src_buffer, VkBuffer dst_buffer,
                    Span<VkBufferCopy> regions) const noexcept {
        diagnostic::Trace("COPY_BUFFER",
                          "cmd={:#x} src={:#x} dst={:#x} region_count={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(src_buffer),
                          diagnostic::HandleBits(dst_buffer), regions.size());
        for (u32 i = 0; i < regions.size(); ++i) {
            const auto& r = regions[i];
            diagnostic::Trace("COPY_BUFFER_REGION",
                              "cmd={:#x} index={} src_offset={} dst_offset={} size={}",
                              diagnostic::HandleBits(handle), i, r.srcOffset, r.dstOffset, r.size);
        }
        dld->vkCmdCopyBuffer(handle, src_buffer, dst_buffer, regions.size(), regions.data());
    }

    void CopyImage(VkImage src_image, VkImageLayout src_layout, VkImage dst_image,
                   VkImageLayout dst_layout, Span<VkImageCopy> regions) const noexcept {
        diagnostic::Trace("COPY_IMAGE",
                          "cmd={:#x} src={:#x} src_layout={} dst={:#x} dst_layout={} regions={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(src_image),
                          static_cast<u32>(src_layout), diagnostic::HandleBits(dst_image),
                          static_cast<u32>(dst_layout), regions.size());
        for (u32 i = 0; i < regions.size(); ++i) {
            const auto& r = regions[i];
            diagnostic::Trace("COPY_IMAGE_REGION",
                              "cmd={:#x} index={} src_aspect={:#x} src_mip={} src_layer={} "
                              "src_offset=({}, {}, {}) dst_aspect={:#x} dst_mip={} dst_layer={} "
                              "dst_offset=({}, {}, {}) extent=({}, {}, {})",
                              diagnostic::HandleBits(handle), i, static_cast<u32>(r.srcSubresource.aspectMask),
                              r.srcSubresource.mipLevel, r.srcSubresource.baseArrayLayer,
                              r.srcOffset.x, r.srcOffset.y, r.srcOffset.z,
                              static_cast<u32>(r.dstSubresource.aspectMask), r.dstSubresource.mipLevel,
                              r.dstSubresource.baseArrayLayer, r.dstOffset.x, r.dstOffset.y,
                              r.dstOffset.z, r.extent.width, r.extent.height, r.extent.depth);
        }
        dld->vkCmdCopyImage(handle, src_image, src_layout, dst_image, dst_layout, regions.size(),
                            regions.data());
    }

    void CopyImageToBuffer(VkImage src_image, VkImageLayout src_layout, VkBuffer dst_buffer,
                           Span<VkBufferImageCopy> regions) const noexcept {
        diagnostic::Trace("COPY_IMAGE_TO_BUFFER",
                          "cmd={:#x} src_image={:#x} src_layout={} dst_buffer={:#x} regions={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(src_image),
                          static_cast<u32>(src_layout), diagnostic::HandleBits(dst_buffer), regions.size());
        for (u32 i = 0; i < regions.size(); ++i) {
            const auto& r = regions[i];
            diagnostic::Trace("COPY_IMAGE_BUFFER_REGION",
                              "cmd={:#x} index={} buffer_offset={} row_length={} image_height={} "
                              "aspect={:#x} mip={} layer={} layers={} image_offset=({}, {}, {}) extent=({}, {}, {})",
                              diagnostic::HandleBits(handle), i, r.bufferOffset, r.bufferRowLength,
                              r.bufferImageHeight, static_cast<u32>(r.imageSubresource.aspectMask),
                              r.imageSubresource.mipLevel, r.imageSubresource.baseArrayLayer,
                              r.imageSubresource.layerCount, r.imageOffset.x, r.imageOffset.y,
                              r.imageOffset.z, r.imageExtent.width, r.imageExtent.height, r.imageExtent.depth);
        }
        dld->vkCmdCopyImageToBuffer(handle, src_image, src_layout, dst_buffer, regions.size(),
                                    regions.data());
    }

    void CopyQueryPoolResults(VkQueryPool query_pool, u32 first_query, u32 query_count,
                              VkBuffer dst_buffer, VkDeviceSize dst_offset, VkDeviceSize stride,
                              VkQueryResultFlags flags) const noexcept {
        dld->vkCmdCopyQueryPoolResults(handle, query_pool, first_query, query_count, dst_buffer,
                                       dst_offset, stride, flags);
    }

    void FillBuffer(VkBuffer dst_buffer, VkDeviceSize dst_offset, VkDeviceSize size,
                    u32 data) const noexcept {
        diagnostic::Trace("FILL_BUFFER",
                          "cmd={:#x} buffer={:#x} offset={} size={} value={:#010x}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(dst_buffer),
                          dst_offset, size, data);
        dld->vkCmdFillBuffer(handle, dst_buffer, dst_offset, size, data);
    }

    void PushConstants(VkPipelineLayout layout, VkShaderStageFlags flags, u32 offset, u32 size,
                       const void* values) const noexcept {
        diagnostic::Trace("PUSH_CONSTANTS",
                          "cmd={:#x} layout={:#x} stages={:#x} offset={} size={} data={:#x}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(layout),
                          static_cast<u32>(flags), offset, size,
                          static_cast<u64>(reinterpret_cast<std::uintptr_t>(values)));
        dld->vkCmdPushConstants(handle, layout, flags, offset, size, values);
    }

    template <typename T>
        requires std::is_trivially_copyable_v<T>
    void PushConstants(VkPipelineLayout layout, VkShaderStageFlags flags, const T& data) const noexcept {
        dld->vkCmdPushConstants(handle, layout, flags, 0, u32(sizeof(T)), std::addressof(data));
    }

    void SetViewport(u32 first, Span<VkViewport> viewports) const noexcept {
        diagnostic::Trace("SET_VIEWPORT", "cmd={:#x} first={} count={}",
                          diagnostic::HandleBits(handle), first, viewports.size());
        for (u32 i = 0; i < viewports.size(); ++i) {
            const auto& v = viewports[i];
            diagnostic::Trace("VIEWPORT",
                              "cmd={:#x} index={} x={} y={} width={} height={} min_depth={} max_depth={}",
                              diagnostic::HandleBits(handle), first + i, v.x, v.y, v.width,
                              v.height, v.minDepth, v.maxDepth);
        }
        dld->vkCmdSetViewport(handle, first, viewports.size(), viewports.data());
    }

    void SetScissor(u32 first, Span<VkRect2D> scissors) const noexcept {
        diagnostic::Trace("SET_SCISSOR", "cmd={:#x} first={} count={}",
                          diagnostic::HandleBits(handle), first, scissors.size());
        for (u32 i = 0; i < scissors.size(); ++i) {
            const auto& r = scissors[i];
            diagnostic::Trace("SCISSOR",
                              "cmd={:#x} index={} offset=({}, {}) extent={}x{}",
                              diagnostic::HandleBits(handle), first + i, r.offset.x, r.offset.y,
                              r.extent.width, r.extent.height);
        }
        dld->vkCmdSetScissor(handle, first, scissors.size(), scissors.data());
    }

    void SetBlendConstants(const float blend_constants[4]) const noexcept {
        diagnostic::Trace("SET_BLEND_CONSTANTS", "cmd={:#x} rgba=({}, {}, {}, {})",
                          diagnostic::HandleBits(handle), blend_constants[0], blend_constants[1],
                          blend_constants[2], blend_constants[3]);
        dld->vkCmdSetBlendConstants(handle, blend_constants);
    }

    void SetStencilCompareMask(VkStencilFaceFlags face_mask, u32 compare_mask) const noexcept {
        diagnostic::Trace("SET_STENCIL_COMPARE_MASK", "cmd={:#x} faces={:#x} mask={:#x}",
                          diagnostic::HandleBits(handle), static_cast<u32>(face_mask), compare_mask);
        dld->vkCmdSetStencilCompareMask(handle, face_mask, compare_mask);
    }

    void SetStencilReference(VkStencilFaceFlags face_mask, u32 reference) const noexcept {
        diagnostic::Trace("SET_STENCIL_REFERENCE", "cmd={:#x} faces={:#x} reference={:#x}",
                          diagnostic::HandleBits(handle), static_cast<u32>(face_mask), reference);
        dld->vkCmdSetStencilReference(handle, face_mask, reference);
    }

    void SetStencilWriteMask(VkStencilFaceFlags face_mask, u32 write_mask) const noexcept {
        diagnostic::Trace("SET_STENCIL_WRITE_MASK", "cmd={:#x} faces={:#x} mask={:#x}",
                          diagnostic::HandleBits(handle), static_cast<u32>(face_mask), write_mask);
        dld->vkCmdSetStencilWriteMask(handle, face_mask, write_mask);
    }

    void SetDepthBias(float constant_factor, float clamp, float slope_factor) const noexcept {
        diagnostic::Trace("SET_DEPTH_BIAS",
                          "cmd={:#x} constant={} clamp={} slope={}",
                          diagnostic::HandleBits(handle), constant_factor, clamp, slope_factor);
        dld->vkCmdSetDepthBias(handle, constant_factor, clamp, slope_factor);
    }

    void SetDepthBias(float constant_factor, float clamp, float slope_factor,
                      VkDepthBiasRepresentationInfoEXT* extra) const noexcept {
        VkDepthBiasInfoEXT info{
            .sType = VK_STRUCTURE_TYPE_DEPTH_BIAS_INFO_EXT,
            .pNext = extra,
            .depthBiasConstantFactor = constant_factor,
            .depthBiasClamp = clamp,
            .depthBiasSlopeFactor = slope_factor,
        };
        diagnostic::Trace("SET_DEPTH_BIAS2",
                          "cmd={:#x} constant={} clamp={} slope={} representation={}",
                          diagnostic::HandleBits(handle), constant_factor, clamp, slope_factor,
                          extra ? static_cast<u32>(extra->depthBiasRepresentation) : UINT32_MAX);
        dld->vkCmdSetDepthBias2EXT(handle, &info);
    }

    void SetDepthBounds(float min_depth_bounds, float max_depth_bounds) const noexcept {
        diagnostic::Trace("SET_DEPTH_BOUNDS", "cmd={:#x} min={} max={}",
                          diagnostic::HandleBits(handle), min_depth_bounds, max_depth_bounds);
        dld->vkCmdSetDepthBounds(handle, min_depth_bounds, max_depth_bounds);
    }

    void SetEvent(VkEvent event, VkPipelineStageFlags stage_flags) const noexcept {
        diagnostic::Trace("SET_EVENT", "cmd={:#x} event={:#x} stage={:#x}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(event),
                          static_cast<u64>(stage_flags));
        dld->vkCmdSetEvent(handle, event, stage_flags);
    }

    void WaitEvents(Span<VkEvent> events, VkPipelineStageFlags src_stage_mask,
                    VkPipelineStageFlags dst_stage_mask, Span<VkMemoryBarrier> memory_barriers,
                    Span<VkBufferMemoryBarrier> buffer_barriers,
                    Span<VkImageMemoryBarrier> image_barriers) const noexcept {
        diagnostic::Trace("WAIT_EVENTS",
                          "cmd={:#x} event_count={} src_stage={:#x} dst_stage={:#x} "
                          "memory_count={} buffer_count={} image_count={}",
                          diagnostic::HandleBits(handle), events.size(),
                          static_cast<u64>(src_stage_mask), static_cast<u64>(dst_stage_mask),
                          memory_barriers.size(), buffer_barriers.size(), image_barriers.size());
        for (u32 i = 0; i < events.size(); ++i) {
            diagnostic::Trace("WAIT_EVENT_HANDLE", "cmd={:#x} index={} event={:#x}",
                              diagnostic::HandleBits(handle), i, diagnostic::HandleBits(events[i]));
        }
        for (u32 i = 0; i < memory_barriers.size(); ++i) {
            const auto& b = memory_barriers[i];
            diagnostic::Trace("WAIT_EVENT_MEMORY_BARRIER",
                              "cmd={:#x} index={} src_access={:#x} dst_access={:#x}",
                              diagnostic::HandleBits(handle), i,
                              static_cast<u64>(b.srcAccessMask), static_cast<u64>(b.dstAccessMask));
        }
        for (u32 i = 0; i < buffer_barriers.size(); ++i) {
            const auto& b = buffer_barriers[i];
            diagnostic::Trace("WAIT_EVENT_BUFFER_BARRIER",
                              "cmd={:#x} index={} buffer={:#x} offset={} size={} src_access={:#x} dst_access={:#x} src_queue={} dst_queue={}",
                              diagnostic::HandleBits(handle), i, diagnostic::HandleBits(b.buffer),
                              b.offset, b.size, static_cast<u64>(b.srcAccessMask),
                              static_cast<u64>(b.dstAccessMask), b.srcQueueFamilyIndex,
                              b.dstQueueFamilyIndex);
        }
        for (u32 i = 0; i < image_barriers.size(); ++i) {
            const auto& b = image_barriers[i];
            const auto& r = b.subresourceRange;
            diagnostic::Trace("WAIT_EVENT_IMAGE_BARRIER",
                              "cmd={:#x} index={} image={:#x} old_layout={} new_layout={} src_access={:#x} dst_access={:#x} aspect={:#x} base_mip={} mip_count={} base_layer={} layer_count={}",
                              diagnostic::HandleBits(handle), i, diagnostic::HandleBits(b.image),
                              static_cast<u32>(b.oldLayout), static_cast<u32>(b.newLayout),
                              static_cast<u64>(b.srcAccessMask), static_cast<u64>(b.dstAccessMask),
                              static_cast<u32>(r.aspectMask), r.baseMipLevel, r.levelCount,
                              r.baseArrayLayer, r.layerCount);
        }
        dld->vkCmdWaitEvents(handle, events.size(), events.data(), src_stage_mask, dst_stage_mask,
                             memory_barriers.size(), memory_barriers.data(), buffer_barriers.size(),
                             buffer_barriers.data(), image_barriers.size(), image_barriers.data());
    }

    void BindVertexBuffers2EXT(u32 first_binding, u32 binding_count, const VkBuffer* buffers,
                               const VkDeviceSize* offsets, const VkDeviceSize* sizes,
                               const VkDeviceSize* strides) const noexcept {
        diagnostic::Trace("BIND_VERTEX_BUFFERS_2",
                          "cmd={:#x} first_binding={} count={} sizes_present={} strides_present={}",
                          diagnostic::HandleBits(handle), first_binding, binding_count,
                          sizes != nullptr, strides != nullptr);
        for (u32 i = 0; i < binding_count; ++i) {
            diagnostic::Trace("BIND_VERTEX_BUFFER_2",
                              "cmd={:#x} binding={} buffer={:#x} offset={} size={} stride={}",
                              diagnostic::HandleBits(handle), first_binding + i,
                              diagnostic::HandleBits(buffers[i]), offsets ? offsets[i] : 0,
                              sizes ? sizes[i] : VK_WHOLE_SIZE, strides ? strides[i] : 0);
        }
        dld->vkCmdBindVertexBuffers2EXT(handle, first_binding, binding_count, buffers, offsets,
                                        sizes, strides);
    }

    void SetCullModeEXT(VkCullModeFlags cull_mode) const noexcept {
        diagnostic::Trace("SET_CULL_MODE", "cmd={:#x} cull_mode={:#x}",
                          diagnostic::HandleBits(handle), static_cast<u32>(cull_mode));
        dld->vkCmdSetCullModeEXT(handle, cull_mode);
    }

    void SetDepthBoundsTestEnableEXT(bool enable) const noexcept {
        diagnostic::Trace("SET_DEPTH_BOUNDS_TEST_ENABLE", "cmd={:#x} enable={}",
                          diagnostic::HandleBits(handle), enable);
        dld->vkCmdSetDepthBoundsTestEnableEXT(handle, enable ? VK_TRUE : VK_FALSE);
    }

    void SetDepthCompareOpEXT(VkCompareOp compare_op) const noexcept {
        diagnostic::Trace("SET_DEPTH_COMPARE_OP", "cmd={:#x} compare_op={}",
                          diagnostic::HandleBits(handle), static_cast<u32>(compare_op));
        dld->vkCmdSetDepthCompareOpEXT(handle, compare_op);
    }

    void SetDepthTestEnableEXT(bool enable) const noexcept {
        diagnostic::Trace("SET_DEPTH_TEST_ENABLE", "cmd={:#x} enable={}",
                          diagnostic::HandleBits(handle), enable);
        dld->vkCmdSetDepthTestEnableEXT(handle, enable ? VK_TRUE : VK_FALSE);
    }

    void SetDepthWriteEnableEXT(bool enable) const noexcept {
        diagnostic::Trace("SET_DEPTH_WRITE_ENABLE", "cmd={:#x} enable={}",
                          diagnostic::HandleBits(handle), enable);
        dld->vkCmdSetDepthWriteEnableEXT(handle, enable ? VK_TRUE : VK_FALSE);
    }

    void SetPrimitiveRestartEnableEXT(bool enable) const noexcept {
        diagnostic::Trace("SET_PRIMITIVE_RESTART_ENABLE", "cmd={:#x} enable={}",
                          diagnostic::HandleBits(handle), enable);
        dld->vkCmdSetPrimitiveRestartEnableEXT(handle, enable ? VK_TRUE : VK_FALSE);
    }

    void SetRasterizerDiscardEnableEXT(bool enable) const noexcept {
        diagnostic::Trace("SET_RASTERIZER_DISCARD_ENABLE", "cmd={:#x} enable={}",
                          diagnostic::HandleBits(handle), enable);
        dld->vkCmdSetRasterizerDiscardEnableEXT(handle, enable ? VK_TRUE : VK_FALSE);
    }

    void SetConservativeRasterizationModeEXT(VkConservativeRasterizationModeEXT mode) const noexcept
    {
        dld->vkCmdSetConservativeRasterizationModeEXT(handle, mode);
    }

    void SetLineRasterizationModeEXT(VkLineRasterizationModeEXT mode) const noexcept
    {
        dld->vkCmdSetLineRasterizationModeEXT(handle, mode);
    }

    void SetLineStippleEnableEXT(bool enable) const noexcept
    {
        dld->vkCmdSetLineStippleEnableEXT(handle, enable ? VK_TRUE : VK_FALSE);
    }

    void SetLineStippleEXT(u32 factor, u16 pattern) const noexcept
    {
        dld->vkCmdSetLineStippleEXT(handle, factor, pattern);
    }

    void SetDepthBiasEnableEXT(bool enable) const noexcept {
        dld->vkCmdSetDepthBiasEnableEXT(handle, enable ? VK_TRUE : VK_FALSE);
    }

    void SetLogicOpEnableEXT(bool enable) const noexcept {
        diagnostic::Trace("SET_LOGIC_OP_ENABLE", "cmd={:#x} enable={}",
                          diagnostic::HandleBits(handle), enable);
        dld->vkCmdSetLogicOpEnableEXT(handle, enable ? VK_TRUE : VK_FALSE);
    }

    void SetAlphaToCoverageEnableEXT(bool enable) const noexcept {
        diagnostic::Trace("SET_ALPHA_TO_COVERAGE_ENABLE", "cmd={:#x} enable={}",
                          diagnostic::HandleBits(handle), enable);
        dld->vkCmdSetAlphaToCoverageEnableEXT(handle, enable ? VK_TRUE : VK_FALSE);
    }

    void SetAlphaToOneEnableEXT(bool enable) const noexcept {
        diagnostic::Trace("SET_ALPHA_TO_ONE_ENABLE", "cmd={:#x} enable={}",
                          diagnostic::HandleBits(handle), enable);
        dld->vkCmdSetAlphaToOneEnableEXT(handle, enable ? VK_TRUE : VK_FALSE);
    }

    void SetDepthClampEnableEXT(bool enable) const noexcept {
        diagnostic::Trace("SET_DEPTH_CLAMP_ENABLE", "cmd={:#x} enable={}",
                          diagnostic::HandleBits(handle), enable);
        dld->vkCmdSetDepthClampEnableEXT(handle, enable ? VK_TRUE : VK_FALSE);
    }

    void SetFrontFaceEXT(VkFrontFace front_face) const noexcept {
        diagnostic::Trace("SET_FRONT_FACE", "cmd={:#x} front_face={}",
                          diagnostic::HandleBits(handle), static_cast<u32>(front_face));
        dld->vkCmdSetFrontFaceEXT(handle, front_face);
    }

    void SetLogicOpEXT(VkLogicOp logic_op) const noexcept {
        diagnostic::Trace("SET_LOGIC_OP", "cmd={:#x} logic_op={}",
                          diagnostic::HandleBits(handle), static_cast<u32>(logic_op));
        dld->vkCmdSetLogicOpEXT(handle, logic_op);
    }

    void SetPatchControlPointsEXT(uint32_t patch_control_points) const noexcept {
        dld->vkCmdSetPatchControlPointsEXT(handle, patch_control_points);
    }

    void SetColorWriteMaskEXT(u32 first, Span<VkColorComponentFlags> masks) const noexcept {
        diagnostic::Trace("SET_COLOR_WRITE_MASK", "cmd={:#x} first={} count={}",
                          diagnostic::HandleBits(handle), first, masks.size());
        for (u32 i = 0; i < masks.size(); ++i) {
            diagnostic::Trace("COLOR_WRITE_MASK", "cmd={:#x} attachment={} mask={:#x}",
                              diagnostic::HandleBits(handle), first + i,
                              static_cast<u32>(masks[i]));
        }
        dld->vkCmdSetColorWriteMaskEXT(handle, first, masks.size(), masks.data());
    }

    void SetColorWriteEnableEXT(Span<VkBool32> enables) const noexcept {
        diagnostic::Trace("SET_COLOR_WRITE_ENABLE", "cmd={:#x} count={}",
                          diagnostic::HandleBits(handle), enables.size());
        for (u32 i = 0; i < enables.size(); ++i) {
            diagnostic::Trace("COLOR_WRITE_ENABLE", "cmd={:#x} attachment={} enabled={}",
                              diagnostic::HandleBits(handle), i, enables[i] != VK_FALSE);
        }
        dld->vkCmdSetColorWriteEnableEXT(handle, enables.size(), enables.data());
    }

    void SetColorBlendEnableEXT(u32 first, Span<VkBool32> enables) const noexcept {
        diagnostic::Trace("SET_COLOR_BLEND_ENABLE", "cmd={:#x} first={} count={}",
                          diagnostic::HandleBits(handle), first, enables.size());
        for (u32 i = 0; i < enables.size(); ++i) {
            diagnostic::Trace("COLOR_BLEND_ENABLE", "cmd={:#x} attachment={} enabled={}",
                              diagnostic::HandleBits(handle), first + i, enables[i] != VK_FALSE);
        }
        dld->vkCmdSetColorBlendEnableEXT(handle, first, enables.size(), enables.data());
    }

    void SetColorBlendEquationEXT(u32 first,
                                  Span<VkColorBlendEquationEXT> equations) const noexcept {
        diagnostic::Trace("SET_COLOR_BLEND_EQUATION", "cmd={:#x} first={} count={}",
                          diagnostic::HandleBits(handle), first, equations.size());
        for (u32 i = 0; i < equations.size(); ++i) {
            const auto& e = equations[i];
            diagnostic::Trace("COLOR_BLEND_EQUATION",
                              "cmd={:#x} attachment={} src_color={} dst_color={} color_op={} "
                              "src_alpha={} dst_alpha={} alpha_op={}",
                              diagnostic::HandleBits(handle), first + i,
                              static_cast<u32>(e.srcColorBlendFactor),
                              static_cast<u32>(e.dstColorBlendFactor),
                              static_cast<u32>(e.colorBlendOp),
                              static_cast<u32>(e.srcAlphaBlendFactor),
                              static_cast<u32>(e.dstAlphaBlendFactor),
                              static_cast<u32>(e.alphaBlendOp));
        }
        dld->vkCmdSetColorBlendEquationEXT(handle, first, equations.size(), equations.data());
    }

    void SetLineWidth(float line_width) const noexcept {
        dld->vkCmdSetLineWidth(handle, line_width);
    }

    void SetPrimitiveTopologyEXT(VkPrimitiveTopology primitive_topology) const noexcept {
        dld->vkCmdSetPrimitiveTopologyEXT(handle, primitive_topology);
    }

    void SetStencilOpEXT(VkStencilFaceFlags face_mask, VkStencilOp fail_op, VkStencilOp pass_op,
                         VkStencilOp depth_fail_op, VkCompareOp compare_op) const noexcept {
        diagnostic::Trace("SET_STENCIL_OP",
                          "cmd={:#x} faces={:#x} fail={} pass={} depth_fail={} compare={}",
                          diagnostic::HandleBits(handle), static_cast<u32>(face_mask),
                          static_cast<u32>(fail_op), static_cast<u32>(pass_op),
                          static_cast<u32>(depth_fail_op), static_cast<u32>(compare_op));
        dld->vkCmdSetStencilOpEXT(handle, face_mask, fail_op, pass_op, depth_fail_op, compare_op);
    }

    void SetStencilTestEnableEXT(bool enable) const noexcept {
        diagnostic::Trace("SET_STENCIL_TEST_ENABLE", "cmd={:#x} enable={}",
                          diagnostic::HandleBits(handle), enable);
        dld->vkCmdSetStencilTestEnableEXT(handle, enable ? VK_TRUE : VK_FALSE);
    }

    void SetVertexInputEXT(
        vk::Span<VkVertexInputBindingDescription2EXT> bindings,
        vk::Span<VkVertexInputAttributeDescription2EXT> attributes) const noexcept {
        diagnostic::Trace("SET_VERTEX_INPUT",
                          "cmd={:#x} binding_count={} attribute_count={}",
                          diagnostic::HandleBits(handle), bindings.size(), attributes.size());
        for (u32 i = 0; i < bindings.size(); ++i) {
            const auto& b = bindings[i];
            diagnostic::Trace("VERTEX_INPUT_BINDING",
                              "cmd={:#x} index={} binding={} stride={} input_rate={} divisor={}",
                              diagnostic::HandleBits(handle), i, b.binding, b.stride,
                              static_cast<u32>(b.inputRate), b.divisor);
        }
        for (u32 i = 0; i < attributes.size(); ++i) {
            const auto& a = attributes[i];
            diagnostic::Trace("VERTEX_INPUT_ATTRIBUTE",
                              "cmd={:#x} index={} location={} binding={} format={} offset={}",
                              diagnostic::HandleBits(handle), i, a.location, a.binding,
                              static_cast<s32>(a.format), a.offset);
        }
        dld->vkCmdSetVertexInputEXT(handle, bindings.size(), bindings.data(), attributes.size(),
                                    attributes.data());
    }

    void BindTransformFeedbackBuffersEXT(u32 first, u32 count, const VkBuffer* buffers,
                                         const VkDeviceSize* offsets,
                                         const VkDeviceSize* sizes) const noexcept {
        diagnostic::Trace("BIND_TRANSFORM_FEEDBACK_BUFFERS",
                          "cmd={:#x} first={} count={}", diagnostic::HandleBits(handle), first, count);
        for (u32 i = 0; i < count; ++i) {
            diagnostic::Trace("TRANSFORM_FEEDBACK_BUFFER",
                              "cmd={:#x} binding={} buffer={:#x} offset={} size={}",
                              diagnostic::HandleBits(handle), first + i,
                              diagnostic::HandleBits(buffers[i]), offsets ? offsets[i] : 0,
                              sizes ? sizes[i] : VK_WHOLE_SIZE);
        }
        dld->vkCmdBindTransformFeedbackBuffersEXT(handle, first, count, buffers, offsets, sizes);
    }

    void BeginTransformFeedbackEXT(u32 first_counter_buffer, u32 counter_buffers_count,
                                   const VkBuffer* counter_buffers,
                                   const VkDeviceSize* counter_buffer_offsets) const noexcept {
        diagnostic::Trace("BEGIN_TRANSFORM_FEEDBACK",
                          "cmd={:#x} first_counter={} count={}",
                          diagnostic::HandleBits(handle), first_counter_buffer, counter_buffers_count);
        dld->vkCmdBeginTransformFeedbackEXT(handle, first_counter_buffer, counter_buffers_count,
                                            counter_buffers, counter_buffer_offsets);
    }

    void EndTransformFeedbackEXT(u32 first_counter_buffer, u32 counter_buffers_count,
                                 const VkBuffer* counter_buffers,
                                 const VkDeviceSize* counter_buffer_offsets) const noexcept {
        diagnostic::Trace("END_TRANSFORM_FEEDBACK",
                          "cmd={:#x} first_counter={} count={}",
                          diagnostic::HandleBits(handle), first_counter_buffer, counter_buffers_count);
        dld->vkCmdEndTransformFeedbackEXT(handle, first_counter_buffer, counter_buffers_count,
                                          counter_buffers, counter_buffer_offsets);
    }

    void BeginConditionalRenderingEXT(
        const VkConditionalRenderingBeginInfoEXT& info) const noexcept {
        diagnostic::Trace("BEGIN_CONDITIONAL_RENDERING",
                          "cmd={:#x} buffer={:#x} offset={} flags={:#x}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(info.buffer),
                          info.offset, static_cast<u32>(info.flags));
        dld->vkCmdBeginConditionalRenderingEXT(handle, &info);
    }

    void EndConditionalRenderingEXT() const noexcept {
        diagnostic::Trace("END_CONDITIONAL_RENDERING", "cmd={:#x}",
                          diagnostic::HandleBits(handle));
        dld->vkCmdEndConditionalRenderingEXT(handle);
    }

    void BeginDebugUtilsLabelEXT(const char* label, std::span<float, 4> color) const noexcept {
        const VkDebugUtilsLabelEXT label_info{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
            .pNext = nullptr,
            .pLabelName = label,
            .color{color[0], color[1], color[2], color[3]},
        };
        diagnostic::Trace("DEBUG_LABEL_BEGIN",
                          "cmd={:#x} label={} color=({}, {}, {}, {})",
                          diagnostic::HandleBits(handle), label ? label : "(unnamed)",
                          color[0], color[1], color[2], color[3]);
        dld->vkCmdBeginDebugUtilsLabelEXT(handle, &label_info);
    }

    void EndDebugUtilsLabelEXT() const noexcept {
        diagnostic::Trace("DEBUG_LABEL_END", "cmd={:#x}", diagnostic::HandleBits(handle));
        dld->vkCmdEndDebugUtilsLabelEXT(handle);
    }

private:
    VkCommandBuffer handle;
    const DeviceDispatch* dld;
};

u32 AvailableVersion(const InstanceDispatch& dld) noexcept;

std::optional<std::vector<VkExtensionProperties>> EnumerateInstanceExtensionProperties(
    const InstanceDispatch& dld);

std::optional<std::vector<VkLayerProperties>> EnumerateInstanceLayerProperties(
    const InstanceDispatch& dld);

std::string GetDriverName(VkPhysicalDeviceDriverProperties driver);

} // namespace Vulkan::vk
