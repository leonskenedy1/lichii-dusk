// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "common/logging.h"
#include "video_core/vulkan_common/vk_enum_string_helper.h"
#include "video_core/vulkan_common/vma.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan::vk {

namespace {

template <typename Func>
void SortPhysicalDevices(std::vector<VkPhysicalDevice>& devices, const InstanceDispatch& dld,
                         Func&& func) {
    // Calling GetProperties calls Vulkan more than needed. But they are supposed to be cheap
    // functions.
    std::stable_sort(devices.begin(), devices.end(),
                     [&dld, &func](VkPhysicalDevice lhs, VkPhysicalDevice rhs) {
                         return func(vk::PhysicalDevice(lhs, dld).GetProperties(),
                                     vk::PhysicalDevice(rhs, dld).GetProperties());
                     });
}

void SortPhysicalDevicesPerVendor(std::vector<VkPhysicalDevice>& devices,
                                  const InstanceDispatch& dld,
                                  std::initializer_list<u32> vendor_ids) {
    for (auto it = vendor_ids.end(); it != vendor_ids.begin();) {
        --it;
        SortPhysicalDevices(devices, dld, [id = *it](const auto& lhs, const auto& rhs) {
            return lhs.vendorID == id && rhs.vendorID != id;
        });
    }
}

bool IsMicrosoftDozen(const char* device_name) {
    return std::strstr(device_name, "Microsoft") != nullptr;
}

void SortPhysicalDevices(std::vector<VkPhysicalDevice>& devices, const InstanceDispatch& dld) {
    // Sort by name, this will set a base and make GPUs with higher numbers appear first
    // (e.g. GTX 1650 will intentionally be listed before a GTX 1080).
    SortPhysicalDevices(devices, dld, [](const auto& lhs, const auto& rhs) {
        return std::string_view{lhs.deviceName} > std::string_view{rhs.deviceName};
    });
    // Prefer discrete over non-discrete
    SortPhysicalDevices(devices, dld, [](const auto& lhs, const auto& rhs) {
        return lhs.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
               rhs.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    });
    // Prefer Nvidia over AMD, AMD over Intel, Intel over the rest.
    SortPhysicalDevicesPerVendor(devices, dld, {0x10DE, 0x1002, 0x8086});
    // Demote Microsoft's Dozen devices to the bottom.
    SortPhysicalDevices(
        devices, dld,
        [](const VkPhysicalDeviceProperties& lhs, const VkPhysicalDeviceProperties& rhs) {
            return IsMicrosoftDozen(rhs.deviceName) && !IsMicrosoftDozen(lhs.deviceName);
        });
}

template <typename T>
bool Proc(T& result, const InstanceDispatch& dld, const char* proc_name,
          VkInstance instance = nullptr) noexcept {
    result = reinterpret_cast<T>(dld.vkGetInstanceProcAddr(instance, proc_name));
    return result != nullptr;
}

template <typename T>
void Proc(T& result, const DeviceDispatch& dld, const char* proc_name, VkDevice device) noexcept {
    result = reinterpret_cast<T>(dld.vkGetDeviceProcAddr(device, proc_name));
}

void Load(VkDevice device, DeviceDispatch& dld) noexcept {
#define X(name) Proc(dld.name, dld, #name, device)
    X(vkAcquireNextImageKHR);
    X(vkAllocateCommandBuffers);
    X(vkAllocateDescriptorSets);
    X(vkAllocateMemory);
    X(vkBeginCommandBuffer);
    X(vkBindBufferMemory);
    X(vkBindImageMemory);
    X(vkCmdBeginConditionalRenderingEXT);
    X(vkCmdBeginQuery);
    X(vkCmdBeginRenderPass);
    X(vkCmdBeginTransformFeedbackEXT);
    X(vkCmdBeginDebugUtilsLabelEXT);
    X(vkCmdBindDescriptorSets);
    X(vkCmdBindIndexBuffer);
    X(vkCmdBindPipeline);
    X(vkCmdBindTransformFeedbackBuffersEXT);
    X(vkCmdBindVertexBuffers);
    X(vkCmdBlitImage);
    X(vkCmdClearAttachments);
    X(vkCmdClearColorImage);
    X(vkCmdCopyBuffer);
    X(vkCmdCopyBufferToImage);
    X(vkCmdCopyImage);
    X(vkCmdCopyImageToBuffer);
    X(vkCmdCopyQueryPoolResults);
    X(vkCmdDispatch);
    X(vkCmdDispatchIndirect);
    X(vkCmdDraw);
    X(vkCmdDrawIndexed);
    X(vkCmdDrawIndirect);
    X(vkCmdDrawIndexedIndirect);
    X(vkCmdDrawIndirectCount);
    X(vkCmdDrawIndexedIndirectCount);
    X(vkCmdDrawIndirectByteCountEXT);
    X(vkCmdDrawMeshTasksEXT);
    X(vkCmdEndConditionalRenderingEXT);
    X(vkCmdEndQuery);
    X(vkCmdEndRenderPass);
    X(vkCmdEndTransformFeedbackEXT);
    X(vkCmdEndDebugUtilsLabelEXT);
    X(vkCmdFillBuffer);
    X(vkCmdPipelineBarrier);
    X(vkCmdPipelineBarrier2);
    X(vkCmdPushConstants);
    X(vkCmdPushDescriptorSetWithTemplateKHR);
    X(vkCmdSetBlendConstants);
    X(vkCmdSetDepthBias);
    X(vkCmdSetDepthBias2EXT);
    X(vkCmdSetDepthBounds);
    X(vkCmdSetEvent);
    X(vkCmdSetScissor);
    X(vkCmdSetStencilCompareMask);
    X(vkCmdSetStencilReference);
    X(vkCmdSetStencilWriteMask);
    X(vkCmdSetViewport);
    X(vkCmdWaitEvents);
    X(vkCmdBindVertexBuffers2EXT);
    X(vkCmdSetCullModeEXT);
    X(vkCmdSetDepthBoundsTestEnableEXT);
    X(vkCmdSetDepthCompareOpEXT);
    X(vkCmdSetDepthTestEnableEXT);
    X(vkCmdSetDepthWriteEnableEXT);
    X(vkCmdSetPrimitiveRestartEnableEXT);
    X(vkCmdSetRasterizerDiscardEnableEXT);
    X(vkCmdSetAlphaToCoverageEnableEXT);
    X(vkCmdSetAlphaToOneEnableEXT);
    X(vkCmdSetConservativeRasterizationModeEXT);
    X(vkCmdSetLineRasterizationModeEXT);
    X(vkCmdSetLineStippleEnableEXT);
    X(vkCmdSetDepthBiasEnableEXT);
    X(vkCmdSetLogicOpEnableEXT);
    X(vkCmdSetDepthClampEnableEXT);
    X(vkCmdSetFrontFaceEXT);
    X(vkCmdSetLogicOpEXT);
    X(vkCmdSetPatchControlPointsEXT);
    X(vkCmdSetLineWidth);
    X(vkCmdSetPrimitiveTopologyEXT);
    X(vkCmdSetStencilOpEXT);
    X(vkCmdSetStencilTestEnableEXT);
    X(vkCmdSetVertexInputEXT);
    X(vkCmdSetColorWriteMaskEXT);
    X(vkCmdSetColorWriteEnableEXT);
    X(vkCmdSetColorBlendEnableEXT);
    X(vkCmdSetColorBlendEquationEXT);
    X(vkCmdResetQueryPool);
    X(vkCmdWriteTimestamp);
    X(vkCmdResolveImage);
    X(vkCreateBuffer);
    X(vkCreateBufferView);
    X(vkCreateCommandPool);
    X(vkCreateComputePipelines);
    X(vkCreateDescriptorPool);
    X(vkCreateDescriptorSetLayout);
    X(vkCreateDescriptorUpdateTemplate);
    X(vkCreateEvent);
    X(vkCreateFence);
    X(vkCreateFramebuffer);
    X(vkCreateGraphicsPipelines);
    X(vkCreateImage);
    X(vkCreateImageView);
    X(vkCreatePipelineCache);
    X(vkCreatePipelineLayout);
    X(vkCreateQueryPool);
    X(vkCreateRenderPass);
    X(vkCreateRenderPass2);
    X(vkCreateSampler);
    X(vkCreateSemaphore);
    X(vkCreateShaderModule);
    X(vkCreateSwapchainKHR);
    X(vkDestroyBuffer);
    X(vkDestroyBufferView);
    X(vkDestroyCommandPool);
    X(vkDestroyDescriptorPool);
    X(vkDestroyDescriptorSetLayout);
    X(vkDestroyDescriptorUpdateTemplate);
    X(vkDestroyEvent);
    X(vkDestroyFence);
    X(vkDestroyFramebuffer);
    X(vkDestroyImage);
    X(vkDestroyImageView);
    X(vkDestroyPipeline);
    X(vkDestroyPipelineCache);
    X(vkDestroyPipelineLayout);
    X(vkDestroyQueryPool);
    X(vkDestroyRenderPass);
    X(vkDestroySampler);
    X(vkDestroySemaphore);
    X(vkDestroyShaderModule);
    X(vkDestroySwapchainKHR);
    X(vkDeviceWaitIdle);
    X(vkEndCommandBuffer);
    X(vkFreeCommandBuffers);
    X(vkFreeDescriptorSets);
    X(vkFreeMemory);
    X(vkGetBufferMemoryRequirements2);
    X(vkGetDeviceQueue);
    X(vkGetEventStatus);
    X(vkGetFenceStatus);
    X(vkGetImageMemoryRequirements);
    X(vkGetImageSparseMemoryRequirements);
    X(vkGetPipelineCacheData);
    X(vkGetMemoryFdKHR);
#ifdef _WIN32
    X(vkGetMemoryWin32HandleKHR);
#endif
    X(vkGetQueryPoolResults);
    X(vkGetDeviceFaultInfoEXT);
    X(vkGetPipelineExecutablePropertiesKHR);
    X(vkGetPipelineExecutableStatisticsKHR);
    X(vkGetSemaphoreCounterValue);
    X(vkMapMemory);
    X(vkQueueBindSparse);
    X(vkQueueSubmit);
    X(vkQueueSubmit2);
    X(vkResetFences);
    X(vkResetQueryPool);
    X(vkSetDebugUtilsObjectNameEXT);
    X(vkSetDebugUtilsObjectTagEXT);
    X(vkUnmapMemory);
    X(vkUpdateDescriptorSetWithTemplate);
    X(vkUpdateDescriptorSets);
    X(vkGetBufferDeviceAddress);
    X(vkGetDescriptorSetLayoutSizeEXT);
    X(vkGetDescriptorSetLayoutBindingOffsetEXT);
    X(vkGetDescriptorEXT);
    X(vkCmdBindDescriptorBuffersEXT);
    X(vkCmdSetDescriptorBufferOffsetsEXT);
    X(vkWaitForFences);
    X(vkWaitSemaphores);

    // Support for timeline semaphores is mandatory in Vulkan 1.2
    if (!dld.vkGetSemaphoreCounterValue) {
        Proc(dld.vkGetSemaphoreCounterValue, dld, "vkGetSemaphoreCounterValueKHR", device);
        Proc(dld.vkWaitSemaphores, dld, "vkWaitSemaphoresKHR", device);
    }

    // Support for host query reset is mandatory in Vulkan 1.2
    if (!dld.vkResetQueryPool) {
        Proc(dld.vkResetQueryPool, dld, "vkResetQueryPoolEXT", device);
    }

    // Support for draw indirect with count is optional in Vulkan 1.2
    if (!dld.vkCmdDrawIndirectCount) {
        Proc(dld.vkCmdDrawIndirectCount, dld, "vkCmdDrawIndirectCountKHR", device);
        Proc(dld.vkCmdDrawIndexedIndirectCount, dld, "vkCmdDrawIndexedIndirectCountKHR", device);
    }

    // Synchronization2 is core in Vulkan 1.3, otherwise requires VK_KHR_synchronization2
    if (!dld.vkCmdPipelineBarrier2) {
        Proc(dld.vkCmdPipelineBarrier2, dld, "vkCmdPipelineBarrier2KHR", device);
    }
    if (!dld.vkQueueSubmit2) {
        Proc(dld.vkQueueSubmit2, dld, "vkQueueSubmit2KHR", device);
    }

    if (!dld.vkCreateRenderPass2) {
        Proc(dld.vkCreateRenderPass2, dld, "vkCreateRenderPass2KHR", device);
    }
#undef X
}

template <typename T>
void SetObjectName(const DeviceDispatch* dld, VkDevice device, T handle, VkObjectType type,
                   const char* name) {
    diagnostic::Trace("SET_OBJECT_NAME", "device={:#x} object_type={} object={:#x} name={}",
                      diagnostic::HandleBits(device), static_cast<u32>(type),
                      diagnostic::HandleBits(handle), name ? name : "(null)");
    const VkDebugUtilsObjectNameInfoEXT name_info{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
        .pNext = nullptr,
        .objectType = type,
        .objectHandle = reinterpret_cast<u64>(handle),
        .pObjectName = name,
    };
    if (dld->vkSetDebugUtilsObjectNameEXT) {
        Check(dld->vkSetDebugUtilsObjectNameEXT(device, &name_info));
    }
}

} // Anonymous namespace

bool Load(InstanceDispatch& dld) noexcept {
#define X(name) Proc(dld.name, dld, #name)
    return X(vkCreateInstance) && X(vkEnumerateInstanceExtensionProperties) &&
           X(vkEnumerateInstanceLayerProperties);
#undef X
}

bool Load(VkInstance instance, InstanceDispatch& dld) noexcept {
#define X(name) Proc(dld.name, dld, #name, instance)
    // These functions may fail to load depending on the enabled extensions.
    // Don't return a failure on these.
    X(vkCreateDebugUtilsMessengerEXT);
    X(vkCreateDebugReportCallbackEXT);
    X(vkDestroyDebugUtilsMessengerEXT);
    X(vkDestroyDebugReportCallbackEXT);
    X(vkDestroySurfaceKHR);
    X(vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR);
    X(vkGetPhysicalDeviceFeatures2);
    X(vkGetPhysicalDeviceFormatProperties2);
    X(vkGetPhysicalDeviceProperties2);
    X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    X(vkGetPhysicalDeviceSurfaceFormatsKHR);
    X(vkGetPhysicalDeviceSurfacePresentModesKHR);
    X(vkGetPhysicalDeviceSurfaceSupportKHR);
    X(vkGetPhysicalDeviceToolProperties);
    X(vkGetSwapchainImagesKHR);
    X(vkQueuePresentKHR);

    return X(vkCreateDevice) && X(vkDestroyDevice) && X(vkDestroyDevice) &&
           X(vkEnumerateDeviceExtensionProperties) && X(vkEnumeratePhysicalDevices) &&
           X(vkGetDeviceProcAddr) && X(vkGetPhysicalDeviceFormatProperties) &&
           X(vkGetPhysicalDeviceMemoryProperties) && X(vkGetPhysicalDeviceMemoryProperties2) &&
           X(vkGetPhysicalDeviceProperties) && X(vkGetPhysicalDeviceQueueFamilyProperties);
#undef X
}

const char* Exception::what() const noexcept {
    return string_VkResult(result);
}

void Destroy(VkInstance instance, const InstanceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_INSTANCE", "instance={:#x}", diagnostic::HandleBits(instance));
    dld.vkDestroyInstance(instance, nullptr);
}

void Destroy(VkDevice device, const InstanceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_DEVICE", "device={:#x}", diagnostic::HandleBits(device));
    dld.vkDestroyDevice(device, nullptr);
}

void Destroy(VkDevice device, VkBuffer handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_BUFFER", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyBuffer(device, handle, nullptr);
}

void Destroy(VkDevice device, VkBufferView handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_BUFFER_VIEW", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyBufferView(device, handle, nullptr);
}

void Destroy(VkDevice device, VkCommandPool handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_COMMAND_POOL", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyCommandPool(device, handle, nullptr);
}

void Destroy(VkDevice device, VkDescriptorPool handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_DESCRIPTOR_POOL", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyDescriptorPool(device, handle, nullptr);
}

void Destroy(VkDevice device, VkDescriptorSetLayout handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_DESCRIPTOR_SET_LAYOUT", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyDescriptorSetLayout(device, handle, nullptr);
}

void Destroy(VkDevice device, VkDescriptorUpdateTemplate handle,
             const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_DESCRIPTOR_UPDATE_TEMPLATE", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyDescriptorUpdateTemplate(device, handle, nullptr);
}

void Destroy(VkDevice device, VkDeviceMemory handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("FREE_DEVICE_MEMORY", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkFreeMemory(device, handle, nullptr);
}

void Destroy(VkDevice device, VkEvent handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_EVENT", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyEvent(device, handle, nullptr);
}

void Destroy(VkDevice device, VkFence handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_FENCE", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyFence(device, handle, nullptr);
}

void Destroy(VkDevice device, VkFramebuffer handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_FRAMEBUFFER", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyFramebuffer(device, handle, nullptr);
}

void Destroy(VkDevice device, VkImage handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_IMAGE", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyImage(device, handle, nullptr);
}

void Destroy(VkDevice device, VkImageView handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_IMAGE_VIEW", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyImageView(device, handle, nullptr);
}

void Destroy(VkDevice device, VkPipeline handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_PIPELINE", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyPipeline(device, handle, nullptr);
}

void Destroy(VkDevice device, VkPipelineCache handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_PIPELINE_CACHE", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyPipelineCache(device, handle, nullptr);
}

void Destroy(VkDevice device, VkPipelineLayout handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_PIPELINE_LAYOUT", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyPipelineLayout(device, handle, nullptr);
}

void Destroy(VkDevice device, VkQueryPool handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_QUERY_POOL", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyQueryPool(device, handle, nullptr);
}

void Destroy(VkDevice device, VkRenderPass handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_RENDER_PASS", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyRenderPass(device, handle, nullptr);
}

void Destroy(VkDevice device, VkSampler handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_SAMPLER", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroySampler(device, handle, nullptr);
}

void Destroy(VkDevice device, VkSwapchainKHR handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_SWAPCHAIN", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroySwapchainKHR(device, handle, nullptr);
}

void Destroy(VkDevice device, VkSemaphore handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_SEMAPHORE", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroySemaphore(device, handle, nullptr);
}

void Destroy(VkDevice device, VkShaderModule handle, const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_SHADER_MODULE", "device={:#x} object={:#x}", diagnostic::HandleBits(device), diagnostic::HandleBits(handle));
    dld.vkDestroyShaderModule(device, handle, nullptr);
}

void Destroy(VkInstance instance, VkDebugUtilsMessengerEXT handle,
             const InstanceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_DEBUG_UTILS_MESSENGER", "instance={:#x} object={:#x}", diagnostic::HandleBits(instance), diagnostic::HandleBits(handle));
    dld.vkDestroyDebugUtilsMessengerEXT(instance, handle, nullptr);
}

void Destroy(VkInstance instance, VkDebugReportCallbackEXT handle,
             const InstanceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_DEBUG_REPORT_CALLBACK", "instance={:#x} object={:#x}", diagnostic::HandleBits(instance), diagnostic::HandleBits(handle));
    dld.vkDestroyDebugReportCallbackEXT(instance, handle, nullptr);
}

void Destroy(VkInstance instance, VkSurfaceKHR handle, const InstanceDispatch& dld) noexcept {
    diagnostic::Trace("DESTROY_SURFACE", "instance={:#x} object={:#x}", diagnostic::HandleBits(instance), diagnostic::HandleBits(handle));
    dld.vkDestroySurfaceKHR(instance, handle, nullptr);
}

VkResult Free(VkDevice device, VkDescriptorPool handle, Span<VkDescriptorSet> sets,
              const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("FREE_DESCRIPTOR_SETS", "device={:#x} pool={:#x} count={}",
                      diagnostic::HandleBits(device), diagnostic::HandleBits(handle), sets.size());
    for (u32 i = 0; i < sets.size(); ++i) {
        diagnostic::Trace("FREE_DESCRIPTOR_SET", "pool={:#x} index={} set={:#x}",
                          diagnostic::HandleBits(handle), i, diagnostic::HandleBits(sets[i]));
    }
    const VkResult result = dld.vkFreeDescriptorSets(device, handle, sets.size(), sets.data());
    diagnostic::Trace("FREE_DESCRIPTOR_SETS_RESULT", "pool={:#x} result={}",
                      diagnostic::HandleBits(handle), static_cast<s32>(result));
    return result;
}

VkResult Free(VkDevice device, VkCommandPool handle, Span<VkCommandBuffer> buffers,
              const DeviceDispatch& dld) noexcept {
    diagnostic::Trace("FREE_COMMAND_BUFFERS", "device={:#x} pool={:#x} count={}",
                      diagnostic::HandleBits(device), diagnostic::HandleBits(handle), buffers.size());
    for (u32 i = 0; i < buffers.size(); ++i) {
        diagnostic::Trace("FREE_COMMAND_BUFFER", "pool={:#x} index={} command_buffer={:#x}",
                          diagnostic::HandleBits(handle), i, diagnostic::HandleBits(buffers[i]));
    }
    dld.vkFreeCommandBuffers(device, handle, buffers.size(), buffers.data());
    return VK_SUCCESS;
}

Instance Instance::Create(u32 version, Span<const char*> layers, Span<const char*> extensions,
                          InstanceDispatch& dispatch) {
#ifdef __APPLE__
    constexpr VkFlags ci_flags{VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR};
#else
    constexpr VkFlags ci_flags{};
#endif
    // DO NOT TOUCH, breaks RNDA3!!
	// Don't know why, but gloom + yellow line glitch appears
    const VkApplicationInfo application_info{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = nullptr,
        .pApplicationName = "yuzu Emulator",
        .applicationVersion = VK_MAKE_VERSION(1, 3, 0),
        .pEngineName = "yuzu Emulator",
        .engineVersion = VK_MAKE_VERSION(1, 3, 0),
        .apiVersion = VK_API_VERSION_1_3,
    };
    const VkInstanceCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = ci_flags,
        .pApplicationInfo = &application_info,
        .enabledLayerCount = layers.size(),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = extensions.size(),
        .ppEnabledExtensionNames = extensions.data(),
    };
    VkInstance instance;
    Check(dispatch.vkCreateInstance(&ci, nullptr, &instance));
    if (!Proc(dispatch.vkDestroyInstance, dispatch, "vkDestroyInstance", instance)) {
        // We successfully created an instance but the destroy function couldn't be loaded.
        // This is a good moment to panic.
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }
    return Instance(instance, dispatch);
}

std::vector<VkPhysicalDevice> Instance::EnumeratePhysicalDevices() const {
    u32 num;
    Check(dld->vkEnumeratePhysicalDevices(handle, &num, nullptr));
    std::vector<VkPhysicalDevice> physical_devices(num);
    Check(dld->vkEnumeratePhysicalDevices(handle, &num, physical_devices.data()));
    SortPhysicalDevices(physical_devices, *dld);
    return physical_devices;
}

DebugUtilsMessenger Instance::CreateDebugUtilsMessenger(
    const VkDebugUtilsMessengerCreateInfoEXT& create_info) const {
    VkDebugUtilsMessengerEXT object;
    Check(dld->vkCreateDebugUtilsMessengerEXT(handle, &create_info, nullptr, &object));
    return DebugUtilsMessenger(object, handle, *dld);
}

DebugReportCallback Instance::CreateDebugReportCallback(
    const VkDebugReportCallbackCreateInfoEXT& create_info) const {
    VkDebugReportCallbackEXT object;
    Check(dld->vkCreateDebugReportCallbackEXT(handle, &create_info, nullptr, &object));
    return DebugReportCallback(object, handle, *dld);
}

void Image::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_IMAGE, name);
}

void Image::Release() const noexcept {
    if (handle) {
        diagnostic::Trace("RELEASE_VMA_IMAGE", "image={:#x} allocation={:#x}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(allocation));
        vmaDestroyImage(allocator, handle, allocation);
    }
}

Image::MemoryBlock Image::GetMemoryBlock() const noexcept {
    if (allocation == nullptr) {
        return {};
    }
    VmaAllocationInfo2 info{};
    vmaGetAllocationInfo2(allocator, allocation, &info);
    if (info.dedicatedMemory) {
        return {0, info.blockSize, info.allocationInfo.size};
    }
    return {reinterpret_cast<u64>(info.allocationInfo.deviceMemory), info.blockSize,
            info.allocationInfo.size};
}

void Buffer::Flush() const {
    if (!is_coherent) {
        const VkResult result = vmaFlushAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
        diagnostic::Trace("BUFFER_FLUSH", "buffer={:#x} allocation={:#x} result={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(allocation),
                          static_cast<s32>(result));
    }
}

void Buffer::Invalidate() const {
    if (!is_coherent) {
        const VkResult result = vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
        diagnostic::Trace("BUFFER_INVALIDATE", "buffer={:#x} allocation={:#x} result={}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(allocation),
                          static_cast<s32>(result));
    }
}

void Buffer::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_BUFFER, name);
}

void Buffer::Release() const noexcept {
    if (handle) {
        diagnostic::Trace("RELEASE_VMA_BUFFER", "buffer={:#x} allocation={:#x}",
                          diagnostic::HandleBits(handle), diagnostic::HandleBits(allocation));
        vmaDestroyBuffer(allocator, handle, allocation);
    }
}

void BufferView::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_BUFFER_VIEW, name);
}

void ImageView::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_IMAGE_VIEW, name);
}

int DeviceMemory::GetMemoryFdKHR() const {
    const VkMemoryGetFdInfoKHR get_fd_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .pNext = nullptr,
        .memory = handle,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT_KHR,
    };
    int fd;
    Check(dld->vkGetMemoryFdKHR(owner, &get_fd_info, &fd));
    return fd;
}

#ifdef _WIN32
HANDLE DeviceMemory::GetMemoryWin32HandleKHR() const {
    const VkMemoryGetWin32HandleInfoKHR get_win32_handle_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
        .pNext = nullptr,
        .memory = handle,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR,
    };
    HANDLE win32_handle;
    Check(dld->vkGetMemoryWin32HandleKHR(owner, &get_win32_handle_info, &win32_handle));
    return win32_handle;
}
#endif

void DeviceMemory::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_DEVICE_MEMORY, name);
}

void Fence::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_FENCE, name);
}

void Framebuffer::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_FRAMEBUFFER, name);
}

DescriptorSets DescriptorPool::Allocate(const VkDescriptorSetAllocateInfo& ai) const {
    const std::size_t num = ai.descriptorSetCount;
    std::unique_ptr sets = std::make_unique<VkDescriptorSet[]>(num);
    const VkResult result = dld->vkAllocateDescriptorSets(owner, &ai, sets.get());
    diagnostic::Trace("ALLOCATE_DESCRIPTOR_SETS", "pool={:#x} layout_count={} count={} result={}",
                      diagnostic::HandleBits(handle), ai.descriptorSetCount, num,
                      static_cast<s32>(result));
    if (result == VK_SUCCESS) {
        for (std::size_t i = 0; i < num; ++i) {
            diagnostic::Trace("ALLOCATED_DESCRIPTOR_SET", "pool={:#x} index={} set={:#x}",
                              diagnostic::HandleBits(handle), i, diagnostic::HandleBits(sets[i]));
        }
        return DescriptorSets(std::move(sets), num, owner, handle, *dld);
    }
    if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
        return {};
    }
    throw Exception(result);
}

void DescriptorPool::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_DESCRIPTOR_POOL, name);
}

CommandBuffers CommandPool::Allocate(std::size_t num_buffers, VkCommandBufferLevel level) const {
    const VkCommandBufferAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = handle,
        .level = level,
        .commandBufferCount = static_cast<u32>(num_buffers),
    };

    std::unique_ptr buffers = std::make_unique<VkCommandBuffer[]>(num_buffers);
    const VkResult result = dld->vkAllocateCommandBuffers(owner, &ai, buffers.get());
    diagnostic::Trace("ALLOCATE_COMMAND_BUFFERS", "pool={:#x} count={} level={} result={}",
                      diagnostic::HandleBits(handle), num_buffers, static_cast<u32>(level),
                      static_cast<s32>(result));
    if (result == VK_SUCCESS) {
        for (std::size_t i = 0; i < num_buffers; ++i) {
            diagnostic::Trace("ALLOCATED_COMMAND_BUFFER", "pool={:#x} index={} command_buffer={:#x}",
                              diagnostic::HandleBits(handle), i, diagnostic::HandleBits(buffers[i]));
        }
        return CommandBuffers(std::move(buffers), num_buffers, owner, handle, *dld);
    }
    if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
        return {};
    }
    throw Exception(result);
}

void CommandPool::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_COMMAND_POOL, name);
}

std::vector<VkImage> SwapchainKHR::GetImages() const {
    u32 num;
    Check(dld->vkGetSwapchainImagesKHR(owner, handle, &num, nullptr));
    std::vector<VkImage> images(num);
    Check(dld->vkGetSwapchainImagesKHR(owner, handle, &num, images.data()));
    diagnostic::Trace("SWAPCHAIN_IMAGES", "swapchain={:#x} image_count={}",
                      diagnostic::HandleBits(handle), num);
    for (u32 i = 0; i < num; ++i) {
        diagnostic::Trace("SWAPCHAIN_IMAGE", "swapchain={:#x} index={} image={:#x}",
                          diagnostic::HandleBits(handle), i, diagnostic::HandleBits(images[i]));
    }
    return images;
}

void Event::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_EVENT, name);
}

void ShaderModule::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_SHADER_MODULE, name);
}

void PipelineCache::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_PIPELINE_CACHE, name);
}

void Semaphore::SetObjectNameEXT(const char* name) const {
    SetObjectName(dld, owner, handle, VK_OBJECT_TYPE_SEMAPHORE, name);
}

Device Device::Create(VkPhysicalDevice physical_device, Span<VkDeviceQueueCreateInfo> queues_ci,
                      Span<const char*> enabled_extensions, const void* next,
                      DeviceDispatch& dispatch) {
    const VkDeviceCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = next,
        .flags = 0,
        .queueCreateInfoCount = queues_ci.size(),
        .pQueueCreateInfos = queues_ci.data(),
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = enabled_extensions.size(),
        .ppEnabledExtensionNames = enabled_extensions.data(),
        .pEnabledFeatures = nullptr,
    };
    VkDevice device;
    Check(dispatch.vkCreateDevice(physical_device, &ci, nullptr, &device));
    diagnostic::Trace("CREATE_DEVICE", "device={:#x} physical_device={:#x} queue_infos={} extensions={}",
                      diagnostic::HandleBits(device), diagnostic::HandleBits(physical_device),
                      queues_ci.size(), enabled_extensions.size());
    for (u32 i = 0; i < queues_ci.size(); ++i) {
        const auto& queue = queues_ci[i];
        diagnostic::Trace("DEVICE_QUEUE_CREATE_INFO",
                          "device={:#x} index={} family={} count={} flags={:#x}",
                          diagnostic::HandleBits(device), i, queue.queueFamilyIndex,
                          queue.queueCount, static_cast<u32>(queue.flags));
    }
    for (u32 i = 0; i < enabled_extensions.size(); ++i) {
        diagnostic::Trace("DEVICE_EXTENSION", "device={:#x} index={} name={}",
                          diagnostic::HandleBits(device), i,
                          enabled_extensions[i] ? enabled_extensions[i] : "(null)");
    }
    Load(device, dispatch);
    return Device(device, dispatch);
}

Queue Device::GetQueue(u32 family_index) const noexcept {
    VkQueue queue;
    dld->vkGetDeviceQueue(handle, family_index, 0, &queue);
    diagnostic::Trace("GET_DEVICE_QUEUE", "device={:#x} family={} index=0 queue={:#x}",
                      diagnostic::HandleBits(handle), family_index, diagnostic::HandleBits(queue));
    return Queue(queue, *dld);
}

BufferView Device::CreateBufferView(const VkBufferViewCreateInfo& ci) const {
    VkBufferView object;
    Check(dld->vkCreateBufferView(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_BUFFER_VIEW",
                      "view={:#x} buffer={:#x} format={} offset={} range={}",
                      diagnostic::HandleBits(object), diagnostic::HandleBits(ci.buffer),
                      static_cast<s32>(ci.format), ci.offset, ci.range);
    return BufferView(object, handle, *dld);
}

ImageView Device::CreateImageView(const VkImageViewCreateInfo& ci) const {
    VkImageView object;
    Check(dld->vkCreateImageView(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_IMAGE_VIEW",
                      "view={:#x} image={:#x} view_type={} format={} components=({},{},{},{}) "
                      "aspect={:#x} base_mip={} mip_count={} base_layer={} layer_count={}",
                      diagnostic::HandleBits(object), diagnostic::HandleBits(ci.image),
                      static_cast<u32>(ci.viewType), static_cast<s32>(ci.format),
                      static_cast<u32>(ci.components.r), static_cast<u32>(ci.components.g),
                      static_cast<u32>(ci.components.b), static_cast<u32>(ci.components.a),
                      static_cast<u32>(ci.subresourceRange.aspectMask),
                      ci.subresourceRange.baseMipLevel, ci.subresourceRange.levelCount,
                      ci.subresourceRange.baseArrayLayer, ci.subresourceRange.layerCount);
    return ImageView(object, handle, *dld);
}

Semaphore Device::CreateSemaphore() const {
    static constexpr VkSemaphoreCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    return CreateSemaphore(ci);
}

Semaphore Device::CreateSemaphore(const VkSemaphoreCreateInfo& ci) const {
    VkSemaphore object;
    Check(dld->vkCreateSemaphore(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_SEMAPHORE", "semaphore={:#x} flags={:#x}",
                      diagnostic::HandleBits(object), static_cast<u32>(ci.flags));
    return Semaphore(object, handle, *dld);
}

Fence Device::CreateFence(const VkFenceCreateInfo& ci) const {
    VkFence object;
    Check(dld->vkCreateFence(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_FENCE", "fence={:#x} flags={:#x}",
                      diagnostic::HandleBits(object), static_cast<u32>(ci.flags));
    return Fence(object, handle, *dld);
}

DescriptorPool Device::CreateDescriptorPool(const VkDescriptorPoolCreateInfo& ci) const {
    VkDescriptorPool object;
    Check(dld->vkCreateDescriptorPool(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_DESCRIPTOR_POOL",
                      "pool={:#x} flags={:#x} max_sets={} pool_size_count={}",
                      diagnostic::HandleBits(object), static_cast<u32>(ci.flags),
                      ci.maxSets, ci.poolSizeCount);
    for (u32 i = 0; i < ci.poolSizeCount; ++i) {
        diagnostic::Trace("DESCRIPTOR_POOL_SIZE", "pool={:#x} index={} type={} count={}",
                          diagnostic::HandleBits(object), i,
                          static_cast<u32>(ci.pPoolSizes[i].type),
                          ci.pPoolSizes[i].descriptorCount);
    }
    return DescriptorPool(object, handle, *dld);
}

RenderPass Device::CreateRenderPass(const VkRenderPassCreateInfo& ci) const {
    VkRenderPass object;
    Check(dld->vkCreateRenderPass(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_RENDER_PASS",
                      "render_pass={:#x} attachments={} subpasses={} dependencies={}",
                      diagnostic::HandleBits(object), ci.attachmentCount, ci.subpassCount,
                      ci.dependencyCount);
    for (u32 i = 0; i < ci.attachmentCount; ++i) {
        const auto& a = ci.pAttachments[i];
        diagnostic::Trace("RENDER_PASS_ATTACHMENT",
                          "render_pass={:#x} index={} format={} samples={} load={} store={} "
                          "stencil_load={} stencil_store={} initial_layout={} final_layout={}",
                          diagnostic::HandleBits(object), i, static_cast<s32>(a.format),
                          static_cast<u32>(a.samples), static_cast<u32>(a.loadOp),
                          static_cast<u32>(a.storeOp), static_cast<u32>(a.stencilLoadOp),
                          static_cast<u32>(a.stencilStoreOp), static_cast<u32>(a.initialLayout),
                          static_cast<u32>(a.finalLayout));
    }
    for (u32 i = 0; i < ci.subpassCount; ++i) {
        const auto& subpass = ci.pSubpasses[i];
        diagnostic::Trace("RENDER_PASS_SUBPASS",
                          "render_pass={:#x} index={} bind_point={} inputs={} colors={} resolves={} "
                          "preserves={} has_depth={}",
                          diagnostic::HandleBits(object), i,
                          static_cast<u32>(subpass.pipelineBindPoint),
                          subpass.inputAttachmentCount, subpass.colorAttachmentCount,
                          subpass.pResolveAttachments != nullptr,
                          subpass.preserveAttachmentCount,
                          subpass.pDepthStencilAttachment != nullptr);
        for (u32 j = 0; j < subpass.inputAttachmentCount; ++j) {
            const auto& ref = subpass.pInputAttachments[j];
            diagnostic::Trace("RENDER_PASS_INPUT_REF",
                              "render_pass={:#x} subpass={} index={} attachment={} layout={}",
                              diagnostic::HandleBits(object), i, j, ref.attachment,
                              static_cast<u32>(ref.layout));
        }
        for (u32 j = 0; j < subpass.colorAttachmentCount; ++j) {
            const auto& ref = subpass.pColorAttachments[j];
            diagnostic::Trace("RENDER_PASS_COLOR_REF",
                              "render_pass={:#x} subpass={} index={} attachment={} layout={}",
                              diagnostic::HandleBits(object), i, j, ref.attachment,
                              static_cast<u32>(ref.layout));
            if (subpass.pResolveAttachments) {
                const auto& resolve = subpass.pResolveAttachments[j];
                diagnostic::Trace("RENDER_PASS_RESOLVE_REF",
                                  "render_pass={:#x} subpass={} index={} attachment={} layout={}",
                                  diagnostic::HandleBits(object), i, j, resolve.attachment,
                                  static_cast<u32>(resolve.layout));
            }
        }
        if (subpass.pDepthStencilAttachment) {
            const auto& ref = *subpass.pDepthStencilAttachment;
            diagnostic::Trace("RENDER_PASS_DEPTH_REF",
                              "render_pass={:#x} subpass={} attachment={} layout={}",
                              diagnostic::HandleBits(object), i, ref.attachment,
                              static_cast<u32>(ref.layout));
        }
        for (u32 j = 0; j < subpass.preserveAttachmentCount; ++j) {
            diagnostic::Trace("RENDER_PASS_PRESERVE_REF",
                              "render_pass={:#x} subpass={} index={} attachment={}",
                              diagnostic::HandleBits(object), i, j,
                              subpass.pPreserveAttachments[j]);
        }
    }
    for (u32 i = 0; i < ci.dependencyCount; ++i) {
        const auto& d = ci.pDependencies[i];
        diagnostic::Trace("RENDER_PASS_DEPENDENCY",
                          "render_pass={:#x} index={} src_subpass={} dst_subpass={} "
                          "src_stage={:#x} dst_stage={:#x} src_access={:#x} dst_access={:#x} flags={:#x}",
                          diagnostic::HandleBits(object), i, d.srcSubpass, d.dstSubpass,
                          static_cast<u64>(d.srcStageMask), static_cast<u64>(d.dstStageMask),
                          static_cast<u64>(d.srcAccessMask), static_cast<u64>(d.dstAccessMask),
                          static_cast<u32>(d.dependencyFlags));
    }
    return RenderPass(object, handle, *dld);
}

RenderPass Device::CreateRenderPass2(const VkRenderPassCreateInfo2& ci) const {
    VkRenderPass object;
    Check(dld->vkCreateRenderPass2(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_RENDER_PASS2",
                      "render_pass={:#x} flags={:#x} attachments={} subpasses={} dependencies={}",
                      diagnostic::HandleBits(object), static_cast<u32>(ci.flags),
                      ci.attachmentCount, ci.subpassCount, ci.dependencyCount);
    for (u32 i = 0; i < ci.attachmentCount; ++i) {
        const auto& a = ci.pAttachments[i];
        diagnostic::Trace("RENDER_PASS2_ATTACHMENT",
                          "render_pass={:#x} index={} flags={:#x} format={} samples={} load={} store={} "
                          "stencil_load={} stencil_store={} initial_layout={} final_layout={}",
                          diagnostic::HandleBits(object), i, static_cast<u32>(a.flags),
                          static_cast<s32>(a.format), static_cast<u32>(a.samples),
                          static_cast<u32>(a.loadOp), static_cast<u32>(a.storeOp),
                          static_cast<u32>(a.stencilLoadOp), static_cast<u32>(a.stencilStoreOp),
                          static_cast<u32>(a.initialLayout), static_cast<u32>(a.finalLayout));
    }
    for (u32 i = 0; i < ci.subpassCount; ++i) {
        const auto& subpass = ci.pSubpasses[i];
        diagnostic::Trace("RENDER_PASS2_SUBPASS",
                          "render_pass={:#x} index={} flags={:#x} bind_point={} view_mask={} "
                          "inputs={} colors={} has_resolve={} preserves={} has_depth={}",
                          diagnostic::HandleBits(object), i, static_cast<u32>(subpass.flags),
                          static_cast<u32>(subpass.pipelineBindPoint), subpass.viewMask,
                          subpass.inputAttachmentCount, subpass.colorAttachmentCount,
                          subpass.pResolveAttachments != nullptr, subpass.preserveAttachmentCount,
                          subpass.pDepthStencilAttachment != nullptr);
        for (u32 j = 0; j < subpass.inputAttachmentCount; ++j) {
            const auto& ref = subpass.pInputAttachments[j];
            diagnostic::Trace("RENDER_PASS2_INPUT_REF",
                              "render_pass={:#x} subpass={} index={} attachment={} layout={} aspect={:#x}",
                              diagnostic::HandleBits(object), i, j, ref.attachment,
                              static_cast<u32>(ref.layout), static_cast<u32>(ref.aspectMask));
        }
        for (u32 j = 0; j < subpass.colorAttachmentCount; ++j) {
            const auto& ref = subpass.pColorAttachments[j];
            diagnostic::Trace("RENDER_PASS2_COLOR_REF",
                              "render_pass={:#x} subpass={} index={} attachment={} layout={} aspect={:#x}",
                              diagnostic::HandleBits(object), i, j, ref.attachment,
                              static_cast<u32>(ref.layout), static_cast<u32>(ref.aspectMask));
            if (subpass.pResolveAttachments) {
                const auto& resolve = subpass.pResolveAttachments[j];
                diagnostic::Trace("RENDER_PASS2_RESOLVE_REF",
                                  "render_pass={:#x} subpass={} index={} attachment={} layout={} aspect={:#x}",
                                  diagnostic::HandleBits(object), i, j, resolve.attachment,
                                  static_cast<u32>(resolve.layout), static_cast<u32>(resolve.aspectMask));
            }
        }
        if (subpass.pDepthStencilAttachment) {
            const auto& ref = *subpass.pDepthStencilAttachment;
            diagnostic::Trace("RENDER_PASS2_DEPTH_REF",
                              "render_pass={:#x} subpass={} attachment={} layout={} aspect={:#x}",
                              diagnostic::HandleBits(object), i, ref.attachment,
                              static_cast<u32>(ref.layout), static_cast<u32>(ref.aspectMask));
        }
        for (u32 j = 0; j < subpass.preserveAttachmentCount; ++j) {
            diagnostic::Trace("RENDER_PASS2_PRESERVE_REF",
                              "render_pass={:#x} subpass={} index={} attachment={}",
                              diagnostic::HandleBits(object), i, j,
                              subpass.pPreserveAttachments[j]);
        }
    }
    for (u32 i = 0; i < ci.dependencyCount; ++i) {
        const auto& d = ci.pDependencies[i];
        diagnostic::Trace("RENDER_PASS2_DEPENDENCY",
                          "render_pass={:#x} index={} src_subpass={} dst_subpass={} src_stage={:#x} "
                          "dst_stage={:#x} src_access={:#x} dst_access={:#x} flags={:#x} view_offset={}",
                          diagnostic::HandleBits(object), i, d.srcSubpass, d.dstSubpass,
                          static_cast<u64>(d.srcStageMask), static_cast<u64>(d.dstStageMask),
                          static_cast<u64>(d.srcAccessMask), static_cast<u64>(d.dstAccessMask),
                          static_cast<u32>(d.dependencyFlags), d.viewOffset);
    }
    return RenderPass(object, handle, *dld);
}

DescriptorSetLayout Device::CreateDescriptorSetLayout(
    const VkDescriptorSetLayoutCreateInfo& ci) const {
    VkDescriptorSetLayout object;
    Check(dld->vkCreateDescriptorSetLayout(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_DESCRIPTOR_SET_LAYOUT", "layout={:#x} flags={:#x} bindings={}",
                      diagnostic::HandleBits(object), static_cast<u32>(ci.flags), ci.bindingCount);
    for (u32 i = 0; i < ci.bindingCount; ++i) {
        const auto& b = ci.pBindings[i];
        diagnostic::Trace("DESCRIPTOR_LAYOUT_BINDING",
                          "layout={:#x} index={} binding={} type={} count={} stages={:#x} immutable_samplers={}",
                          diagnostic::HandleBits(object), i, b.binding,
                          static_cast<u32>(b.descriptorType), b.descriptorCount,
                          static_cast<u64>(b.stageFlags), b.pImmutableSamplers != nullptr);
    }
    return DescriptorSetLayout(object, handle, *dld);
}

PipelineCache Device::CreatePipelineCache(const VkPipelineCacheCreateInfo& ci) const {
    VkPipelineCache cache;
    Check(dld->vkCreatePipelineCache(handle, &ci, nullptr, &cache));
    diagnostic::Trace("CREATE_PIPELINE_CACHE", "cache={:#x} flags={:#x} initial_data_size={}",
                      diagnostic::HandleBits(cache), static_cast<u32>(ci.flags), ci.initialDataSize);
    return PipelineCache(cache, handle, *dld);
}

PipelineLayout Device::CreatePipelineLayout(const VkPipelineLayoutCreateInfo& ci) const {
    VkPipelineLayout object;
    Check(dld->vkCreatePipelineLayout(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_PIPELINE_LAYOUT",
                      "layout={:#x} flags={:#x} set_layouts={} push_constant_ranges={}",
                      diagnostic::HandleBits(object), static_cast<u32>(ci.flags),
                      ci.setLayoutCount, ci.pushConstantRangeCount);
    for (u32 i = 0; i < ci.pushConstantRangeCount; ++i) {
        const auto& range = ci.pPushConstantRanges[i];
        diagnostic::Trace("PUSH_CONSTANT_RANGE",
                          "layout={:#x} index={} stages={:#x} offset={} size={}",
                          diagnostic::HandleBits(object), i, static_cast<u32>(range.stageFlags),
                          range.offset, range.size);
    }
    for (u32 i = 0; i < ci.setLayoutCount; ++i) {
        diagnostic::Trace("PIPELINE_SET_LAYOUT", "layout={:#x} index={} set_layout={:#x}",
                          diagnostic::HandleBits(object), i,
                          diagnostic::HandleBits(ci.pSetLayouts[i]));
    }
    return PipelineLayout(object, handle, *dld);
}

Pipeline Device::CreateGraphicsPipeline(const VkGraphicsPipelineCreateInfo& ci, VkPipelineCache cache) const {
    VkPipeline object = VK_NULL_HANDLE;
    auto const result = dld->vkCreateGraphicsPipelines(handle, cache, 1, &ci, nullptr, &object);
    diagnostic::Trace("CREATE_GRAPHICS_PIPELINE",
                      "pipeline={:#x} result={} flags={:#x} stages={} layout={:#x} render_pass={:#x} subpass={}",
                      diagnostic::HandleBits(object), static_cast<s32>(result),
                      static_cast<u32>(ci.flags), ci.stageCount, diagnostic::HandleBits(ci.layout),
                      diagnostic::HandleBits(ci.renderPass), ci.subpass);
    for (u32 i = 0; i < ci.stageCount; ++i) {
        const auto& stage = ci.pStages[i];
        diagnostic::Trace("PIPELINE_SHADER_STAGE",
                          "pipeline={:#x} index={} stage={:#x} module={:#x} entry={}",
                          diagnostic::HandleBits(object), i, static_cast<u32>(stage.stage),
                          diagnostic::HandleBits(stage.module), stage.pName ? stage.pName : "");
    }
    // Adreno 5xx drivers do not properly return when a graphics pipeline fails to be created
    // Some (unkown) Mali drivers also do not properly return
    // This result code is out of spec, but should be handled as "kinda working"
    if (result == VK_INCOMPLETE)
        return Pipeline(object, handle, *dld);
    Check(result);
    return Pipeline(object, handle, *dld);
}

Pipeline Device::CreateComputePipeline(const VkComputePipelineCreateInfo& ci,
                                       VkPipelineCache cache) const {
    VkPipeline object = VK_NULL_HANDLE;
    const VkResult result = dld->vkCreateComputePipelines(handle, cache, 1, &ci, nullptr, &object);
    diagnostic::Trace("CREATE_COMPUTE_PIPELINE",
                      "pipeline={:#x} result={} flags={:#x} layout={:#x} module={:#x} entry={}",
                      diagnostic::HandleBits(object), static_cast<s32>(result),
                      static_cast<u32>(ci.flags), diagnostic::HandleBits(ci.layout),
                      diagnostic::HandleBits(ci.stage.module), ci.stage.pName ? ci.stage.pName : "");
    Check(result);
    return Pipeline(object, handle, *dld);
}

Sampler Device::CreateSampler(const VkSamplerCreateInfo& ci) const {
    VkSampler object;
    Check(dld->vkCreateSampler(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_SAMPLER",
                      "sampler={:#x} mag={} min={} mipmap={} address=({},{},{}) compare_enable={} compare_op={} min_lod={} max_lod={} anisotropy_enable={} max_anisotropy={}",
                      diagnostic::HandleBits(object), static_cast<u32>(ci.magFilter),
                      static_cast<u32>(ci.minFilter), static_cast<u32>(ci.mipmapMode),
                      static_cast<u32>(ci.addressModeU), static_cast<u32>(ci.addressModeV),
                      static_cast<u32>(ci.addressModeW), ci.compareEnable != VK_FALSE,
                      static_cast<u32>(ci.compareOp), ci.minLod, ci.maxLod,
                      ci.anisotropyEnable != VK_FALSE, ci.maxAnisotropy);
    return Sampler(object, handle, *dld);
}

Framebuffer Device::CreateFramebuffer(const VkFramebufferCreateInfo& ci) const {
    VkFramebuffer object;
    Check(dld->vkCreateFramebuffer(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_FRAMEBUFFER",
                      "framebuffer={:#x} render_pass={:#x} attachments={} extent={}x{} layers={}",
                      diagnostic::HandleBits(object), diagnostic::HandleBits(ci.renderPass),
                      ci.attachmentCount, ci.width, ci.height, ci.layers);
    for (u32 i = 0; i < ci.attachmentCount; ++i) {
        diagnostic::Trace("FRAMEBUFFER_ATTACHMENT", "framebuffer={:#x} index={} image_view={:#x}",
                          diagnostic::HandleBits(object), i,
                          diagnostic::HandleBits(ci.pAttachments[i]));
    }
    return Framebuffer(object, handle, *dld);
}

CommandPool Device::CreateCommandPool(const VkCommandPoolCreateInfo& ci) const {
    VkCommandPool object;
    Check(dld->vkCreateCommandPool(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_COMMAND_POOL", "pool={:#x} flags={:#x} queue_family={}",
                      diagnostic::HandleBits(object), static_cast<u32>(ci.flags),
                      ci.queueFamilyIndex);
    return CommandPool(object, handle, *dld);
}

DescriptorUpdateTemplate Device::CreateDescriptorUpdateTemplate(
    const VkDescriptorUpdateTemplateCreateInfo& ci) const {
    VkDescriptorUpdateTemplate object;
    Check(dld->vkCreateDescriptorUpdateTemplate(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_DESCRIPTOR_UPDATE_TEMPLATE",
                      "template={:#x} entries={} template_type={} set={} bind_point={}",
                      diagnostic::HandleBits(object), ci.descriptorUpdateEntryCount,
                      static_cast<u32>(ci.templateType), ci.set,
                      static_cast<u32>(ci.pipelineBindPoint));
    return DescriptorUpdateTemplate(object, handle, *dld);
}

QueryPool Device::CreateQueryPool(const VkQueryPoolCreateInfo& ci) const {
    VkQueryPool object;
    Check(dld->vkCreateQueryPool(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_QUERY_POOL", "query_pool={:#x} type={} count={} statistic_flags={:#x}",
                      diagnostic::HandleBits(object), static_cast<u32>(ci.queryType),
                      ci.queryCount, static_cast<u32>(ci.pipelineStatistics));
    return QueryPool(object, handle, *dld);
}

ShaderModule Device::CreateShaderModule(const VkShaderModuleCreateInfo& ci) const {
    VkShaderModule object;
    Check(dld->vkCreateShaderModule(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_SHADER_MODULE", "module={:#x} code_size={} code_ptr={:#x}",
                      diagnostic::HandleBits(object), ci.codeSize,
                      static_cast<u64>(reinterpret_cast<std::uintptr_t>(ci.pCode)));
    return ShaderModule(object, handle, *dld);
}

Event Device::CreateEvent() const {
    static constexpr VkEventCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };

    VkEvent object;
    Check(dld->vkCreateEvent(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_EVENT", "event={:#x} flags={:#x}",
                      diagnostic::HandleBits(object), static_cast<u32>(ci.flags));
    return Event(object, handle, *dld);
}

SwapchainKHR Device::CreateSwapchainKHR(const VkSwapchainCreateInfoKHR& ci) const {
    VkSwapchainKHR object;
    Check(dld->vkCreateSwapchainKHR(handle, &ci, nullptr, &object));
    diagnostic::Trace("CREATE_SWAPCHAIN",
                      "swapchain={:#x} surface={:#x} min_images={} format={} color_space={} extent={}x{} layers={} usage={:#x} sharing={} queue_families={} transform={:#x} composite_alpha={:#x} present_mode={} clipped={} old_swapchain={:#x}",
                      diagnostic::HandleBits(object), diagnostic::HandleBits(ci.surface),
                      ci.minImageCount, static_cast<s32>(ci.imageFormat),
                      static_cast<u32>(ci.imageColorSpace), ci.imageExtent.width,
                      ci.imageExtent.height, ci.imageArrayLayers, static_cast<u32>(ci.imageUsage),
                      static_cast<u32>(ci.imageSharingMode), ci.queueFamilyIndexCount,
                      static_cast<u32>(ci.preTransform), static_cast<u32>(ci.compositeAlpha),
                      static_cast<u32>(ci.presentMode), ci.clipped != VK_FALSE,
                      diagnostic::HandleBits(ci.oldSwapchain));
    return SwapchainKHR(object, handle, *dld);
}

DeviceMemory Device::TryAllocateMemory(const VkMemoryAllocateInfo& ai) const noexcept {
    VkDeviceMemory memory;
    if (dld->vkAllocateMemory(handle, &ai, nullptr, &memory) != VK_SUCCESS) {
        return {};
    }
    return DeviceMemory(memory, handle, *dld);
}

DeviceMemory Device::AllocateMemory(const VkMemoryAllocateInfo& ai) const {
    VkDeviceMemory memory;
    Check(dld->vkAllocateMemory(handle, &ai, nullptr, &memory));
    return DeviceMemory(memory, handle, *dld);
}

VkMemoryRequirements Device::GetBufferMemoryRequirements(VkBuffer buffer,
                                                         void* pnext) const noexcept {
    const VkBufferMemoryRequirementsInfo2 info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2,
        .pNext = nullptr,
        .buffer = buffer,
    };
    VkMemoryRequirements2 requirements{
        .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
        .pNext = pnext,
        .memoryRequirements{},
    };
    dld->vkGetBufferMemoryRequirements2(handle, &info, &requirements);
    return requirements.memoryRequirements;
}

VkMemoryRequirements Device::GetImageMemoryRequirements(VkImage image) const noexcept {
    VkMemoryRequirements requirements;
    dld->vkGetImageMemoryRequirements(handle, image, &requirements);
    return requirements;
}

std::vector<VkSparseImageMemoryRequirements> Device::GetImageSparseMemoryRequirements(
    VkImage image) const {
    u32 count = 0;
    dld->vkGetImageSparseMemoryRequirements(handle, image, &count, nullptr);
    std::vector<VkSparseImageMemoryRequirements> requirements(count);
    dld->vkGetImageSparseMemoryRequirements(handle, image, &count, requirements.data());
    requirements.resize(count);
    return requirements;
}

std::vector<VkPipelineExecutablePropertiesKHR> Device::GetPipelineExecutablePropertiesKHR(
    VkPipeline pipeline) const {
    const VkPipelineInfoKHR info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR,
        .pNext = nullptr,
        .pipeline = pipeline,
    };
    u32 num{};
    dld->vkGetPipelineExecutablePropertiesKHR(handle, &info, &num, nullptr);
    std::vector<VkPipelineExecutablePropertiesKHR> properties(num);
    for (auto& property : properties) {
        property.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR;
    }
    Check(dld->vkGetPipelineExecutablePropertiesKHR(handle, &info, &num, properties.data()));
    return properties;
}

std::vector<VkPipelineExecutableStatisticKHR> Device::GetPipelineExecutableStatisticsKHR(
    VkPipeline pipeline, u32 executable_index) const {
    const VkPipelineExecutableInfoKHR executable_info{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
        .pNext = nullptr,
        .pipeline = pipeline,
        .executableIndex = executable_index,
    };
    u32 num{};
    dld->vkGetPipelineExecutableStatisticsKHR(handle, &executable_info, &num, nullptr);
    std::vector<VkPipelineExecutableStatisticKHR> statistics(num);
    for (auto& statistic : statistics) {
        statistic.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
    }
    Check(dld->vkGetPipelineExecutableStatisticsKHR(handle, &executable_info, &num,
                                                    statistics.data()));
    return statistics;
}

void Device::UpdateDescriptorSets(Span<VkWriteDescriptorSet> writes,
                                  Span<VkCopyDescriptorSet> copies) const noexcept {
    diagnostic::Trace("UPDATE_DESCRIPTOR_SETS", "device={:#x} writes={} copies={}",
                      diagnostic::HandleBits(handle), writes.size(), copies.size());
    for (u32 i = 0; i < writes.size(); ++i) {
        const auto& write = writes[i];
        diagnostic::Trace("DESCRIPTOR_WRITE",
                          "index={} set={:#x} binding={} array_element={} count={} type={}",
                          i, diagnostic::HandleBits(write.dstSet), write.dstBinding,
                          write.dstArrayElement, write.descriptorCount,
                          static_cast<u32>(write.descriptorType));
        if (write.pImageInfo) {
            for (u32 j = 0; j < write.descriptorCount; ++j) {
                const auto& image = write.pImageInfo[j];
                diagnostic::Trace("DESCRIPTOR_IMAGE_INFO",
                                  "write={} element={} sampler={:#x} image_view={:#x} layout={}",
                                  i, j, diagnostic::HandleBits(image.sampler),
                                  diagnostic::HandleBits(image.imageView),
                                  static_cast<u32>(image.imageLayout));
            }
        }
        if (write.pBufferInfo) {
            for (u32 j = 0; j < write.descriptorCount; ++j) {
                const auto& buffer = write.pBufferInfo[j];
                diagnostic::Trace("DESCRIPTOR_BUFFER_INFO",
                                  "write={} element={} buffer={:#x} offset={} range={}",
                                  i, j, diagnostic::HandleBits(buffer.buffer), buffer.offset,
                                  buffer.range);
            }
        }
        if (write.pTexelBufferView) {
            for (u32 j = 0; j < write.descriptorCount; ++j) {
                diagnostic::Trace("DESCRIPTOR_TEXEL_VIEW", "write={} element={} view={:#x}",
                                  i, j, diagnostic::HandleBits(write.pTexelBufferView[j]));
            }
        }
    }
    for (u32 i = 0; i < copies.size(); ++i) {
        const auto& copy = copies[i];
        diagnostic::Trace("DESCRIPTOR_COPY",
                          "index={} src_set={:#x} src_binding={} src_array={} dst_set={:#x} dst_binding={} dst_array={} count={}",
                          i, diagnostic::HandleBits(copy.srcSet), copy.srcBinding,
                          copy.srcArrayElement, diagnostic::HandleBits(copy.dstSet),
                          copy.dstBinding, copy.dstArrayElement, copy.descriptorCount);
    }
    dld->vkUpdateDescriptorSets(handle, writes.size(), writes.data(), copies.size(), copies.data());
}

VkPhysicalDeviceProperties PhysicalDevice::GetProperties() const noexcept {
    VkPhysicalDeviceProperties properties;
    dld->vkGetPhysicalDeviceProperties(physical_device, &properties);
    return properties;
}

void PhysicalDevice::GetProperties2(VkPhysicalDeviceProperties2& properties) const noexcept {
    dld->vkGetPhysicalDeviceProperties2(physical_device, &properties);
}

VkPhysicalDeviceFeatures PhysicalDevice::GetFeatures() const noexcept {
    VkPhysicalDeviceFeatures2 features2;
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = nullptr;
    dld->vkGetPhysicalDeviceFeatures2(physical_device, &features2);
    return features2.features;
}

void PhysicalDevice::GetFeatures2(VkPhysicalDeviceFeatures2& features) const noexcept {
    dld->vkGetPhysicalDeviceFeatures2(physical_device, &features);
}

VkFormatProperties PhysicalDevice::GetFormatProperties(VkFormat format) const noexcept {
    VkFormatProperties properties;
    dld->vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
    return properties;
}

VkFormatProperties3 PhysicalDevice::GetFormatProperties3(VkFormat format) const noexcept {
    VkFormatProperties3 properties3{
        .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3,
        .pNext = nullptr,
        .linearTilingFeatures = 0,
        .optimalTilingFeatures = 0,
        .bufferFeatures = 0,
    };
    VkFormatProperties2 properties2{
        .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
        .pNext = &properties3,
        .formatProperties = {},
    };
    dld->vkGetPhysicalDeviceFormatProperties2(physical_device, format, &properties2);
    return properties3;
}

std::vector<VkExtensionProperties> PhysicalDevice::EnumerateDeviceExtensionProperties() const {
    u32 num;
    dld->vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &num, nullptr);
    std::vector<VkExtensionProperties> properties(num);
    dld->vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &num, properties.data());
    return properties;
}

std::vector<VkQueueFamilyProperties> PhysicalDevice::GetQueueFamilyProperties() const {
    u32 num;
    dld->vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &num, nullptr);
    std::vector<VkQueueFamilyProperties> properties(num);
    dld->vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &num, properties.data());
    return properties;
}

std::vector<VkPhysicalDeviceToolProperties> PhysicalDevice::GetPhysicalDeviceToolProperties()
    const {
    u32 num = 0;
    if (!dld->vkGetPhysicalDeviceToolProperties) {
        return {};
    }
    dld->vkGetPhysicalDeviceToolProperties(physical_device, &num, nullptr);
    std::vector<VkPhysicalDeviceToolProperties> properties(num);
    dld->vkGetPhysicalDeviceToolProperties(physical_device, &num, properties.data());
    return properties;
}

std::vector<VkCooperativeMatrixPropertiesKHR> PhysicalDevice::GetCooperativeMatrixPropertiesKHR()
    const {
    if (!dld->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR) {
        return {};
    }
    u32 num = 0;
    if (dld->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physical_device, &num, nullptr) !=
        VK_SUCCESS) {
        return {};
    }
    std::vector<VkCooperativeMatrixPropertiesKHR> properties(
        num, VkCooperativeMatrixPropertiesKHR{
                 .sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
                 .pNext = nullptr,
             });
    if (dld->vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR(physical_device, &num,
                                                               properties.data()) != VK_SUCCESS) {
        return {};
    }
    properties.resize(num);
    return properties;
}

bool PhysicalDevice::GetSurfaceSupportKHR(u32 queue_family_index, VkSurfaceKHR surface) const {
    VkBool32 supported;
    Check(dld->vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, queue_family_index, surface,
                                                    &supported));
    return supported == VK_TRUE;
}

VkSurfaceCapabilitiesKHR PhysicalDevice::GetSurfaceCapabilitiesKHR(VkSurfaceKHR surface) const {
    VkSurfaceCapabilitiesKHR capabilities;
    Check(dld->vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities));
    return capabilities;
}

std::vector<VkSurfaceFormatKHR> PhysicalDevice::GetSurfaceFormatsKHR(VkSurfaceKHR surface) const {
    u32 num;
    Check(dld->vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &num, nullptr));
    std::vector<VkSurfaceFormatKHR> formats(num);
    Check(
        dld->vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &num, formats.data()));
    return formats;
}

std::vector<VkPresentModeKHR> PhysicalDevice::GetSurfacePresentModesKHR(
    VkSurfaceKHR surface) const {
    u32 num;
    Check(dld->vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &num, nullptr));
    std::vector<VkPresentModeKHR> modes(num);
    Check(dld->vkGetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &num,
                                                         modes.data()));
    return modes;
}

VkPhysicalDeviceMemoryProperties2 PhysicalDevice::GetMemoryProperties(
    void* next_structures) const noexcept {
    VkPhysicalDeviceMemoryProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    properties.pNext = next_structures;
    dld->vkGetPhysicalDeviceMemoryProperties2(physical_device, &properties);
    return properties;
}

u32 AvailableVersion(const InstanceDispatch& dld) noexcept {
    PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion;
    if (!Proc(vkEnumerateInstanceVersion, dld, "vkEnumerateInstanceVersion")) {
        // If the procedure is not found, Vulkan 1.0 is assumed
        return VK_API_VERSION_1_0;
    }
    u32 version;
    if (const VkResult result = vkEnumerateInstanceVersion(&version); result != VK_SUCCESS) {
        LOG_ERROR(Render_Vulkan, "vkEnumerateInstanceVersion returned {}, assuming Vulkan 1.1",
                  string_VkResult(result));
        return VK_API_VERSION_1_1;
    }
    return version;
}

std::optional<std::vector<VkExtensionProperties>> EnumerateInstanceExtensionProperties(
    const InstanceDispatch& dld) {
    u32 num;
    if (dld.vkEnumerateInstanceExtensionProperties(nullptr, &num, nullptr) != VK_SUCCESS) {
        return std::nullopt;
    }
    std::vector<VkExtensionProperties> properties(num);
    if (dld.vkEnumerateInstanceExtensionProperties(nullptr, &num, properties.data()) !=
        VK_SUCCESS) {
        return std::nullopt;
    }
    return properties;
}

std::optional<std::vector<VkLayerProperties>> EnumerateInstanceLayerProperties(
    const InstanceDispatch& dld) {
    u32 num;
    if (dld.vkEnumerateInstanceLayerProperties(&num, nullptr) != VK_SUCCESS) {
        return std::nullopt;
    }
    std::vector<VkLayerProperties> properties(num);
    if (dld.vkEnumerateInstanceLayerProperties(&num, properties.data()) != VK_SUCCESS) {
        return std::nullopt;
    }
    return properties;
}

std::string GetDriverName(VkPhysicalDeviceDriverProperties driver) {
    switch (driver.driverID) {
    case VK_DRIVER_ID_AMD_PROPRIETARY:
        return "AMD";
    case VK_DRIVER_ID_AMD_OPEN_SOURCE:
        return "AMDVLK";
    case VK_DRIVER_ID_MESA_RADV:
        return "RADV";
    case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
        return "Nvidia";
    case VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS:
        return "Intel";
    case VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA:
        return "ANV";
    case VK_DRIVER_ID_IMAGINATION_PROPRIETARY:
        return "PowerVR";
    case VK_DRIVER_ID_QUALCOMM_PROPRIETARY:
        return "Qualcomm";
    case VK_DRIVER_ID_ARM_PROPRIETARY:
        return "Mali";
    case VK_DRIVER_ID_SAMSUNG_PROPRIETARY:
        return "Xclipse";
    case VK_DRIVER_ID_GOOGLE_SWIFTSHADER:
        return "SwiftShader";
    case VK_DRIVER_ID_BROADCOM_PROPRIETARY:
        return "Broadcom";
    case VK_DRIVER_ID_MESA_LLVMPIPE:
        return "llvmpipe";
    case VK_DRIVER_ID_MOLTENVK:
        return "MoltenVK";
    case VK_DRIVER_ID_VERISILICON_PROPRIETARY:
        return "Vivante";
    case VK_DRIVER_ID_MESA_TURNIP:
        return "Turnip";
    case VK_DRIVER_ID_MESA_V3DV:
        return "V3DV";
    case VK_DRIVER_ID_MESA_PANVK:
        return "PanVK";
    case VK_DRIVER_ID_MESA_VENUS:
        return "Venus";
    case VK_DRIVER_ID_MESA_DOZEN:
        return "Dozen";
    case VK_DRIVER_ID_MESA_NVK:
        return "NVK";
    case VK_DRIVER_ID_IMAGINATION_OPEN_SOURCE_MESA:
        return "PVR";
    case VK_DRIVER_ID_MESA_HONEYKRISP:
        return "HoneyKrisp";
    case VK_DRIVER_ID_MESA_KOSMICKRISP:
        return "KosmicKrisp";
    case VK_DRIVER_ID_GGP_PROPRIETARY:
    case VK_DRIVER_ID_COREAVI_PROPRIETARY:
    case VK_DRIVER_ID_JUICE_PROPRIETARY:
    case VK_DRIVER_ID_VULKAN_SC_EMULATION_ON_VULKAN:
    case VK_DRIVER_ID_MAX_ENUM:
    default:
        return driver.driverName;
    }
}

} // namespace Vulkan::vk
