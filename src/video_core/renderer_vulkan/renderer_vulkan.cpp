// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <fmt/ranges.h>

#include "common/display_hdr.h"
#include "common/logging.h"
#include <ranges>
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core_timing.h"
#include "core/frontend/graphics_context.h"
#include "video_core/capture.h"
#include "video_core/gpu.h"
#include "video_core/dlss_mode.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/present.h"
#include "video_core/render_trace.h"
#include "video_core/renderer_vulkan/present/util.h"
#ifdef HAS_RESHADE
#include "video_core/post_processing/fx_chain.h"
#endif
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_blit_screen.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/textures/decoders.h"
#include "video_core/vulkan_common/vk_enum_string_helper.h"
#include "video_core/vulkan_common/vulkan_address_map.h"
#include "video_core/vulkan_common/vulkan_debug_callback.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_instance.h"
#include "video_core/vulkan_common/vulkan_library.h"
#include "video_core/vulkan_common/vulkan_memory_allocator.h"
#include "video_core/vulkan_common/vulkan_surface.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"
#ifdef EGLANTINE_STREAMLINE
#include "video_core/renderer_vulkan/vk_streamline_dlss.h"
#include "video_core/vulkan_common/streamline.h"
#include "video_core/vulkan_common/streamline_frame_gen.h"
#endif
#ifdef __ANDROID__
#include <jni.h>
#endif
namespace Vulkan {
namespace {

constexpr VkExtent2D CaptureImageSize{
    .width = VideoCore::Capture::LinearWidth,
    .height = VideoCore::Capture::LinearHeight,
};

#ifdef HAS_LSFG
[[nodiscard]] VkExtent2D GuestExtent(std::span<const Tegra::FramebufferConfig> framebuffers) {
    if (framebuffers.empty()) {
        return VkExtent2D{};
    }

    const auto& framebuffer = framebuffers.front();
    if (framebuffer.crop_rect.IsEmpty()) {
        return VkExtent2D{.width = framebuffer.width, .height = framebuffer.height};
    }
    return VkExtent2D{
        .width = static_cast<u32>(framebuffer.crop_rect.GetWidth()),
        .height = static_cast<u32>(framebuffer.crop_rect.GetHeight()),
    };
}
#endif

constexpr VkExtent3D CaptureImageExtent{
    .width = VideoCore::Capture::LinearWidth,
    .height = VideoCore::Capture::LinearHeight,
    .depth = VideoCore::Capture::LinearDepth,
};

constexpr VkFormat CaptureFormat = VK_FORMAT_A8B8G8R8_UNORM_PACK32;

std::string GetReadableVersion(u32 version) {
    return fmt::format("{}.{}.{}", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version),
                       VK_VERSION_PATCH(version));
}

std::string GetDriverVersion(const Device& device) {
    // Extracted from
    // https://github.com/SaschaWillems/vulkan.gpuinfo.org/blob/5dddea46ea1120b0df14eef8f15ff8e318e35462/functions.php#L308-L314
    const u32 version = device.GetDriverVersion();

    if (device.GetDriverID() == VK_DRIVER_ID_NVIDIA_PROPRIETARY) {
        const u32 major = (version >> 22) & 0x3ff;
        const u32 minor = (version >> 14) & 0x0ff;
        const u32 secondary = (version >> 6) & 0x0ff;
        const u32 tertiary = version & 0x003f;
        return fmt::format("{}.{}.{}.{}", major, minor, secondary, tertiary);
    }
    if (device.GetDriverID() == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS) {
        const u32 major = version >> 14;
        const u32 minor = version & 0x3fff;
        return fmt::format("{}.{}", major, minor);
    }
    return GetReadableVersion(version);
}

std::string BuildCommaSeparatedExtensions(
    const std::set<std::string, std::less<>>& available_extensions) {
    return fmt::format("{}", fmt::join(available_extensions, ","));
}

// The swapchain format and colour space to present in: an HDR pair the surface lists while HDR
// output is on (Off/On/Auto with the monitor's state), else SDR.
VkSurfaceFormatKHR ChooseOutputFormat(const vk::PhysicalDevice& physical, VkSurfaceKHR surface,
                                      void* window) {
    auto& monitor = Common::DisplayHdrMonitor::Instance();
    monitor.Watch(window);
    const auto output = Common::ResolveHdrOutput(
        monitor.Current(), Settings::values.hdr_mode.GetValue(),
        Settings::values.hdr_peak_luminance.GetValue(),
        Settings::values.hdr_reference_white.GetValue());
    if (output.enabled) {
        const auto formats = physical.GetSurfaceFormatsKHR(surface);
        const auto hdr =
            ChooseHdrSurfaceFormat(formats, Settings::values.hdr_swapchain.GetValue());
        if (hdr) {
            return *hdr;
        }
        LOG_WARNING(Render_Vulkan,
                    "HDR output is on, but the surface lists no HDR10 or scRGB format");
    }
#ifdef __ANDROID__
    return {VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
#else
    return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
#endif
}

/// Frame generation shows its frames only while the real frame rate is at least
/// EDEN_NX2_FG_MIN_FPS (default 40; 0 always): below it the generated frames carry large
/// interpolation errors and add latency. It opens again 10 FPS higher, so it does not flap.
bool FrameGenGateOpen(f32 interval_s) {
    static const f32 min_fps = [] {
        const char* const value = std::getenv("EDEN_NX2_FG_MIN_FPS");
        return value ? static_cast<f32>(std::strtod(value, nullptr)) : 40.0f;
    }();
    static bool open = true;
    if (min_fps <= 0.0f || interval_s <= 0.0f) {
        return open = true;
    }
    const f32 fps = 1.0f / interval_s;
    const bool next = open ? fps >= min_fps : fps >= min_fps + 10.0f;
    if (next != open) {
        LOG_INFO(Render_Vulkan, "Frame generation {} at {:.1f} real FPS (gate {:.0f}-{:.0f} FPS)",
                 next ? "resumes" : "pauses", fps, min_fps, min_fps + 10.0f);
    }
    return open = next;
}

/// Settings that do nothing or harm on this setup, once per session (frame generation once its
/// state is known).
void LogSetupWarnings(const Device& device, VideoCore::FrameGenState frame_gen_state) {
    const auto& v = Settings::values;
    const u64 program_id = Settings::GetCurrentProgramID();
    const bool scaled = v.resolution_info.up_scale > (1U << v.resolution_info.down_shift);
    static bool logged = false;
    if (!std::exchange(logged, true)) {
        std::vector<std::string> warnings;
        if (v.nvidia_dlss.GetValue() && !device.IsNvidia()) {
            warnings.emplace_back("NVIDIA DLSS is on but the GPU is not NVIDIA: ignored, FSR 3.1 "
                                  "runs");
        }
        if (v.amd_fsr4.GetValue() && program_id != 0x0400486024DEC000ULL) {
            warnings.emplace_back("AMD FSR 4 is on but this game has no FSR 4 path: ignored");
        }
        if (program_id == VideoCommon::DlssMode::DuskbloodsProgramId && !Settings::IsDockedMode() &&
            scaled) {
            warnings.emplace_back("The Duskbloods in handheld mode above 1x is untested (it "
                                  "renders a smaller region there); docked is the tested setup");
        }
        if (v.scaling_filter.GetValue() == Settings::ScalingFilter::NearestNeighbor && scaled) {
            warnings.emplace_back("Nearest neighbour window filter above 1x: the frame is "
                                  "area-sampled when it is larger than the window; Bilinear or "
                                  "Area look the same there and smoother when it is smaller");
        }
        for (const std::string& warning : warnings) {
            LOG_WARNING(Render_Vulkan, "Setup warning: {}", warning);
        }
    }
    static bool frame_gen_logged = false;
    if (v.frame_gen.GetValue() && !frame_gen_logged &&
        frame_gen_state != VideoCore::FrameGenState::Pending) {
        frame_gen_logged = true;
        const char* reason = nullptr;
        switch (frame_gen_state) {
        case VideoCore::FrameGenState::UnsupportedGpu:
            reason = "this GPU lacks features LSFG needs (16-bit floats, Vulkan memory model, null "
                     "descriptors); the FSR3 frame generation type does not need them";
            break;
        case VideoCore::FrameGenState::MissingLosslessDll:
            reason = "Lossless.dll is missing; the FSR3 frame generation type does not need it";
            break;
        case VideoCore::FrameGenState::UnsupportedSurface:
            reason = "the window's swapchain format can't be written from a shader";
            break;
        case VideoCore::FrameGenState::UnsupportedBuild:
            reason = "this build has no frame generation";
            break;
        default:
            break;
        }
        if (reason) {
            LOG_WARNING(Render_Vulkan, "Setup warning: frame generation is on but unavailable: {}",
                        reason);
        }
    }
}

/// Every frame interval of the session in 0.1 ms buckets (the last one: 2 s or more), for the
/// "Nx2Summary run" line when the game stops.
struct RunFrames {
    static constexpr size_t Buckets = 20001;
    std::array<u32, Buckets> histogram{};
    u64 frames{};
    double sum_ms{};
    u64 over_100ms{};
    float longest_ms{};

    void Add(float ms) {
        ++histogram[(std::min)(static_cast<size_t>(ms * 10.0f), Buckets - 1)];
        ++frames;
        sum_ms += ms;
        over_100ms += ms > 100.0f ? 1 : 0;
        longest_ms = (std::max)(longest_ms, ms);
    }

    /// The frame time `share` of the frames reach or exceed.
    [[nodiscard]] double Slowest(double share) const {
        const u64 target = (std::max)(u64{1}, static_cast<u64>(static_cast<double>(frames) * share));
        u64 seen{};
        for (size_t bucket = Buckets; bucket-- > 0;) {
            seen += histogram[bucket];
            if (seen >= target) {
                return static_cast<double>(bucket) / 10.0;
            }
        }
        return 0.0;
    }
};

RunFrames& Run() {
    static RunFrames run;
    return run;
}

/// "Nx2Summary run:" once, when the game stops: the session's average and 1% / 0.1% lows (the
/// frame rate of the slowest 1% and 0.1% of frames), whatever the switches.
void LogRunSummary() {
    const RunFrames& run = Run();
    if (run.frames == 0 || run.sum_ms <= 0.0) {
        return;
    }
    const auto fps = [](double ms) { return ms > 0.0 ? 1000.0 / ms : 0.0; };
    LOG_INFO(Render_Vulkan,
             "Nx2Summary run: {:.0f} s, {} frames, average {:.1f} fps, 1% low {:.1f} fps, 0.1% low "
             "{:.1f} fps, {} frames over 100 ms, longest {:.0f} ms",
             run.sum_ms / 1000.0, run.frames, fps(run.sum_ms / static_cast<double>(run.frames)),
             fps(run.Slowest(0.01)), fps(run.Slowest(0.001)), run.over_100ms, run.longest_ms);
    Run() = RunFrames{}; // the next game starts over
}

/// "Nx2Summary perf:" every 10 s (GPU thread, at each present): what a report needs to judge the
/// frame rate and the image without other switches.
void LogPerfSummary(Scheduler& scheduler, const VideoCore::DlssStatus& dlss, u64 generated) {
    namespace Perf = VideoCommon::Nx2Perf;
    using Clock = std::chrono::steady_clock;
    static Clock::time_point start{};
    static Clock::time_point last{};
    static std::vector<float> intervals;
    static u64 generated_frames{};
    static u64 dlss_evaluations{};
    const Clock::time_point now = Clock::now();
    if (last != Clock::time_point{}) {
        intervals.push_back(std::chrono::duration<float, std::milli>(now - last).count());
        Run().Add(intervals.back());
    } else {
        start = now;
        dlss_evaluations = dlss.evaluations;
    }
    last = now;
    generated_frames += generated;
    if (now - start < std::chrono::seconds{10} || intervals.empty()) {
        return;
    }
    std::vector<float> sorted = intervals;
    std::ranges::sort(sorted);
    const double frames = static_cast<double>(sorted.size());
    double sum{};
    for (const float value : sorted) {
        sum += value;
    }
    const float p99 = sorted[static_cast<size_t>(frames * 0.99) < sorted.size()
                                 ? static_cast<size_t>(frames * 0.99)
                                 : sorted.size() - 1];
    const u64 sync_ns = Perf::g_sync_submit_wait_ns.exchange(0, std::memory_order_relaxed);
    const u64 sync_waits = Perf::g_sync_submit_waits.exchange(0, std::memory_order_relaxed);
    const u64 upscales = Perf::g_upscales.exchange(0, std::memory_order_relaxed);
    const u64 resets = Perf::g_upscale_resets.exchange(0, std::memory_order_relaxed);
    const bool dlss_ran = dlss.evaluations > dlss_evaluations;
    std::string upscaler = "none";
    if (upscales != 0) {
        upscaler = dlss_ran ? fmt::format("DLSS {} preset {}", Settings::CanonicalizeEnum(dlss.mode),
                                          Settings::CanonicalizeEnum(dlss.preset))
                            : std::string{"FSR 3.1"};
    }
    const Scheduler::GpuSplit gpu = scheduler.TakeGpuSplit();
    const std::string gpu_text =
        gpu.frames == 0 ? std::string{"n/a"}
                        : fmt::format("pre {:.2f} upscale {:.2f} (max {:.2f}) post {:.2f} over {} frames",
                                      gpu.pre_ms, gpu.upscale_ms, gpu.upscale_max_ms,
                                      gpu.post_ms, gpu.frames);
    LOG_INFO(Render_Vulkan,
             "Nx2Summary perf: {:.1f} s, {} frames, frame ms avg {:.2f} p99 {:.2f} max {:.2f}; "
             "sync submit wait {:.2f} ms/frame ({} waits); upscaler {} ({} upscales, {} history "
             "resets); GPU ms/frame {}; generated frames {:.0f}%; commit left {} "
             "MiB",
             std::chrono::duration<double>(now - start).count(), sorted.size(), sum / frames, p99,
             sorted.back(), static_cast<double>(sync_ns) / 1e6 / frames, sync_waits, upscaler,
             upscales, resets, gpu_text,
             100.0 * static_cast<double>(generated_frames) /
                 (frames + static_cast<double>(generated_frames)),
             Perf::g_commit_left_mib.load(std::memory_order_relaxed));
    intervals.clear();
    generated_frames = 0;
    dlss_evaluations = dlss.evaluations;
    start = now;
}

} // Anonymous namespace

Device CreateDevice(const vk::Instance& instance, const vk::InstanceDispatch& dld, VkSurfaceKHR surface) {
    const std::vector<VkPhysicalDevice> devices = instance.EnumeratePhysicalDevices();
    const u32 device_index = Settings::values.vulkan_device.GetValue();
    if (device_index >= u32(devices.size())) {
        LOG_ERROR(Render_Vulkan, "Invalid device index {}!", device_index);
        throw vk::Exception(VK_ERROR_INITIALIZATION_FAILED);
    }
    const vk::PhysicalDevice physical_device(devices[device_index], dld);
    return Device(*instance, physical_device, surface, dld);
}

RendererVulkan::RendererVulkan(Core::Frontend::EmuWindow& emu_window,
                               Tegra::MaxwellDeviceMemoryManager& device_memory_,
                               Tegra::GPU& gpu_,
                               std::unique_ptr<Core::Frontend::GraphicsContext> context_)
try
    : RendererBase(emu_window, std::move(context_))
    , device_memory(device_memory_)
    , gpu(gpu_)
    , library(OpenLibrary(context.get()))
    , dld()
#ifdef EGLANTINE_STREAMLINE
    , streamline_started(Streamline::Start(render_window.GetWindowInfo().render_surface))
#endif
    // Create raw Vulkan instance first
    , instance(CreateInstance(*library,
                            dld,
                            VK_API_VERSION_1_1,
                            render_window.GetWindowInfo().type,
                            Settings::values.renderer_debug.GetValue()))
    // Create debug messenger if debug is enabled
    , debug_messenger(Settings::values.renderer_debug ? CreateDebugUtilsCallback(instance)
                                                    : vk::DebugUtilsMessenger{})
    , address_messenger(AddressMap::Wanted() ? CreateAddressBindingCallback(instance, dld)
                                             : vk::DebugUtilsMessenger{})
    // Create surface
    , surface(CreateSurface(instance, render_window.GetWindowInfo()))
    , device(CreateDevice(instance, dld, *surface))
    , memory_allocator(device)
    , state_tracker()
    , scheduler(device, state_tracker)
    , swapchain(PresentationSurface(),
                device,
                scheduler,
               render_window.GetFramebufferLayout().width,
               render_window.GetFramebufferLayout().height,
               ChooseOutputFormat(device.GetPhysical(), *surface,
                                  render_window.GetWindowInfo().render_surface))
    , present_manager(instance,
                      render_window,
                      device,
                      memory_allocator,
                      scheduler,
                      swapchain,
                      surface)
    , blit_swapchain(device_memory,
                   device,
                   memory_allocator,
                   present_manager,
                   scheduler,
                   PresentFiltersForDisplay)
    , blit_capture(device_memory,
                   device,
                   memory_allocator,
                   present_manager,
                   scheduler,
                   PresentFiltersForDisplay)
    , blit_applet(device_memory,
                  device,
                  memory_allocator,
                  present_manager,
                  scheduler,
                  PresentFiltersForAppletCapture)
    , rasterizer(render_window, gpu, device_memory, device, memory_allocator, state_tracker, scheduler)
#ifdef HAS_LSFG
    , frame_gen(memory_allocator, scheduler)
#endif
{

    // Force maximum clocks (off by default): a second device keeps the GPU busy. Its device
    // failing to start skips it instead of failing the whole renderer ("Vulkan initialization
    // error" and no boot). Fritchii a813290651.
    if (!Settings::values.renderer_force_max_clock.GetValue()) {
        LOG_INFO(Render_Vulkan, "Force maximum clocks: off");
    } else if (!device.ShouldBoostClocks()) {
        LOG_INFO(Render_Vulkan, "Force maximum clocks: skipped (driver not on the list, Steam "
                                "Deck, or a debugging tool attached)");
    } else {
        try {
            turbo_mode.emplace(instance, dld);
            scheduler.RegisterOnSubmit([this] { turbo_mode->QueueSubmitted(); });
            LOG_INFO(Render_Vulkan, "Force maximum clocks: running (costs power; not for laptops "
                                    "on battery)");
        } catch (const std::exception& exception) {
            turbo_mode.reset();
            LOG_WARNING(Render_Vulkan, "Force maximum clocks: skipped (its device failed: {})",
                        exception.what());
        }
    }

#ifdef HAS_RESHADE
    VideoCore::FxChain::Instance().LoadFromSettings();
#endif

    device_info = {
        .model = std::string{device.GetModelName()},
        .driver = device.GetDriverVersion() != 0
                      ? fmt::format("{} {}", device.GetVendorName(), GetDriverVersion(device))
                      : std::string{},
        .api = "Vulkan",
        .api_version = GetReadableVersion(device.ApiVersion()),
        .device_local_memory = device.GetDeviceLocalMemory(),
        .extensions = {device.GetAvailableExtensions().begin(),
                       device.GetAvailableExtensions().end()},
        .features =
            {
                {"geometry_shader", device.IsGeometryShaderSupported()},
                {"tessellation", device.IsTessellationSupported()},
                {"mesh_shader", device.IsExtMeshShaderSupported()},
                {"descriptor_buffer", device.IsExtDescriptorBufferSupported()},
                {"buffer_device_address", device.IsBufferDeviceAddressSupported()},
                {"push_descriptor", device.IsKhrPushDescriptorSupported()},
                {"vertex_input_dynamic_state", device.IsExtVertexInputDynamicStateSupported()},
                {"extended_dynamic_state", device.IsExtExtendedDynamicStateSupported()},
                {"transform_feedback", device.IsExtTransformFeedbackSupported()},
                {"conservative_rasterization", device.IsExtConservativeRasterizationSupported()},
                {"line_rasterization", device.IsExtLineRasterizationSupported()},
                {"depth_clip_control", device.IsExtDepthClipControlSupported()},
                {"shader_float_controls", device.IsKhrShaderFloatControlsSupported()},
                {"subgroup_size_control", device.IsExtSubgroupSizeControlSupported()},
                {"vulkan_memory_model", device.IsVulkanMemoryModelSupported()},
                {"cooperative_hmma", device.IsCooperativeHmmaSupported()},
                {"sparse_binding", device.IsSparseBindingSupported()},
                {"optimal_astc", device.IsOptimalAstcSupported()},
                {"optimal_bcn", device.IsOptimalBcnSupported()},
                {"formatless_image_load", device.IsFormatlessImageLoadSupported()},
                {"pipeline_binary", device.IsKhrPipelineBinarySupported()},
            },
    };
    Report();
} catch (const vk::Exception& exception) {
    LOG_ERROR(Render_Vulkan, "Vulkan initialization failed with error: {}", exception.what());
    throw std::runtime_error{fmt::format("Vulkan initialization error {}", exception.what())};
}

RendererVulkan::~RendererVulkan() {
    LogRunSummary();
    scheduler.RegisterOnSubmit([] {});
    void(device.GetLogical().WaitIdle());
}

void RendererVulkan::Composite(std::span<const Tegra::FramebufferConfig> framebuffers) {
    const bool trace = Common::DkbFrameTrace::Enabled();
    const auto entered = trace ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
    SCOPE_EXIT {
        render_window.OnFrameDisplayed();
    };
#ifdef EGLANTINE_STREAMLINE
    // DLSS Frame Generation: this guest frame ends here, presented or not.
    Streamline::FrameGenBeginFrame();
    SCOPE_EXIT {
        Streamline::FrameGenNextFrame();
    };
#endif

    RenderAppletCaptureLayer(framebuffers);

    if (!render_window.IsShown()) {
        return;
    }

    if (VideoCommon::StutterTrace::Enabled()) {
        VideoCommon::StutterTrace::OnPresent();
    }
    RenderScreenshot(framebuffers);
    const auto acquiring = trace ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
    Frame* frame = [this] {
        const VideoCommon::Nx2Profile::Scope scope{VideoCommon::Nx2Profile::Zone::Present};
        return present_manager.GetRenderFrame();
    }();
    const auto acquired = trace ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};

    const HdrPresent output = UpdateHdrPresent();
    frame->color_space = output.color_space;

    scheduler.RequestOutsideRenderPassOperationContext();
    blit_swapchain.DrawToFrame(device, rasterizer, frame, framebuffers,
                              render_window.GetFramebufferLayout(),
                              present_manager.SwapchainImageCount(), output_format.format, output);
    scheduler.MarkGpu(Scheduler::GpuMark::Present);
    u64 perf_generated = 0;

#ifdef HAS_LSFG
    void(frame_gen.WantedGenerations(present_manager.MaxExtraFrames(),
                                     present_manager.FreeFrames(),
                                     present_manager.TakeBlockedTime()));

    frame_gen.Process(device, frame, frame->image_format, GuestExtent(framebuffers));

    const size_t generated_frames =
        FrameGenGateOpen(frame_gen.FrameInterval()) ? frame_gen.GeneratedFrameCount() : 0;
    perf_generated = generated_frames;
    // Each generated frame keeps the screen for an equal share of the real frame interval. Sent
    // back to back, a high refresh rate display shows it for one refresh only.
    const auto spacing = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<f32>(std::min(frame_gen.FrameInterval(), 0.1f) /
                                   static_cast<f32>(generated_frames + 1)));
    const u32 game_frame_presents = static_cast<u32>(generated_frames) + 1;
    frame->game_frame_presents = game_frame_presents;
    for (size_t generation = 0; generation < generated_frames; ++generation) {
        Frame* generated = present_manager.GetRenderFrame();
        generated->color_space = output.color_space;
        generated->present_spacing = spacing;
        generated->game_frame_presents = game_frame_presents;
        blit_swapchain.PrepareFrame(device, generated, render_window.GetFramebufferLayout());
        frame_gen.GenerateInto(device, generated, generation);
        scheduler.Flush(*generated->render_ready);
        present_manager.Present(generated);
    }
#endif

    LogPerfSummary(scheduler, GetDlssStatus(), perf_generated);
#ifdef HAS_LSFG
    LogSetupWarnings(device, frame_gen.CurrentStatus().state);
#else
    LogSetupWarnings(device, VideoCore::FrameGenState::UnsupportedBuild);
#endif
    if (const auto mark = scheduler.NextFrameMark(); mark != Scheduler::FrameMarkState::None) {
        scheduler.MarkFrameEnd(++trace_frame, mark);
    }
    scheduler.Flush(*frame->render_ready);
#ifdef EGLANTINE_STREAMLINE
    {
        const Streamline::FrameGenFrame ended = Streamline::FrameGenEndFrame();
        const Common::Rectangle<u32>& screen = render_window.GetFramebufferLayout().screen;
        frame->frame_gen = Streamline::FrameGenSession();
        frame->frame_gen_frame = ended;
        frame->guest_rect = VkRect2D{
            .offset = {static_cast<s32>(screen.left), static_cast<s32>(screen.top)},
            .extent = {screen.GetWidth(), screen.GetHeight()},
        };
    }
#endif

    const auto presenting = trace ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
    {
        const VideoCommon::Nx2Profile::Scope scope{VideoCommon::Nx2Profile::Zone::Present};
        present_manager.Present(frame);
    }
    const auto presented = trace ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
#ifdef HAS_LSFG
    scheduler.DispatchWork();
#endif

    gpu.RendererFrameEndNotify();
    rasterizer.TickFrame();
    if (trace) {
        const auto end = std::chrono::steady_clock::now();
        const auto ms = [](auto duration) {
            return std::chrono::duration<double, std::milli>(duration).count();
        };
        if (trace_cpu_rate.Ready(end)) {
            LOG_INFO(Render_Vulkan,
                     "DKB renderer renderer_frame={} host_pacing_mode={} vsync_mode={} "
                     "cpu_frame_wall_ms={:.3f} "
                     "cpu_composite_wall_ms={:.3f} capture_wall_ms={:.3f} "
                     "acquire_frame_wall_ms={:.3f} record_submit_wall_ms={:.3f} "
                     "present_call_wall_ms={:.3f} frame_end_wall_ms={:.3f}",
                     trace_frame, static_cast<int>(Settings::values.frame_pacing_mode.GetValue()),
                     static_cast<int>(Settings::values.vsync_mode.GetValue()),
                     trace_last_frame == std::chrono::steady_clock::time_point{}
                         ? -1.0 : ms(entered - trace_last_frame),
                     ms(end - entered), ms(acquiring - entered), ms(acquired - acquiring),
                     ms(presenting - acquired), ms(presented - presenting), ms(end - presented));
        }
        trace_last_frame = entered;
    }
}

HdrPresent RendererVulkan::UpdateHdrPresent() {
    const auto& monitor = Common::DisplayHdrMonitor::Instance();
    const u64 key = (monitor.Generation() << 8) |
                    (u64{static_cast<u8>(Settings::values.hdr_mode.GetValue())} << 4) |
                    static_cast<u8>(Settings::values.hdr_swapchain.GetValue());
    const u32 luminance_key = (u32{Settings::values.hdr_peak_luminance.GetValue()} << 16) |
                              Settings::values.hdr_reference_white.GetValue();
    if (key == output_format_key && luminance_key == output_luminance_key) {
        return {output_format.colorSpace, output_reference_white, output_peak};
    }
    output_format_key = key;
    output_luminance_key = luminance_key;
    const auto previous = output_format;
    output_format = ChooseOutputFormat(device.GetPhysical(), *surface,
                                       render_window.GetWindowInfo().render_surface);
    const auto told = CaptureHdrPresent();
    output_reference_white = told.reference_white;
    output_peak = told.peak;
    if (previous.format != output_format.format ||
        previous.colorSpace != output_format.colorSpace) {
        if (output_format.colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            LOG_INFO(Render_Vulkan,
                     "HDR output: {} {}, reference white {} nits, peak {} nits. Anti-aliasing, "
                     "post-processing and upscaling filters are off",
                     string_VkFormat(output_format.format),
                     string_VkColorSpaceKHR(output_format.colorSpace), output_reference_white,
                     output_peak);
        } else {
            LOG_INFO(Render_Vulkan, "HDR output: SDR {}", string_VkFormat(output_format.format));
        }
    }
    return {output_format.colorSpace, output_reference_white, output_peak};
}

HdrPresent RendererVulkan::CaptureHdrPresent() const {
    // The luminances the game was told (nx2:vi2 QueryDisplayHdrStatus / PaperWhite).
    const auto told = Common::ResolveHdrOutput(Common::DisplayHdrMonitor::Instance().Current(),
                                               Settings::HdrMode::On,
                                               Settings::values.hdr_peak_luminance.GetValue(),
                                               Settings::values.hdr_reference_white.GetValue());
    return {VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, static_cast<f32>(told.reference_white),
            static_cast<f32>(told.peak_luminance)};
}

VkSurfaceKHR RendererVulkan::PresentationSurface() {
#ifdef EGLANTINE_STREAMLINE
    // DLSS-G looks the window up by the swapchain's surface: replace the surface made before the
    // device with one made through Streamline, for the same window. Present support stays valid,
    // as it belongs to the window and the queue family.
    const auto& window_info = render_window.GetWindowInfo();
    if (Streamline::FrameGenSession()) {
        const VkSurfaceKHR registered =
            window_info.type == Core::Frontend::WindowSystemType::Windows
                ? Streamline::CreateFrameGenSurface(*instance, window_info.render_surface)
                : VK_NULL_HANDLE;
        if (registered) {
            surface = vk::SurfaceKHR(registered, *instance, instance.Dispatch());
        } else {
            Streamline::DisableFrameGen("Streamline could not make the window's surface");
        }
    }
#endif
    return *surface;
}

void RendererVulkan::Report() const {
    using namespace Common::Literals;

    const std::string api_version = GetReadableVersion(device.ApiVersion());

    const std::string extensions = BuildCommaSeparatedExtensions(device.GetAvailableExtensions());

    const auto available_vram = static_cast<f64>(device.GetDeviceLocalMemory()) / f64{1_GiB};

    LOG_INFO(Render_Vulkan, "Driver: {} {}", device.GetVendorName(), GetDriverVersion(device));
    LOG_INFO(Render_Vulkan, "Device: {}", device_info.model);
    LOG_INFO(Render_Vulkan, "Vulkan: {}", api_version);
    LOG_INFO(Render_Vulkan, "Available VRAM: {:.2f} GiB", available_vram);
    if (VideoCommon::RenderTrace::Enabled()) {
        LOG_INFO(Render_Vulkan, "Render trace enabled: interval=600 frames, log_filter={}",
                 Settings::values.log_filter.GetValue());
    }
    LOG_DEBUG(Render_Vulkan,
              "Render trace capabilities: geometry_shader={} transform_feedback={} "
              "geometry_streams={} mesh_shader={}",
              device.GetPhysical().GetFeatures().geometryShader, device.IsExtTransformFeedbackSupported(),
              device.AreTransformFeedbackGeometryStreamsSupported(), device.IsExtMeshShaderSupported());
}

vk::Buffer RendererVulkan::RenderToBuffer(std::span<const Tegra::FramebufferConfig> framebuffers,
                                          const Layout::FramebufferLayout& layout, VkFormat format,
                                          VkDeviceSize buffer_size) {
    auto& frame = screenshot_frame;
    auto dst_buffer = CreateWrappedBuffer(memory_allocator, buffer_size, MemoryUsage::Download);
    blit_capture.DrawToFrame(device, rasterizer, &frame, framebuffers, layout, 1, format,
                             CaptureHdrPresent());

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([&](vk::CommandBuffer cmdbuf) {
        DownloadColorImage(cmdbuf, *frame.image, *dst_buffer,
                           VkExtent3D{layout.width, layout.height, 1});
    });

    // Ensure the copy is fully completed before saving the capture
    scheduler.Finish();

    // Copy backing image data to the capture buffer
    dst_buffer.Invalidate();
    return dst_buffer;
}

void RendererVulkan::RenderScreenshot(std::span<const Tegra::FramebufferConfig> framebuffers) {
    if (!renderer_settings.screenshot_requested) {
        return;
    }

    const auto screenshot_layers = Tegra::FilterLayerStack(
        framebuffers, renderer_settings.screenshot_layer_stack, screenshot_layer_scratch);

    const auto& layout{renderer_settings.screenshot_framebuffer_layout};
    const auto dst_buffer = RenderToBuffer(screenshot_layers, layout, VK_FORMAT_B8G8R8A8_UNORM,
                                           layout.width * layout.height * 4);

    std::memcpy(renderer_settings.screenshot_bits, dst_buffer.Mapped().data(),
                dst_buffer.Mapped().size());
    renderer_settings.screenshot_complete_callback(false);
    renderer_settings.screenshot_requested = false;
}

std::vector<u8> RendererVulkan::GetAppletCaptureBuffer() {
    using namespace VideoCore::Capture;

    std::vector<u8> out(VideoCore::Capture::TiledSize);

    if (!applet_frame.image) {
        return out;
    }

    const auto dst_buffer =
        CreateWrappedBuffer(memory_allocator, VideoCore::Capture::TiledSize, MemoryUsage::Download);

    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([&](vk::CommandBuffer cmdbuf) {
        DownloadColorImage(cmdbuf, *applet_frame.image, *dst_buffer, CaptureImageExtent);
    });

    // Ensure the copy is fully completed before writing the capture
    scheduler.Finish();

    // Swizzle image data to the capture buffer
    dst_buffer.Invalidate();
    Tegra::Texture::SwizzleTexture(out, dst_buffer.Mapped(), BytesPerPixel, LinearWidth,
                                   LinearHeight, LinearDepth, BlockHeight, BlockDepth);

    return out;
}

void RendererVulkan::RenderAppletCaptureLayer(
    std::span<const Tegra::FramebufferConfig> framebuffers) {
    const auto capture_layers = Tegra::FilterLayerStack(
        framebuffers, Service::Nvnflinger::LayerStackId::LastFrame, applet_capture_layers);

    if (capture_layers.empty())
        return;

    if (!applet_frame.image) {
        applet_frame.image = CreateWrappedImage(memory_allocator, CaptureImageSize, CaptureFormat);
        applet_frame.image_view = CreateWrappedImageView(device, applet_frame.image, CaptureFormat);
        applet_frame.framebuffer = blit_applet.CreateFramebuffer(device,
            VideoCore::Capture::Layout, *applet_frame.image_view, CaptureFormat);
    }

    scheduler.RequestOutsideRenderPassOperationContext();
    blit_applet.DrawToFrame(device, rasterizer, &applet_frame, capture_layers, VideoCore::Capture::Layout, 1,
                            CaptureFormat, CaptureHdrPresent());
}

#ifdef EGLANTINE_STREAMLINE
VideoCore::DlssStatus RendererVulkan::GetDlssStatus() const {
    return StreamlineDlssUpscaler::Status();
}
#endif

} // namespace Vulkan
