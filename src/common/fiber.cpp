// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <thread>
#include <mutex>

#if defined(TRACY_ENABLE) && defined(TRACY_FIBERS)
#include <atomic>
#include <cstdio>
#include <tracy/Tracy.hpp>
#endif

#include "common/assert.h"
#include "common/fiber.h"

#include <boost/context/detail/fcontext.hpp>

namespace Common {

#if defined(TRACY_ENABLE) && defined(TRACY_FIBERS)
namespace {

// Tracy may request a fiber name asynchronously after the fiber has yielded or
// been destroyed. Keep each unique, null-terminated name alive until process exit.
const char* RegisterTracyFiberName() {
    static std::atomic_uint64_t next_id{1};
    constexpr std::size_t name_capacity = 48;
    auto* name = new char[name_capacity];
    const auto id = next_id.fetch_add(1, std::memory_order_relaxed);
    std::snprintf(name, name_capacity, "Lichii Fiber %llu",
                  static_cast<unsigned long long>(id));
    return name;
}

} // namespace
#endif

#ifdef __OPENORBIS__
constexpr size_t DEFAULT_STACK_SIZE = 128 * 4096;
#else
constexpr size_t DEFAULT_STACK_SIZE = 512 * 4096;
#endif
constexpr u32 CANARY_VALUE = 0xDEADBEEF;

struct Fiber::FiberImpl {
    FiberImpl() {}

    u32 canary_1 = CANARY_VALUE;
    // Left uninitialized: the host gives a stack's pages memory only as they are used. Zero-filled
    // std::arrays (and a second stack nothing switched to) made every fiber, one per guest thread,
    // hold 4 MiB: ~750 MiB for title 1's ~190 threads.
    std::unique_ptr<u8[]> stack;
    u32 canary_2 = CANARY_VALUE;

    boost::context::detail::fcontext_t context{};
    boost::context::detail::fcontext_t rewind_context{};

    std::mutex guard;
    std::function<void()> entry_point;
    std::function<void()> rewind_point;
    std::shared_ptr<Fiber> previous_fiber;

    u8* stack_limit = nullptr;
    bool is_thread_fiber = false;
#if defined(TRACY_ENABLE) && defined(TRACY_FIBERS)
    const char* tracy_name = nullptr;
#endif
    bool released = false;
};

void Fiber::SetRewindPoint(std::function<void()>&& rewind_func) {
    impl->rewind_point = std::move(rewind_func);
}

Fiber::Fiber(std::function<void()>&& entry_point_func) : impl{std::make_unique<FiberImpl>()} {
    impl->entry_point = std::move(entry_point_func);
    impl->stack.reset(new u8[DEFAULT_STACK_SIZE]);
#if defined(TRACY_ENABLE) && defined(TRACY_FIBERS)
    impl->tracy_name = RegisterTracyFiberName();
#endif
    impl->stack_limit = impl->stack.get();
    u8* stack_base = impl->stack_limit + DEFAULT_STACK_SIZE;
    impl->context = boost::context::detail::make_fcontext(stack_base, DEFAULT_STACK_SIZE, [](boost::context::detail::transfer_t transfer) -> void {
        auto* fiber = static_cast<Fiber*>(transfer.data);
        ASSERT(fiber && fiber->impl && fiber->impl->previous_fiber && fiber->impl->previous_fiber->impl);
        ASSERT(fiber->impl->canary_1 == CANARY_VALUE);
        ASSERT(fiber->impl->canary_2 == CANARY_VALUE);
        fiber->impl->previous_fiber->impl->context = transfer.fctx;
        fiber->impl->previous_fiber->impl->guard.unlock();
        fiber->impl->previous_fiber.reset();
        fiber->impl->entry_point();
        UNREACHABLE();
    });
}

Fiber::Fiber() : impl{std::make_unique<FiberImpl>()} {}

Fiber::~Fiber() {
    if (!impl->released) {
        // Make sure the Fiber is not being used
        const bool locked = impl->guard.try_lock();
        ASSERT_MSG(locked, "Destroying a fiber that's still running");
        if (locked) {
            impl->guard.unlock();
        }
    }
}

void Fiber::Exit() {
    ASSERT_MSG(impl->is_thread_fiber, "Exiting non main thread fiber");
    if (impl->is_thread_fiber) {
        impl->guard.unlock();
        impl->released = true;
    }
}

void Fiber::YieldTo(std::weak_ptr<Fiber> weak_from, Fiber& to) {
    to.impl->guard.lock();
    to.impl->previous_fiber = weak_from.lock();

#if defined(TRACY_ENABLE) && defined(TRACY_FIBERS)
    // Entering a fiber tells Tracy which logical execution context is now
    // active. Switching directly from one fiber to another is represented by
    // another Enter (Tracy closes the previous fiber segment itself). Leave
    // is only sent when control returns to the OS thread's host context.
    if (to.impl->is_thread_fiber) {
        TracyFiberLeave;
    } else {
        ASSERT(to.impl->tracy_name != nullptr);
        TracyFiberEnter(to.impl->tracy_name);
    }
#endif

    auto transfer = boost::context::detail::jump_fcontext(to.impl->context, &to);
    // "from" might no longer be valid if the thread was killed
    if (auto from = weak_from.lock()) {
        if (from->impl->previous_fiber == nullptr) {
            ASSERT(false && "previous_fiber is nullptr!");
        } else {
            from->impl->previous_fiber->impl->context = transfer.fctx;
            from->impl->previous_fiber->impl->guard.unlock();
            from->impl->previous_fiber.reset();
        }
    }
}

std::shared_ptr<Fiber> Fiber::ThreadToFiber() {
    std::shared_ptr<Fiber> fiber = std::shared_ptr<Fiber>{new Fiber()};
    fiber->impl->guard.lock();
    fiber->impl->is_thread_fiber = true;
    return fiber;
}

} // namespace Common
