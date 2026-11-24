// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-License-Identifier: GPL-2.0-or-later

#include <thread>
#include <mutex>

#if defined(TRACY_ENABLE) && defined(TRACY_FIBERS)
#include <tracy/Tracy.hpp>
#endif

#include "common/assert.h"
#include "common/fiber.h"

#include <boost/context/detail/fcontext.hpp>

namespace Common {

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
    bool released = false;
};

void Fiber::SetRewindPoint(std::function<void()>&& rewind_func) {
    impl->rewind_point = std::move(rewind_func);
}

Fiber::Fiber(std::function<void()>&& entry_point_func) : impl{std::make_unique<FiberImpl>()} {
    impl->entry_point = std::move(entry_point_func);
    impl->stack.reset(new u8[DEFAULT_STACK_SIZE]);
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
    auto from = weak_from.lock();
    to.impl->previous_fiber = from;

#if defined(TRACY_ENABLE) && defined(TRACY_FIBERS)
    // A context switch has two independent sides: leave the currently running
    // guest/scheduler fiber (if any), then enter the destination fiber (if it
    // is not the OS thread's host context). The scheduler may switch directly
    // between guest fibers through its switch fiber, so checking only the
    // destination would leave the previous fiber active and corrupt Tracy's
    // per-fiber zone stack.
    if (from && !from->impl->is_thread_fiber) {
        TracyFiberLeave;
    }
    if (!to.impl->is_thread_fiber) {
        // Tracy uses this pointer value as the fiber identity; it does not
        // dereference the pointer as a string. FiberImpl remains alive while
        // its context can be resumed.
        TracyFiberEnter(reinterpret_cast<const char*>(to.impl.get()));
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
