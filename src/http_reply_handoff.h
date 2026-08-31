// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once

#include <evhtp/evhtp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace triton { namespace server {

// Reply hand-off primitives, free of logging/metrics so they are unit
// testable; http_server.cc wraps them with telemetry (see DeferHandoff).

using DeferFn = evthr_res (*)(evthr_t*, evthr_cb, void*);

// Attempt 'defer_fn' until it stops returning EVTHR_RES_RETRY or 'budget_ms'
// elapses, sleeping with exponential backoff (100us doubling to 50ms)
// between attempts. A budget <= 0 makes exactly one attempt. If 'waited_us'
// is non-null it receives the total time spent retrying (0 when the first
// attempt decides).
inline evthr_res
RetryDefer(
    evthr_t* thread, evthr_cb callback, void* arg, const int64_t budget_ms,
    DeferFn defer_fn, int64_t* waited_us = nullptr)
{
  if (waited_us != nullptr) {
    *waited_us = 0;
  }
  evthr_res res = defer_fn(thread, callback, arg);
  if ((res != EVTHR_RES_RETRY) || (budget_ms <= 0)) {
    return res;
  }
  const auto start = std::chrono::steady_clock::now();
  const auto deadline = start + std::chrono::milliseconds(budget_ms);
  int64_t sleep_us = 100;
  while (std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
    sleep_us = std::min<int64_t>(sleep_us * 2, 50000);
    res = defer_fn(thread, callback, arg);
    if (res != EVTHR_RES_RETRY) {
      break;
    }
  }
  if (waited_us != nullptr) {
    *waited_us = std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - start)
                     .count();
  }
  return res;
}

// Tracks consecutive whole-budget exhaustions per worker thread; a worker
// that exhausts 'threshold' budgets in a row is presumed dead or wedged and
// callers should fail fast instead of sleeping on it. Thread-safe; Open()
// and Ok() are lock-free while no exhaustion is outstanding (the hot path).
class WorkerCircuit {
 public:
  explicit WorkerCircuit(const int threshold = 3) : threshold_(threshold) {}

  bool Open(evthr_t* thread)
  {
    if (!any_.load(std::memory_order_relaxed)) {
      return false;
    }
    std::lock_guard<std::mutex> lk(mu_);
    const auto it = exhaustions_.find(thread);
    return (it != exhaustions_.end()) && (it->second >= threshold_);
  }

  void Ok(evthr_t* thread)
  {
    if (!any_.load(std::memory_order_relaxed)) {
      return;
    }
    std::lock_guard<std::mutex> lk(mu_);
    exhaustions_.erase(thread);
    if (exhaustions_.empty()) {
      any_.store(false, std::memory_order_relaxed);
    }
  }

  // Record one exhausted budget; returns the consecutive count for 'thread'.
  int Exhausted(evthr_t* thread)
  {
    std::lock_guard<std::mutex> lk(mu_);
    any_.store(true, std::memory_order_relaxed);
    return ++exhaustions_[thread];
  }

  int Threshold() const { return threshold_; }

 private:
  const int threshold_;
  std::atomic<bool> any_{false};
  std::mutex mu_;
  std::unordered_map<evthr_t*, int> exhaustions_;
};

}}  // namespace triton::server
