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

#include <algorithm>
#include <cstddef>
#include <deque>
#include <mutex>
#include <utility>
#include <vector>

namespace triton { namespace server {

// Defaults shared by WorkerQueue and the env-var plumbing that tunes it.
struct ReplyQueueDefaults {
  static constexpr size_t kMaxBatch = 256;
  // The old bound was the 208KiB socketpair over 17-byte commands (~12500
  // in-flight hand-offs); keep the same order of magnitude by default.
  static constexpr size_t kMaxDepth = 12500;
};

// Result of WorkerQueue::Enqueue().
enum class EnqueueResult {
  kArm,       // queued, and the caller must schedule a drain (queue was idle)
  kQueued,    // queued behind a drain that is already scheduled or running
  kRejected,  // depth cap reached: NOT queued, caller must drop-and-report
};

// Per-worker hand-off queue, deliberately free of evhtp/logging dependencies
// so it is unit testable; http_server.cc owns the scheduling around it.
//
// evhtp's worker command pipe is a fixed-size socketpair drained one command
// per event-loop dispatch, so deferring every reply individually costs one
// send()/recv() pair each and can overflow the pipe under a reply burst.
// Instead, replies are collected here and a *single* drain command is
// deferred per batch: the pipe then carries one command per batch rather
// than one per reply.
//
// Scheduling protocol, which is what makes wake-ups neither lost nor
// duplicated: 'armed_' means "a drain is scheduled or running for this
// worker". It is set only by Enqueue() and cleared only by FinishBatch()
// (when empty) or TakeAll() (which empties), all under 'mu_'. So the queue
// is never non-empty with no drain pending, and conversely a non-empty
// queue is always armed.
//
// The depth cap stands in for the bound the old command pipe imposed: an
// unbounded queue would let a stalled worker accumulate replies (each
// holding model references) invisibly and forever. At the cap, Enqueue()
// rejects and the caller reports the drop, restoring the pre-batching
// pipe-full semantics.
template <typename Item>
class WorkerQueue {
 public:
  static constexpr size_t kDefaultMaxBatch = ReplyQueueDefaults::kMaxBatch;
  static constexpr size_t kDefaultMaxDepth = ReplyQueueDefaults::kMaxDepth;

  explicit WorkerQueue(
      const size_t max_batch = kDefaultMaxBatch,
      const size_t max_depth = kDefaultMaxDepth)
      : max_batch_(std::max<size_t>(max_batch, 1)),
        max_depth_(std::max<size_t>(max_depth, 1))
  {
  }

  // Append 'item' unless the depth cap is reached. kArm means the queue was
  // idle and the caller must schedule a drain; kQueued means a drain is
  // already pending; kRejected means the item was NOT queued and the caller
  // must treat it as a dropped hand-off.
  EnqueueResult Enqueue(Item item)
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (items_.size() >= max_depth_) {
      return EnqueueResult::kRejected;
    }
    items_.push_back(std::move(item));
    if (armed_) {
      return EnqueueResult::kQueued;
    }
    armed_ = true;
    return EnqueueResult::kArm;
  }

  // Move up to max_batch() items into 'out' (cleared first). The batch cap
  // bounds how long one drain occupies the worker's event loop, so its other
  // connections are not starved by a deep backlog.
  void TakeBatch(std::vector<Item>* out)
  {
    out->clear();
    std::lock_guard<std::mutex> lk(mu_);
    const size_t count = std::min(max_batch_, items_.size());
    out->reserve(count);
    for (size_t i = 0; i < count; ++i) {
      out->push_back(std::move(items_.front()));
      items_.pop_front();
    }
  }

  // Call once a batch has been processed. Returns true if work remains and
  // the caller must drain again (the queue stays armed); false if the queue
  // is now idle (disarmed).
  bool FinishBatch()
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (!items_.empty()) {
      return true;
    }
    armed_ = false;
    return false;
  }

  // Move every queued item into 'out' (cleared first) and disarm, in one
  // critical section. Used when the drain could not be scheduled (the taken
  // items must then be reported as dropped, since nothing will deliver
  // them) and on worker shutdown.
  void TakeAll(std::vector<Item>* out)
  {
    out->clear();
    std::lock_guard<std::mutex> lk(mu_);
    out->reserve(items_.size());
    while (!items_.empty()) {
      out->push_back(std::move(items_.front()));
      items_.pop_front();
    }
    armed_ = false;
  }

  size_t Depth() const
  {
    std::lock_guard<std::mutex> lk(mu_);
    return items_.size();
  }

  bool Armed() const
  {
    std::lock_guard<std::mutex> lk(mu_);
    return armed_;
  }

  size_t MaxBatch() const { return max_batch_; }
  size_t MaxDepth() const { return max_depth_; }

 private:
  mutable std::mutex mu_;
  std::deque<Item> items_;
  bool armed_{false};
  const size_t max_batch_;
  const size_t max_depth_;
};

}}  // namespace triton::server
