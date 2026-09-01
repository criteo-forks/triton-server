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
// worker". It is set only by Enqueue() and cleared only by FinishBatch(),
// both under 'mu_', and FinishBatch() refuses to clear it while items
// remain. So a queue is never left non-empty with no drain pending.
template <typename Item>
class WorkerQueue {
 public:
  static constexpr size_t kDefaultMaxBatch = 256;

  explicit WorkerQueue(const size_t max_batch = kDefaultMaxBatch)
      : max_batch_(std::max<size_t>(max_batch, 1))
  {
  }

  // Append 'item'. Returns true if the caller must schedule a drain because
  // the queue was idle; false if a drain is already pending, in which case
  // the caller does nothing further.
  bool Enqueue(Item item)
  {
    std::lock_guard<std::mutex> lk(mu_);
    items_.push_back(std::move(item));
    if (armed_) {
      return false;
    }
    armed_ = true;
    return true;
  }

  // Scheduling a drain failed: let the next Enqueue() try to schedule again.
  // The queued items stay queued.
  void Disarm()
  {
    std::lock_guard<std::mutex> lk(mu_);
    armed_ = false;
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

  // Move every queued item into 'out' (cleared first) and disarm, for
  // shutdown accounting.
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

 private:
  mutable std::mutex mu_;
  std::deque<Item> items_;
  bool armed_{false};
  const size_t max_batch_;
};

}}  // namespace triton::server
