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
  // Drain commands a worker may have outstanding in its command pipe at once.
  // See the class comment: the pipe backlog is the worker's visible load, which
  // evhtp reads (FIONREAD) to place new connections on the least loaded worker.
  // 1 keeps the pipe non-empty whenever replies are pending; larger values let
  // the backlog grow with the reply rate, up to the bound.
  static constexpr size_t kMaxPendingCommands = 1;
};

// Result of WorkerQueue::Enqueue().
enum class EnqueueResult {
  kArm,       // queued; the caller must write a drain command to the pipe
  kQueued,    // queued; enough drain commands are already outstanding
  kRejected,  // depth cap reached: NOT queued, caller must drop-and-report
};

// Result of WorkerQueue::FinishBatch(), telling the drain what to do next.
enum class DrainStep {
  kIdle,       // queue empty: the drain is over
  kHandedOff,  // work remains but another drain command is already in the
               // pipe and will pick it up: return to the event loop
  kYield,      // work remains and no command is pending: the caller must
               // write one (then YieldWritten) or keep draining inline
};

// Per-worker hand-off queue, deliberately free of evhtp/logging dependencies
// so it is unit testable; http_server.cc owns the scheduling around it.
//
// evhtp's worker command pipe is a fixed-size socketpair drained one command
// per event-loop dispatch, so deferring every reply individually costs one
// send()/recv() pair each and can overflow the pipe under a reply burst.
// Instead, replies are collected here and drain commands are deferred, each
// delivering whatever is queued: the pipe then carries at most
// 'max_pending_cmds' commands per worker rather than one per reply.
//
// Why more than one command may be outstanding: the pipe backlog is also
// evhtp's load signal. evthr_pool_defer() hands each new connection to the
// worker whose command pipe holds the fewest commands (FIONREAD), so a busy
// worker used to repel new connections through its pending reply commands.
// With a single command per worker, a draining worker's pipe is empty, so it
// looks idle and attracts connections instead. Keeping up to
// 'max_pending_cmds' commands outstanding, one per queued reply while below
// the bound, restores a backlog that grows with load without letting it grow
// without bound.
//
// Scheduling protocol, which is what makes wake-ups neither lost nor
// duplicated. Two pieces of state, both under 'mu_':
//   pending_cmds_  drain commands reserved by Enqueue (and written by the
//                  caller) or by YieldWritten, not yet consumed by BeginDrain;
//   draining_      a drain is executing on the worker right now.
// Invariant: a non-empty queue always has pending_cmds_ > 0 or draining_.
// Enqueue arms (reserves a command) whenever pending_cmds_ is below the bound,
// so in particular whenever nothing is pending and no drain is running.
// FinishBatch reports kIdle only on an empty queue, kHandedOff only while a
// command is pending, and otherwise kYield with draining_ left set until the
// caller either writes the yield command (YieldWritten) or keeps draining.
// A failed arm (ArmFailed) releases its reservation and strands the queue's
// contents only if nothing else will deliver them.
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
  static constexpr size_t kDefaultMaxPendingCommands =
      ReplyQueueDefaults::kMaxPendingCommands;

  explicit WorkerQueue(
      const size_t max_batch = kDefaultMaxBatch,
      const size_t max_depth = kDefaultMaxDepth,
      const size_t max_pending_cmds = kDefaultMaxPendingCommands)
      : max_batch_(std::max<size_t>(max_batch, 1)),
        max_depth_(std::max<size_t>(max_depth, 1)),
        max_pending_cmds_(std::max<size_t>(max_pending_cmds, 1))
  {
  }

  // Append 'item' unless the depth cap is reached. kArm means a drain command
  // was reserved and the caller must write it to the worker's pipe (calling
  // ArmFailed if that fails); kQueued means enough commands are already
  // outstanding; kRejected means the item was NOT queued and the caller must
  // treat it as a dropped hand-off.
  EnqueueResult Enqueue(Item item)
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (items_.size() >= max_depth_) {
      return EnqueueResult::kRejected;
    }
    items_.push_back(std::move(item));
    if (pending_cmds_ < max_pending_cmds_) {
      ++pending_cmds_;
      return EnqueueResult::kArm;
    }
    return EnqueueResult::kQueued;
  }

  // The worker picked a drain command up: consume its reservation and mark
  // the drain as running.
  void BeginDrain()
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (pending_cmds_ > 0) {
      --pending_cmds_;
    }
    draining_ = true;
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

  // Call once a batch has been processed; see DrainStep. On kYield the drain
  // stays marked as running until YieldWritten() (the yield command is in
  // the pipe) or until the caller drains inline and calls FinishBatch again.
  DrainStep FinishBatch()
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (items_.empty()) {
      draining_ = false;
      return DrainStep::kIdle;
    }
    if (pending_cmds_ > 0) {
      draining_ = false;
      return DrainStep::kHandedOff;
    }
    return DrainStep::kYield;
  }

  // The yield command requested by a kYield FinishBatch was written to the
  // pipe: it now owns the queue's remaining work.
  void YieldWritten()
  {
    std::lock_guard<std::mutex> lk(mu_);
    ++pending_cmds_;
    draining_ = false;
  }

  // A drain command reserved by Enqueue could not be written. Releases the
  // reservation and, if nothing else will deliver the queued items (no other
  // command pending, no drain running), moves every item into 'stranded'
  // (cleared first) in the same critical section and returns true: the
  // caller must report them as dropped, since nothing can deliver them
  // afterwards. Returns false when another command or a running drain still
  // covers the queue, in which case nothing is lost and 'stranded' is empty.
  bool ArmFailed(std::vector<Item>* stranded)
  {
    stranded->clear();
    std::lock_guard<std::mutex> lk(mu_);
    if (pending_cmds_ > 0) {
      --pending_cmds_;
    }
    if ((pending_cmds_ > 0) || draining_ || items_.empty()) {
      return false;
    }
    TakeAllLocked(stranded);
    return true;
  }

  // Move every queued item into 'out' (cleared first) and mark no drain as
  // running. Used on worker shutdown; commands still in the pipe are consumed
  // harmlessly by the exiting loop.
  void TakeAll(std::vector<Item>* out)
  {
    out->clear();
    std::lock_guard<std::mutex> lk(mu_);
    TakeAllLocked(out);
    draining_ = false;
  }

  size_t Depth() const
  {
    std::lock_guard<std::mutex> lk(mu_);
    return items_.size();
  }

  // True while a drain command is outstanding or a drain is running, i.e.
  // while queued items have something that will deliver them.
  bool Armed() const
  {
    std::lock_guard<std::mutex> lk(mu_);
    return (pending_cmds_ > 0) || draining_;
  }

  size_t PendingCommands() const
  {
    std::lock_guard<std::mutex> lk(mu_);
    return pending_cmds_;
  }

  bool Draining() const
  {
    std::lock_guard<std::mutex> lk(mu_);
    return draining_;
  }

  size_t MaxBatch() const { return max_batch_; }
  size_t MaxDepth() const { return max_depth_; }
  size_t MaxPendingCommands() const { return max_pending_cmds_; }

 private:
  void TakeAllLocked(std::vector<Item>* out)
  {
    out->reserve(items_.size());
    while (!items_.empty()) {
      out->push_back(std::move(items_.front()));
      items_.pop_front();
    }
  }

  mutable std::mutex mu_;
  std::deque<Item> items_;
  size_t pending_cmds_{0};
  bool draining_{false};
  const size_t max_batch_;
  const size_t max_depth_;
  const size_t max_pending_cmds_;
};

}}  // namespace triton::server
