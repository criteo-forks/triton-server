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
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "triton/common/thread_pool.h"

namespace triton { namespace server { namespace grpc {

// Sharded pool used to run unary response completion+send off the
// backend runner thread. Tasks with the same key always run on the
// same shard thread, preserving per-request ordering.
class ResponseSendPool {
 public:
  explicit ResponseSendPool(size_t shard_count)
  {
    for (size_t i = 0; i < shard_count; ++i) {
      shards_.emplace_back(new triton::common::ThreadPool(1));
    }
  }

  // Run 'task' on the shard selected by 'key'. If the pool is
  // draining, run the task inline on the caller's thread.
  void Enqueue(const void* key, std::function<void()>&& task)
  {
    if (draining_.load(std::memory_order_acquire)) {
      task();
      return;
    }
    // std::hash on pointers is the identity in libstdc++; heap pointers
    // are aligned so their low bits are zero and a plain modulo would
    // collapse every key onto shard 0. Mix the bits first (murmur3
    // finalizer).
    uintptr_t h = reinterpret_cast<uintptr_t>(key);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    const size_t shard = h % shards_.size();
    // NOTE: There is an unavoidable tiny race window here: this Enqueue
    // may pass the draining_ check just before a concurrent Drain()
    // destroys 'shards_'. This is acceptable because Drain() is only
    // called after server shutdown has quiesced inference, so no new
    // Enqueue calls are expected to race with it in practice.
    shards_[shard]->Enqueue(std::move(task));
  }

  // Complete all queued tasks and stop the shard threads. Subsequent
  // Enqueue calls run inline.
  void Drain()
  {
    draining_.store(true, std::memory_order_release);
    shards_.clear();  // ThreadPool dtor drains its queue before joining
  }

 private:
  std::atomic<bool> draining_{false};
  std::vector<std::unique_ptr<triton::common::ThreadPool>> shards_;
};

}}}  // namespace triton::server::grpc
