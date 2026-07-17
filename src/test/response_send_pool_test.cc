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

#include "gtest/gtest.h"

// Undefine the FAIL() macro inside Triton code to avoid redefine error
// from gtest. Okay as FAIL() is not used in response_send_pool
#ifdef FAIL
#undef FAIL
#endif

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "grpc/response_send_pool.h"

namespace ni = triton::server::grpc;

namespace {

TEST(ResponseSendPoolTest, OffThreadExecution)
{
  ni::ResponseSendPool pool(2);
  const std::thread::id producer_id = std::this_thread::get_id();

  std::atomic<bool> done(false);
  std::thread::id task_id;
  int key = 0;
  pool.Enqueue(&key, [&] {
    task_id = std::this_thread::get_id();
    done.store(true);
  });

  while (!done.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  EXPECT_NE(task_id, producer_id);
}

TEST(ResponseSendPoolTest, PerKeyOrderingUnderLoad)
{
  constexpr int kNumProducers = 8;
  constexpr int kNumKeys = 32;
  // Total tasks enqueued per key, summed across all producer threads.
  constexpr int kTasksPerKeyTotal = 5000;
  constexpr int kTasksPerKeyPerProducer = kTasksPerKeyTotal / kNumProducers;

  ni::ResponseSendPool pool(8);

  std::vector<int> keys(kNumKeys);
  for (int i = 0; i < kNumKeys; ++i) {
    keys[i] = i;
  }

  // Guards the "assign next sequence number, then enqueue" critical
  // section per key so the assigned sequence numbers match the order
  // in which tasks are actually queued on the key's shard.
  std::vector<std::mutex> key_producer_mtx(kNumKeys);
  std::vector<int> next_seq(kNumKeys, 0);
  std::vector<std::vector<int>> key_sequence(kNumKeys);

  std::atomic<int> remaining(
      kNumProducers * kNumKeys * kTasksPerKeyPerProducer);

  auto producer = [&]() {
    for (int t = 0; t < kTasksPerKeyPerProducer; ++t) {
      for (int k = 0; k < kNumKeys; ++k) {
        std::lock_guard<std::mutex> lk(key_producer_mtx[k]);
        const int seq = next_seq[k]++;
        pool.Enqueue(&keys[k], [&, k, seq] {
          key_sequence[k].push_back(seq);
          remaining.fetch_sub(1);
        });
      }
    }
  };

  std::vector<std::thread> producers;
  for (int i = 0; i < kNumProducers; ++i) {
    producers.emplace_back(producer);
  }
  for (auto& p : producers) {
    p.join();
  }

  while (remaining.load() > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  for (int k = 0; k < kNumKeys; ++k) {
    const auto& seq = key_sequence[k];
    ASSERT_EQ(
        seq.size(),
        static_cast<size_t>(kNumProducers * kTasksPerKeyPerProducer));
    for (size_t i = 1; i < seq.size(); ++i) {
      ASSERT_LT(seq[i - 1], seq[i]) << "ordering violated for key " << k;
    }
  }
}

TEST(ResponseSendPoolTest, SameKeySameThread)
{
  ni::ResponseSendPool pool(4);
  int key = 42;

  constexpr int kNumTasks = 200;
  std::mutex mtx;
  std::set<std::thread::id> observed_ids;
  std::atomic<int> remaining(kNumTasks);

  for (int i = 0; i < kNumTasks; ++i) {
    pool.Enqueue(&key, [&] {
      {
        std::lock_guard<std::mutex> lk(mtx);
        observed_ids.insert(std::this_thread::get_id());
      }
      remaining.fetch_sub(1);
    });
  }

  while (remaining.load() > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  EXPECT_EQ(observed_ids.size(), 1u);
}

TEST(ResponseSendPoolTest, DrainCompletesAll)
{
  ni::ResponseSendPool pool(4);
  constexpr int kNumTasks = 200;
  std::atomic<int> counter(0);

  for (int i = 0; i < kNumTasks; ++i) {
    pool.Enqueue(reinterpret_cast<void*>(static_cast<intptr_t>(i % 4)), [&] {
      std::this_thread::sleep_for(std::chrono::microseconds(200));
      counter.fetch_add(1);
    });
  }

  pool.Drain();

  EXPECT_EQ(counter.load(), kNumTasks);
}

TEST(ResponseSendPoolTest, PostDrainInline)
{
  ni::ResponseSendPool pool(2);
  pool.Drain();

  const std::thread::id caller_id = std::this_thread::get_id();
  std::thread::id task_id;
  bool executed = false;
  int key = 0;
  pool.Enqueue(&key, [&] {
    task_id = std::this_thread::get_id();
    executed = true;
  });

  EXPECT_TRUE(executed);
  EXPECT_EQ(task_id, caller_id);
}

TEST(ResponseSendPoolTest, ShardSpread)
{
  constexpr int kShardCount = 4;
  constexpr int kNumKeys = 64;

  ni::ResponseSendPool pool(kShardCount);

  std::vector<int> keys(kNumKeys);
  for (int i = 0; i < kNumKeys; ++i) {
    keys[i] = i;
  }

  std::mutex mtx;
  std::set<std::thread::id> observed_ids;
  std::atomic<int> remaining(kNumKeys);

  for (int i = 0; i < kNumKeys; ++i) {
    pool.Enqueue(&keys[i], [&] {
      {
        std::lock_guard<std::mutex> lk(mtx);
        observed_ids.insert(std::this_thread::get_id());
      }
      remaining.fetch_sub(1);
    });
  }

  while (remaining.load() > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  EXPECT_GT(observed_ids.size(), 1u);
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
