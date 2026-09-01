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

#include <atomic>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "http_reply_queue.h"

namespace {

using triton::server::EnqueueResult;
using triton::server::WorkerQueue;

using IntQueue = WorkerQueue<int>;

TEST(WorkerQueueTest, FirstEnqueueArmsAndFurtherOnesDoNot)
{
  IntQueue queue;
  EXPECT_FALSE(queue.Armed());
  EXPECT_EQ(queue.Enqueue(1), EnqueueResult::kArm);
  EXPECT_TRUE(queue.Armed());
  // A drain is already pending, so these must not schedule another one.
  EXPECT_EQ(queue.Enqueue(2), EnqueueResult::kQueued);
  EXPECT_EQ(queue.Enqueue(3), EnqueueResult::kQueued);
  EXPECT_EQ(queue.Depth(), 3u);
}

TEST(WorkerQueueTest, BatchIsDeliveredInFifoOrder)
{
  IntQueue queue;
  for (int i = 0; i < 5; ++i) {
    queue.Enqueue(i);
  }
  std::vector<int> batch;
  queue.TakeBatch(&batch);
  ASSERT_EQ(batch.size(), 5u);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(batch[i], i);
  }
  EXPECT_EQ(queue.Depth(), 0u);
}

TEST(WorkerQueueTest, BatchIsCappedSoTheEventLoopIsNotMonopolised)
{
  IntQueue queue(2 /* max_batch */);
  for (int i = 0; i < 5; ++i) {
    queue.Enqueue(i);
  }
  std::vector<int> batch;
  queue.TakeBatch(&batch);
  EXPECT_EQ(batch.size(), 2u);
  EXPECT_EQ(queue.Depth(), 3u);
  // Work remains, so the queue must stay armed and ask for another drain.
  EXPECT_TRUE(queue.FinishBatch());
  EXPECT_TRUE(queue.Armed());
}

TEST(WorkerQueueTest, TakeBatchClearsPriorContents)
{
  IntQueue queue;
  queue.Enqueue(7);
  std::vector<int> batch{99, 98};
  queue.TakeBatch(&batch);
  ASSERT_EQ(batch.size(), 1u);
  EXPECT_EQ(batch[0], 7);
}

TEST(WorkerQueueTest, FinishBatchDisarmsWhenDrained)
{
  IntQueue queue;
  queue.Enqueue(1);
  std::vector<int> batch;
  queue.TakeBatch(&batch);
  EXPECT_FALSE(queue.FinishBatch());
  EXPECT_FALSE(queue.Armed());
  // Having gone idle, the next enqueue must schedule a fresh drain.
  EXPECT_EQ(queue.Enqueue(2), EnqueueResult::kArm);
}

TEST(WorkerQueueTest, FinishBatchStaysArmedWhenWorkArrivedDuringTheBatch)
{
  IntQueue queue;
  queue.Enqueue(1);
  std::vector<int> batch;
  queue.TakeBatch(&batch);
  // Producer pushes while the drain is running: must not schedule a second
  // drain, and must not let the queue go idle with an item still in it.
  EXPECT_EQ(queue.Enqueue(2), EnqueueResult::kQueued);
  EXPECT_TRUE(queue.FinishBatch());
  EXPECT_TRUE(queue.Armed());
  EXPECT_EQ(queue.Depth(), 1u);
}

TEST(WorkerQueueTest, TakeAllAfterFailedArmAllowsRearm)
{
  IntQueue queue;
  EXPECT_EQ(queue.Enqueue(1), EnqueueResult::kArm);
  EXPECT_EQ(queue.Enqueue(2), EnqueueResult::kQueued);
  // Scheduling the drain failed: everything queued is stranded and must be
  // taken out (and reported by the caller) in one atomic step, so nothing
  // can be delivered later and contradict the report.
  std::vector<int> stranded;
  queue.TakeAll(&stranded);
  EXPECT_EQ(stranded.size(), 2u);
  EXPECT_EQ(queue.Depth(), 0u);
  EXPECT_FALSE(queue.Armed());
  // The next producer starts a fresh arm attempt.
  EXPECT_EQ(queue.Enqueue(3), EnqueueResult::kArm);
}

TEST(WorkerQueueTest, TakeAllDrainsAndDisarms)
{
  IntQueue queue(2);
  for (int i = 0; i < 5; ++i) {
    queue.Enqueue(i);
  }
  std::vector<int> all;
  queue.TakeAll(&all);
  EXPECT_EQ(all.size(), 5u);
  EXPECT_EQ(queue.Depth(), 0u);
  EXPECT_FALSE(queue.Armed());
}

TEST(WorkerQueueTest, DepthCapRejectsWithoutQueueing)
{
  IntQueue queue(2 /* max_batch */, 3 /* max_depth */);
  EXPECT_EQ(queue.Enqueue(0), EnqueueResult::kArm);
  EXPECT_EQ(queue.Enqueue(1), EnqueueResult::kQueued);
  EXPECT_EQ(queue.Enqueue(2), EnqueueResult::kQueued);
  // At the cap: rejected, not queued, and the armed state is untouched (a
  // non-empty queue is always armed, so a rejection never needs to arm).
  EXPECT_EQ(queue.Enqueue(3), EnqueueResult::kRejected);
  EXPECT_EQ(queue.Depth(), 3u);
  EXPECT_TRUE(queue.Armed());
  // Draining below the cap admits producers again.
  std::vector<int> batch;
  queue.TakeBatch(&batch);
  EXPECT_EQ(batch.size(), 2u);
  EXPECT_EQ(queue.Enqueue(4), EnqueueResult::kQueued);
  EXPECT_EQ(queue.Depth(), 2u);
}

TEST(WorkerQueueTest, MaxBatchAndDepthAreAtLeastOne)
{
  IntQueue queue(0, 0);
  EXPECT_EQ(queue.MaxBatch(), 1u);
  EXPECT_EQ(queue.MaxDepth(), 1u);
  EXPECT_EQ(queue.Enqueue(1), EnqueueResult::kArm);
  EXPECT_EQ(queue.Enqueue(2), EnqueueResult::kRejected);
  std::vector<int> batch;
  queue.TakeBatch(&batch);
  EXPECT_EQ(batch.size(), 1u);
}

// The invariant that matters under load: exactly one producer is told to
// schedule a drain per idle period, and no item is ever left queued with the
// queue disarmed (a lost wake-up would strand replies and leak the model
// references they hold).
TEST(WorkerQueueTest, ConcurrentProducersNeverStrandItems)
{
  constexpr int kProducers = 8;
  constexpr int kPerProducer = 2000;
  IntQueue queue(64, 1u << 20 /* effectively uncapped for this test */);
  std::atomic<int> arm_requests{0};
  std::atomic<int> delivered{0};

  auto drain = [&]() {
    std::vector<int> batch;
    while (true) {
      queue.TakeBatch(&batch);
      delivered += static_cast<int>(batch.size());
      if (!queue.FinishBatch()) {
        return;
      }
    }
  };

  std::vector<std::thread> producers;
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&]() {
      for (int i = 0; i < kPerProducer; ++i) {
        if (queue.Enqueue(i) == EnqueueResult::kArm) {
          ++arm_requests;
          // Stand in for the worker thread picking the drain command up.
          drain();
        }
      }
    });
  }
  for (auto& producer : producers) {
    producer.join();
  }
  if (queue.Depth() > 0) {
    drain();
  }

  EXPECT_EQ(delivered.load(), kProducers * kPerProducer);
  EXPECT_EQ(queue.Depth(), 0u);
  EXPECT_FALSE(queue.Armed());
  // Batching is the point: far fewer drain commands than replies.
  EXPECT_GT(arm_requests.load(), 0);
  EXPECT_LT(arm_requests.load(), kProducers * kPerProducer);
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
