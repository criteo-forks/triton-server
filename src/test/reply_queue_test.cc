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

#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "http_reply_queue.h"

namespace {

using triton::server::DrainStep;
using triton::server::EnqueueResult;
using triton::server::WorkerQueue;

using IntQueue = WorkerQueue<int>;

// Mirrors what the worker does when it picks a drain command up: consume the
// command, deliver batches until the queue is idle or another command owns
// the rest. Returns the number of items delivered.
size_t
RunDrain(IntQueue* queue, std::vector<int>* delivered = nullptr)
{
  queue->BeginDrain();
  std::vector<int> batch;
  size_t n = 0;
  while (true) {
    queue->TakeBatch(&batch);
    n += batch.size();
    if (delivered != nullptr) {
      delivered->insert(delivered->end(), batch.begin(), batch.end());
    }
    switch (queue->FinishBatch()) {
      case DrainStep::kIdle:
      case DrainStep::kHandedOff:
        return n;
      case DrainStep::kYield:
        // The real drain writes a yield command here; stand in for it being
        // written and immediately consumed.
        queue->YieldWritten();
        queue->BeginDrain();
        break;
    }
  }
}

TEST(WorkerQueueTest, FirstEnqueueArmsAndFurtherOnesDoNot)
{
  IntQueue queue;  // default: one outstanding command
  EXPECT_FALSE(queue.Armed());
  EXPECT_EQ(queue.Enqueue(1), EnqueueResult::kArm);
  EXPECT_TRUE(queue.Armed());
  EXPECT_EQ(queue.PendingCommands(), 1u);
  // A drain command is already pending, so these must not schedule another.
  EXPECT_EQ(queue.Enqueue(2), EnqueueResult::kQueued);
  EXPECT_EQ(queue.Enqueue(3), EnqueueResult::kQueued);
  EXPECT_EQ(queue.Depth(), 3u);
  EXPECT_EQ(queue.PendingCommands(), 1u);
}

TEST(WorkerQueueTest, PendingCommandsGrowWithLoadUpToTheBound)
{
  IntQueue queue(256, 12500, 3 /* max_pending_cmds */);
  // The first three enqueues each reserve a command: the worker's pipe
  // backlog, which evhtp reads to place connections, tracks the load.
  EXPECT_EQ(queue.Enqueue(0), EnqueueResult::kArm);
  EXPECT_EQ(queue.Enqueue(1), EnqueueResult::kArm);
  EXPECT_EQ(queue.Enqueue(2), EnqueueResult::kArm);
  EXPECT_EQ(queue.PendingCommands(), 3u);
  // At the bound the pipe stops growing regardless of the reply rate.
  EXPECT_EQ(queue.Enqueue(3), EnqueueResult::kQueued);
  EXPECT_EQ(queue.Enqueue(4), EnqueueResult::kQueued);
  EXPECT_EQ(queue.PendingCommands(), 3u);
  EXPECT_EQ(queue.Depth(), 5u);
}

TEST(WorkerQueueTest, FirstCommandDrainsEverythingAndLaterOnesFindItEmpty)
{
  IntQueue queue(256, 12500, 3);
  for (int i = 0; i < 5; ++i) {
    queue.Enqueue(i);
  }
  ASSERT_EQ(queue.PendingCommands(), 3u);
  std::vector<int> delivered;
  EXPECT_EQ(RunDrain(&queue, &delivered), 5u);
  EXPECT_EQ(queue.PendingCommands(), 2u);
  EXPECT_FALSE(queue.Draining());
  // The two surplus commands still in the pipe are cheap no-ops.
  EXPECT_EQ(RunDrain(&queue), 0u);
  EXPECT_EQ(RunDrain(&queue), 0u);
  EXPECT_EQ(queue.PendingCommands(), 0u);
  EXPECT_FALSE(queue.Armed());
  ASSERT_EQ(delivered.size(), 5u);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(delivered[i], i);
  }
}

TEST(WorkerQueueTest, EnqueueDuringADrainKeepsThePipeNonEmpty)
{
  IntQueue queue;  // bound 1
  queue.Enqueue(1);
  queue.BeginDrain();
  EXPECT_EQ(queue.PendingCommands(), 0u);
  EXPECT_TRUE(queue.Draining());
  std::vector<int> batch;
  queue.TakeBatch(&batch);
  // A producer arriving mid-drain must be told to write a command: with the
  // pipe empty this worker would otherwise look idle to evhtp's connection
  // placement while it is in fact the busiest.
  EXPECT_EQ(queue.Enqueue(2), EnqueueResult::kArm);
  EXPECT_EQ(queue.PendingCommands(), 1u);
  // That command now owns the remaining work: the drain hands off instead of
  // yielding a second command.
  EXPECT_EQ(queue.FinishBatch(), DrainStep::kHandedOff);
  EXPECT_FALSE(queue.Draining());
  EXPECT_TRUE(queue.Armed());
  EXPECT_EQ(RunDrain(&queue), 1u);
  EXPECT_FALSE(queue.Armed());
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

TEST(WorkerQueueTest, BatchIsCappedAndTheDrainYieldsBetweenBatches)
{
  IntQueue queue(2 /* max_batch */);
  for (int i = 0; i < 5; ++i) {
    queue.Enqueue(i);
  }
  queue.BeginDrain();
  std::vector<int> batch;
  queue.TakeBatch(&batch);
  EXPECT_EQ(batch.size(), 2u);
  EXPECT_EQ(queue.Depth(), 3u);
  // Work remains and no command is pending: the drain must yield through the
  // pipe (or keep draining inline), and stays marked as running meanwhile.
  EXPECT_EQ(queue.FinishBatch(), DrainStep::kYield);
  EXPECT_TRUE(queue.Draining());
  EXPECT_TRUE(queue.Armed());
  queue.YieldWritten();
  EXPECT_FALSE(queue.Draining());
  EXPECT_EQ(queue.PendingCommands(), 1u);
  EXPECT_EQ(RunDrain(&queue), 3u);
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

TEST(WorkerQueueTest, FinishBatchGoesIdleWhenDrained)
{
  IntQueue queue;
  queue.Enqueue(1);
  queue.BeginDrain();
  std::vector<int> batch;
  queue.TakeBatch(&batch);
  EXPECT_EQ(queue.FinishBatch(), DrainStep::kIdle);
  EXPECT_FALSE(queue.Armed());
  // Having gone idle, the next enqueue must schedule a fresh drain.
  EXPECT_EQ(queue.Enqueue(2), EnqueueResult::kArm);
}

TEST(WorkerQueueTest, FailedArmStrandsOnlyWhenNothingElseCoversTheQueue)
{
  IntQueue queue;
  EXPECT_EQ(queue.Enqueue(1), EnqueueResult::kArm);
  EXPECT_EQ(queue.Enqueue(2), EnqueueResult::kQueued);
  // The only command could not be written: everything queued is stranded and
  // must be taken out (and reported by the caller) in one atomic step, so
  // nothing can be delivered later and contradict the report.
  std::vector<int> stranded;
  EXPECT_TRUE(queue.ArmFailed(&stranded));
  EXPECT_EQ(stranded.size(), 2u);
  EXPECT_EQ(queue.Depth(), 0u);
  EXPECT_FALSE(queue.Armed());
  // The next producer starts a fresh arm attempt.
  EXPECT_EQ(queue.Enqueue(3), EnqueueResult::kArm);
}

TEST(WorkerQueueTest, FailedArmIsHarmlessWhileAnotherCommandIsPending)
{
  IntQueue queue(256, 12500, 2);
  EXPECT_EQ(queue.Enqueue(1), EnqueueResult::kArm);
  EXPECT_EQ(queue.Enqueue(2), EnqueueResult::kArm);
  std::vector<int> stranded{42};
  // The second command failed but the first is in the pipe: nothing is lost
  // and nothing may be reported.
  EXPECT_FALSE(queue.ArmFailed(&stranded));
  EXPECT_TRUE(stranded.empty());
  EXPECT_EQ(queue.PendingCommands(), 1u);
  EXPECT_EQ(queue.Depth(), 2u);
  EXPECT_EQ(RunDrain(&queue), 2u);
}

TEST(WorkerQueueTest, FailedArmIsHarmlessWhileADrainIsRunning)
{
  IntQueue queue;
  queue.Enqueue(1);
  queue.BeginDrain();
  std::vector<int> batch;
  queue.TakeBatch(&batch);
  EXPECT_EQ(queue.Enqueue(2), EnqueueResult::kArm);
  std::vector<int> stranded;
  // The running drain will see item 2 at its next FinishBatch.
  EXPECT_FALSE(queue.ArmFailed(&stranded));
  EXPECT_TRUE(stranded.empty());
  EXPECT_EQ(queue.FinishBatch(), DrainStep::kYield);
  queue.YieldWritten();
  EXPECT_EQ(RunDrain(&queue), 1u);
}

TEST(WorkerQueueTest, TakeAllDrainsAndStopsTheDrain)
{
  IntQueue queue(2);
  for (int i = 0; i < 5; ++i) {
    queue.Enqueue(i);
  }
  queue.BeginDrain();
  std::vector<int> all;
  queue.TakeAll(&all);
  EXPECT_EQ(all.size(), 5u);
  EXPECT_EQ(queue.Depth(), 0u);
  EXPECT_FALSE(queue.Draining());
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
  EXPECT_EQ(queue.PendingCommands(), 1u);
  // Draining below the cap admits producers again.
  std::vector<int> batch;
  queue.TakeBatch(&batch);
  EXPECT_EQ(batch.size(), 2u);
  EXPECT_EQ(queue.Enqueue(4), EnqueueResult::kQueued);
  EXPECT_EQ(queue.Depth(), 2u);
}

TEST(WorkerQueueTest, BoundsAreAtLeastOne)
{
  IntQueue queue(0, 0, 0);
  EXPECT_EQ(queue.MaxBatch(), 1u);
  EXPECT_EQ(queue.MaxDepth(), 1u);
  EXPECT_EQ(queue.MaxPendingCommands(), 1u);
  EXPECT_EQ(queue.Enqueue(1), EnqueueResult::kArm);
  EXPECT_EQ(queue.Enqueue(2), EnqueueResult::kRejected);
  std::vector<int> batch;
  queue.TakeBatch(&batch);
  EXPECT_EQ(batch.size(), 1u);
}

// Stand-in for the worker command pipe: producers write commands, the worker
// consumes them one at a time and runs a drain per command, as the event loop
// does. Tracks how many commands were ever in flight at once.
class FakePipe {
 public:
  explicit FakePipe(IntQueue* queue) : queue_(queue) {}

  void Write()
  {
    std::lock_guard<std::mutex> lk(mu_);
    ++written_;
    ++in_pipe_;
    max_in_pipe_ = std::max(max_in_pipe_, in_pipe_);
  }

  bool ConsumeOne(std::atomic<int>* delivered)
  {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (in_pipe_ == 0) {
        return false;
      }
      --in_pipe_;
    }
    queue_->BeginDrain();
    std::vector<int> batch;
    while (true) {
      queue_->TakeBatch(&batch);
      *delivered += static_cast<int>(batch.size());
      switch (queue_->FinishBatch()) {
        case DrainStep::kIdle:
        case DrainStep::kHandedOff:
          return true;
        case DrainStep::kYield:
          queue_->YieldWritten();
          Write();
          return true;  // the yield command continues on the next consume
      }
    }
  }

  int written() const { return written_; }
  int in_pipe() const { return in_pipe_; }
  int max_in_pipe() const { return max_in_pipe_; }

 private:
  IntQueue* queue_;
  std::mutex mu_;
  int written_{0};
  int in_pipe_{0};
  int max_in_pipe_{0};
};

// The invariant that matters under load: no item is ever left queued with
// nothing to deliver it (a lost wake-up would strand replies and leak the
// model references they hold), the number of outstanding commands never
// exceeds the bound, and far fewer commands than replies are written.
TEST(WorkerQueueTest, ConcurrentProducersNeverStrandItems)
{
  constexpr int kProducers = 8;
  constexpr int kPerProducer = 2000;
  constexpr size_t kMaxCmds = 4;
  IntQueue queue(
      64, 1u << 20 /* effectively uncapped for this test */, kMaxCmds);
  FakePipe pipe(&queue);
  std::atomic<int> delivered{0};
  std::atomic<bool> stop{false};

  std::thread worker([&]() {
    while (!stop.load()) {
      if (!pipe.ConsumeOne(&delivered)) {
        std::this_thread::yield();
      }
    }
    while (pipe.ConsumeOne(&delivered)) {
    }
  });

  std::vector<std::thread> producers;
  for (int p = 0; p < kProducers; ++p) {
    producers.emplace_back([&]() {
      for (int i = 0; i < kPerProducer; ++i) {
        if (queue.Enqueue(i) == EnqueueResult::kArm) {
          pipe.Write();
        }
      }
    });
  }
  for (auto& producer : producers) {
    producer.join();
  }
  stop.store(true);
  worker.join();

  EXPECT_EQ(delivered.load(), kProducers * kPerProducer);
  EXPECT_EQ(queue.Depth(), 0u);
  EXPECT_FALSE(queue.Armed());
  EXPECT_EQ(pipe.in_pipe(), 0);
  // The bound holds, so replies alone can never fill the pipe. A yield may
  // briefly coexist with a full set of arms, hence the +1.
  EXPECT_LE(pipe.max_in_pipe(), static_cast<int>(kMaxCmds) + 1);
  // Batching is the point: far fewer commands than replies.
  EXPECT_GT(pipe.written(), 0);
  EXPECT_LT(pipe.written(), kProducers * kPerProducer);
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
