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

#include "gtest/gtest.h"
#include "http_reply_handoff.h"

namespace {

using triton::server::RetryDefer;
using triton::server::WorkerCircuit;

// RetryDefer takes a plain function pointer, so the fakes coordinate
// through file-scope state reset by the fixture.
int g_call_count = 0;
int g_succeed_after = 0;  // number of EVTHR_RES_RETRY results before OK
bool g_fatal_on_second = false;

evthr_res
FakeDefer(evthr_t*, evthr_cb, void*)
{
  ++g_call_count;
  if (g_fatal_on_second && (g_call_count >= 2)) {
    return EVTHR_RES_FATAL;
  }
  if (g_call_count > g_succeed_after) {
    return EVTHR_RES_OK;
  }
  return EVTHR_RES_RETRY;
}

evthr_res
AlwaysRetry(evthr_t*, evthr_cb, void*)
{
  ++g_call_count;
  return EVTHR_RES_RETRY;
}

void
NoopCallback(evthr_t*, void*, void*)
{
}

evthr_t*
FakeThread(const uintptr_t id)
{
  return reinterpret_cast<evthr_t*>(id);
}

class RetryDeferTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    g_call_count = 0;
    g_succeed_after = 0;
    g_fatal_on_second = false;
  }
};

TEST_F(RetryDeferTest, FirstAttemptSuccessDoesNotWait)
{
  int64_t waited_us = -1;
  const evthr_res res = RetryDefer(
      FakeThread(1), NoopCallback, nullptr, 1000 /* budget_ms */, FakeDefer,
      &waited_us);
  EXPECT_EQ(res, EVTHR_RES_OK);
  EXPECT_EQ(g_call_count, 1);
  EXPECT_EQ(waited_us, 0);
}

TEST_F(RetryDeferTest, RetriesUntilSuccess)
{
  g_succeed_after = 3;
  int64_t waited_us = 0;
  const evthr_res res = RetryDefer(
      FakeThread(1), NoopCallback, nullptr, 1000, FakeDefer, &waited_us);
  EXPECT_EQ(res, EVTHR_RES_OK);
  EXPECT_EQ(g_call_count, 4);
  EXPECT_GT(waited_us, 0);
}

TEST_F(RetryDeferTest, ZeroBudgetMakesExactlyOneAttempt)
{
  const evthr_res res =
      RetryDefer(FakeThread(1), NoopCallback, nullptr, 0, AlwaysRetry);
  EXPECT_EQ(res, EVTHR_RES_RETRY);
  EXPECT_EQ(g_call_count, 1);
}

TEST_F(RetryDeferTest, ExhaustsBudgetAndReportsWaitedTime)
{
  int64_t waited_us = 0;
  const evthr_res res = RetryDefer(
      FakeThread(1), NoopCallback, nullptr, 5 /* ms */, AlwaysRetry,
      &waited_us);
  EXPECT_EQ(res, EVTHR_RES_RETRY);
  EXPECT_GT(g_call_count, 1);
  // The loop only stops once the deadline has passed; scheduling slack only
  // makes the wait longer.
  EXPECT_GE(waited_us, 5000);
}

TEST_F(RetryDeferTest, NonRetryableResultStopsRetrying)
{
  g_fatal_on_second = true;
  g_succeed_after = 100;
  const evthr_res res =
      RetryDefer(FakeThread(1), NoopCallback, nullptr, 1000, FakeDefer);
  EXPECT_EQ(res, EVTHR_RES_FATAL);
  EXPECT_EQ(g_call_count, 2);
}

TEST(WorkerCircuitTest, StartsClosed)
{
  WorkerCircuit circuit;
  EXPECT_FALSE(circuit.Open(FakeThread(1)));
}

TEST(WorkerCircuitTest, OpensAfterThresholdConsecutiveExhaustions)
{
  WorkerCircuit circuit(3);
  EXPECT_EQ(circuit.Exhausted(FakeThread(1)), 1);
  EXPECT_EQ(circuit.Exhausted(FakeThread(1)), 2);
  EXPECT_FALSE(circuit.Open(FakeThread(1)));
  EXPECT_EQ(circuit.Exhausted(FakeThread(1)), 3);
  EXPECT_TRUE(circuit.Open(FakeThread(1)));
  // Other workers are unaffected.
  EXPECT_FALSE(circuit.Open(FakeThread(2)));
}

TEST(WorkerCircuitTest, SuccessResetsTheCount)
{
  WorkerCircuit circuit(3);
  circuit.Exhausted(FakeThread(1));
  circuit.Exhausted(FakeThread(1));
  circuit.Ok(FakeThread(1));
  EXPECT_FALSE(circuit.Open(FakeThread(1)));
  EXPECT_EQ(circuit.Exhausted(FakeThread(1)), 1);
}

TEST(WorkerCircuitTest, ReopensAfterResetOnRepeatedExhaustion)
{
  WorkerCircuit circuit(2);
  circuit.Exhausted(FakeThread(1));
  circuit.Exhausted(FakeThread(1));
  EXPECT_TRUE(circuit.Open(FakeThread(1)));
  circuit.Ok(FakeThread(1));
  EXPECT_FALSE(circuit.Open(FakeThread(1)));
  circuit.Exhausted(FakeThread(1));
  circuit.Exhausted(FakeThread(1));
  EXPECT_TRUE(circuit.Open(FakeThread(1)));
}

TEST(WorkerCircuitTest, OkWithoutHistoryIsANoop)
{
  WorkerCircuit circuit;
  circuit.Ok(FakeThread(7));
  EXPECT_FALSE(circuit.Open(FakeThread(7)));
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
