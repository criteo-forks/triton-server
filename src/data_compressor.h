// Copyright (c) 2021-2025, NVIDIA CORPORATION. All rights reserved.
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

#include <event2/buffer.h>
#include <zlib.h>
#include <zstd.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "triton/core/tritonserver.h"

namespace triton { namespace server {

//
// DataCompressor
//
class DataCompressor {
 public:
  enum class Type { UNKNOWN, IDENTITY, GZIP, DEFLATE, ZSTD };

  // Specialization where the source and destination buffer are stored as
  // evbuffer
  static TRITONSERVER_Error* CompressData(
      const Type type, evbuffer* source, evbuffer* compressed_data)
  {
    size_t expected_compressed_size = evbuffer_get_length(source);
    // nothing to be compressed
    if (expected_compressed_size == 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG, "nothing to be compressed");
    }

    if (type == Type::ZSTD) {
      return CompressZstd(source, compressed_data, expected_compressed_size);
    }

    z_stream stream;
    stream.zalloc = Z_NULL;
    stream.zfree = Z_NULL;
    stream.opaque = Z_NULL;
    switch (type) {
      case Type::UNKNOWN:
      case Type::IDENTITY:
      case Type::ZSTD: {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG, "nothing to be compressed");
      }
      case Type::GZIP:
        if (deflateInit2(
                &stream, Z_DEFAULT_COMPRESSION /* level */,
                Z_DEFLATED /* method */, 15 | 16 /* windowBits */,
                8 /* memLevel */, Z_DEFAULT_STRATEGY /* strategy */) != Z_OK) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "failed to initialize state for gzip data compression");
        }
        break;
      case Type::DEFLATE: {
        if (deflateInit(&stream, Z_DEFAULT_COMPRESSION /* level */) != Z_OK) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "failed to initialize state for deflate data compression");
        }
        break;
      }
    }
    // ensure the internal state are cleaned up on function return
    std::unique_ptr<z_stream, decltype(&deflateEnd)> managed_stream(
        &stream, deflateEnd);

    // Get the addr and size of each chunk of memory in 'source'
    std::unique_ptr<struct evbuffer_iovec[]> buffer_array_holder;
    struct evbuffer_iovec* buffer_array = nullptr;
    int buffer_count = evbuffer_peek(source, -1, NULL, NULL, 0);
    if (buffer_count > 0) {
      buffer_array_holder.reset(new struct evbuffer_iovec[buffer_count]);
      buffer_array = buffer_array_holder.get();
      if (evbuffer_peek(source, -1, NULL, buffer_array, buffer_count) !=
          buffer_count) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            "unexpected error getting buffers to be compressed");
      }
    }
    // Reserve the same size as source for compressed data, it is less likely
    // that a negative compression happens.
    struct evbuffer_iovec current_reserved_space;
    RETURN_MSG_IF_ERR(
        AllocEVBuffer(
            expected_compressed_size, compressed_data, &current_reserved_space),
        "unexpected error allocating output buffer for compression");
    stream.next_out =
        reinterpret_cast<unsigned char*>(current_reserved_space.iov_base);
    stream.avail_out = expected_compressed_size;

    // Compress until end of 'source'
    for (int idx = 0; idx < buffer_count; ++idx) {
      stream.next_in =
          reinterpret_cast<unsigned char*>(buffer_array[idx].iov_base);
      stream.avail_in = buffer_array[idx].iov_len;

      // run deflate() on input until source has been read in
      do {
        // Need additional buffer
        if (stream.avail_out == 0) {
          RETURN_MSG_IF_ERR(
              CommitEVBuffer(
                  compressed_data, &current_reserved_space,
                  expected_compressed_size),
              "unexpected error committing output buffer for compression");
          RETURN_MSG_IF_ERR(
              AllocEVBuffer(
                  expected_compressed_size, compressed_data,
                  &current_reserved_space),
              "unexpected error allocating output buffer for compression");
          stream.next_out =
              reinterpret_cast<unsigned char*>(current_reserved_space.iov_base);
          stream.avail_out = expected_compressed_size;
        }
        auto flush = ((idx + 1) != buffer_count) ? Z_NO_FLUSH : Z_FINISH;
        auto ret = deflate(&stream, flush);
        if (ret == Z_STREAM_ERROR) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "encountered inconsistent stream state during compression");
        }
      } while (stream.avail_out == 0);
    }
    // Make sure the last buffer is committed
    if (current_reserved_space.iov_base != nullptr) {
      RETURN_MSG_IF_ERR(
          CommitEVBuffer(
              compressed_data, &current_reserved_space,
              expected_compressed_size - stream.avail_out),
          "unexpected error committing output buffer for compression");
    }
    return nullptr;  // success
  }

  static TRITONSERVER_Error* DecompressData(
      const Type type, evbuffer* source, evbuffer* decompressed_data,
      const size_t max_decompressed_size = 0)
  {
    size_t source_byte_size = evbuffer_get_length(source);
    // nothing to be decompressed
    if (evbuffer_get_length(source) == 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG, "nothing to be decompressed");
    }
    // Set reasonable size for each output buffer to be allocated
    size_t output_buffer_size = (source_byte_size > (1 << 20 /* 1MB */))
                                    ? source_byte_size
                                    : (1 << 20 /* 1MB */);

    // Cap the initial buffer allocation to the decompression limit.
    // This avoids over-allocating when a decompression limit is set.
    if (max_decompressed_size > 0 &&
        output_buffer_size > max_decompressed_size) {
      output_buffer_size = max_decompressed_size;
    }

    switch (type) {
      case Type::UNKNOWN:
      case Type::IDENTITY: {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG, "nothing to be decompressed");
      }
      case Type::ZSTD:
        return DecompressZstd(
            source, decompressed_data, output_buffer_size,
            max_decompressed_size);
      case Type::GZIP:
      case Type::DEFLATE:
        // zlib can automatically detect compression type
        {
          z_stream stream;
          stream.zalloc = Z_NULL;
          stream.zfree = Z_NULL;
          stream.opaque = Z_NULL;
          stream.avail_in = 0;
          stream.next_in = Z_NULL;

          if (inflateInit2(&stream, 15 | 32) != Z_OK) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                "failed to initialize state for data decompression");
          }
          // ensure the internal state are cleaned up on function return
          std::unique_ptr<z_stream, decltype(&inflateEnd)> managed_stream(
              &stream, inflateEnd);

          // Get the addr and size of each chunk of memory in 'source'
          std::unique_ptr<struct evbuffer_iovec[]> buffer_array_holder;
          struct evbuffer_iovec* buffer_array = nullptr;
          int buffer_count = evbuffer_peek(source, -1, NULL, NULL, 0);
          if (buffer_count > 0) {
            buffer_array_holder.reset(new struct evbuffer_iovec[buffer_count]);
            buffer_array = buffer_array_holder.get();
            if (evbuffer_peek(source, -1, NULL, buffer_array, buffer_count) !=
                buffer_count) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INTERNAL,
                  "unexpected error getting buffers to be decompressed");
            }
          }
          // Reserve the same size as source for compressed data, it is less
          // likely that a negative compression happens.
          struct evbuffer_iovec current_reserved_space;
          RETURN_MSG_IF_ERR(
              AllocEVBuffer(
                  output_buffer_size, decompressed_data,
                  &current_reserved_space),
              "unexpected error allocating output buffer for decompression");
          stream.next_out =
              reinterpret_cast<unsigned char*>(current_reserved_space.iov_base);
          stream.avail_out = output_buffer_size;

          // Track total decompressed size and current buffer size for limit
          // checking
          size_t total_decompressed = 0;
          size_t current_buffer_size = output_buffer_size;

          // Decompress until end of 'source'
          for (int idx = 0; idx < buffer_count; ++idx) {
            stream.next_in =
                reinterpret_cast<unsigned char*>(buffer_array[idx].iov_base);
            stream.avail_in = buffer_array[idx].iov_len;

            // run inflate() on input until source has been read in
            do {
              // Need additional buffer
              if (stream.avail_out == 0) {
                total_decompressed += current_buffer_size;

                // Check decompression size limit before allocating memory
                if (max_decompressed_size > 0 &&
                    total_decompressed > max_decompressed_size) {
                  return TRITONSERVER_ErrorNew(
                      TRITONSERVER_ERROR_INVALID_ARG,
                      ("Decompressed data size exceeds the maximum allowed "
                       "value of " +
                       std::to_string(max_decompressed_size) +
                       " bytes. Use --http-max-input-size to increase the "
                       "limit.")
                          .c_str());
                }

                RETURN_MSG_IF_ERR(
                    CommitEVBuffer(
                        decompressed_data, &current_reserved_space,
                        current_buffer_size),
                    "unexpected error committing output buffer for "
                    "decompression");

                // Calculate next buffer size, capped by remaining limit
                current_buffer_size = output_buffer_size;
                if (max_decompressed_size > 0) {
                  size_t remaining_size =
                      max_decompressed_size - total_decompressed;
                  // If no space remains but decompression needs more, we've hit
                  // the limit
                  if (remaining_size == 0) {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INVALID_ARG,
                        ("Decompressed data size exceeds the maximum allowed "
                         "value of " +
                         std::to_string(max_decompressed_size) +
                         " bytes. Use --http-max-input-size to increase the "
                         "limit.")
                            .c_str());
                  }
                  if (current_buffer_size > remaining_size) {
                    current_buffer_size = remaining_size;
                  }
                }

                RETURN_MSG_IF_ERR(
                    AllocEVBuffer(
                        current_buffer_size, decompressed_data,
                        &current_reserved_space),
                    "unexpected error allocating output buffer for "
                    "decompression");
                stream.next_out = reinterpret_cast<unsigned char*>(
                    current_reserved_space.iov_base);
                stream.avail_out = current_buffer_size;
              }
              auto ret = inflate(&stream, Z_NO_FLUSH);
              if (ret == Z_STREAM_ERROR) {
                return TRITONSERVER_ErrorNew(
                    TRITONSERVER_ERROR_INTERNAL,
                    "encountered inconsistent stream state during "
                    "decompression");
              }
              // Break if decompression is complete, even if buffer is exactly
              // full
              if (ret == Z_STREAM_END) {
                break;
              }
            } while (stream.avail_out == 0);
          }
          // Make sure the last buffer is committed
          if (current_reserved_space.iov_base != nullptr) {
            size_t final_chunk_size = current_buffer_size - stream.avail_out;
            if (max_decompressed_size > 0 &&
                (total_decompressed + final_chunk_size) >
                    max_decompressed_size) {
              return TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_INVALID_ARG,
                  ("Decompressed data size exceeds the maximum allowed value "
                   "of " +
                   std::to_string(max_decompressed_size) +
                   " bytes. Use --http-max-input-size to increase the limit.")
                      .c_str());
            }

            RETURN_MSG_IF_ERR(
                CommitEVBuffer(
                    decompressed_data, &current_reserved_space,
                    final_chunk_size),
                "unexpected error committing output buffer for "
                "decompression");
          }
          break;
        }
    }
    return nullptr;  // success
  }

 private:
  static TRITONSERVER_Error* GetSourceBuffers(
      evbuffer* source, std::unique_ptr<struct evbuffer_iovec[]>* holder,
      int* buffer_count)
  {
    *buffer_count = evbuffer_peek(source, -1, NULL, NULL, 0);
    if (*buffer_count > 0) {
      holder->reset(new struct evbuffer_iovec[*buffer_count]);
      if (evbuffer_peek(source, -1, NULL, holder->get(), *buffer_count) !=
          *buffer_count) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            "unexpected error getting buffers from evbuffer");
      }
    }
    return nullptr;  // success
  }

  static TRITONSERVER_Error* ZstdError(const char* what, size_t code)
  {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string(what) + ": " + ZSTD_getErrorName(code)).c_str());
  }

  static TRITONSERVER_Error* DecompressedSizeExceeded(
      const size_t max_decompressed_size)
  {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        ("Decompressed data size exceeds the maximum allowed value of " +
         std::to_string(max_decompressed_size) +
         " bytes. Use --http-max-input-size to increase the limit.")
            .c_str());
  }

  static TRITONSERVER_Error* CompressZstd(
      evbuffer* source, evbuffer* compressed_data, const size_t source_size)
  {
    thread_local std::unique_ptr<ZSTD_CCtx, decltype(&ZSTD_freeCCtx)> cctx(
        ZSTD_createCCtx(), ZSTD_freeCCtx);
    if (cctx == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "failed to initialize state for zstd data compression");
    }
    ZSTD_CCtx_reset(cctx.get(), ZSTD_reset_session_only);
    size_t ret = ZSTD_CCtx_setParameter(
        cctx.get(), ZSTD_c_compressionLevel, ZSTD_CLEVEL_DEFAULT);
    if (ZSTD_isError(ret)) {
      return ZstdError("failed to set zstd compression level", ret);
    }
    ret = ZSTD_CCtx_setPledgedSrcSize(cctx.get(), source_size);
    if (ZSTD_isError(ret)) {
      return ZstdError("failed to set zstd source size", ret);
    }

    std::unique_ptr<struct evbuffer_iovec[]> buffer_array_holder;
    int buffer_count = 0;
    RETURN_IF_ERR(
        GetSourceBuffers(source, &buffer_array_holder, &buffer_count));
    struct evbuffer_iovec* buffer_array = buffer_array_holder.get();

    const size_t output_size = ZSTD_compressBound(source_size);
    struct evbuffer_iovec current_reserved_space;
    RETURN_MSG_IF_ERR(
        AllocEVBuffer(output_size, compressed_data, &current_reserved_space),
        "unexpected error allocating output buffer for compression");
    ZSTD_outBuffer out{current_reserved_space.iov_base, output_size, 0};

    for (int idx = 0; idx < buffer_count; ++idx) {
      ZSTD_inBuffer in{
          buffer_array[idx].iov_base, buffer_array[idx].iov_len, 0};
      const bool last = (idx + 1) == buffer_count;
      const ZSTD_EndDirective mode = last ? ZSTD_e_end : ZSTD_e_continue;
      do {
        ret = ZSTD_compressStream2(cctx.get(), &out, &in, mode);
        if (ZSTD_isError(ret)) {
          ZSTD_CCtx_reset(cctx.get(), ZSTD_reset_session_only);
          return ZstdError("zstd compression failed", ret);
        }
        if ((out.pos == out.size) && ((in.pos < in.size) || (last && ret))) {
          ZSTD_CCtx_reset(cctx.get(), ZSTD_reset_session_only);
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "zstd compressed data exceeds the compression bound");
        }
      } while ((in.pos < in.size) || (last && (ret != 0)));
    }

    RETURN_MSG_IF_ERR(
        CommitEVBuffer(compressed_data, &current_reserved_space, out.pos),
        "unexpected error committing output buffer for compression");
    return nullptr;  // success
  }

  static TRITONSERVER_Error* DecompressZstd(
      evbuffer* source, evbuffer* decompressed_data,
      const size_t output_buffer_size, const size_t max_decompressed_size)
  {
    thread_local std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> dctx(
        ZSTD_createDCtx(), ZSTD_freeDCtx);
    if (dctx == nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "failed to initialize state for zstd data decompression");
    }
    ZSTD_DCtx_reset(dctx.get(), ZSTD_reset_session_only);

    std::unique_ptr<struct evbuffer_iovec[]> buffer_array_holder;
    int buffer_count = 0;
    RETURN_IF_ERR(
        GetSourceBuffers(source, &buffer_array_holder, &buffer_count));
    struct evbuffer_iovec* buffer_array = buffer_array_holder.get();

    struct evbuffer_iovec current_reserved_space;
    RETURN_MSG_IF_ERR(
        AllocEVBuffer(
            output_buffer_size, decompressed_data, &current_reserved_space),
        "unexpected error allocating output buffer for decompression");
    ZSTD_outBuffer out{current_reserved_space.iov_base, output_buffer_size, 0};
    size_t total_committed = 0;
    size_t ret = 1;

    for (int idx = 0; idx < buffer_count; ++idx) {
      ZSTD_inBuffer in{
          buffer_array[idx].iov_base, buffer_array[idx].iov_len, 0};
      while (true) {
        if (out.pos == out.size) {
          total_committed += out.size;
          if ((max_decompressed_size > 0) &&
              (total_committed >= max_decompressed_size)) {
            return DecompressedSizeExceeded(max_decompressed_size);
          }
          RETURN_MSG_IF_ERR(
              CommitEVBuffer(
                  decompressed_data, &current_reserved_space, out.size),
              "unexpected error committing output buffer for decompression");
          size_t next_size = output_buffer_size;
          if ((max_decompressed_size > 0) &&
              (next_size > (max_decompressed_size - total_committed))) {
            next_size = max_decompressed_size - total_committed;
          }
          RETURN_MSG_IF_ERR(
              AllocEVBuffer(
                  next_size, decompressed_data, &current_reserved_space),
              "unexpected error allocating output buffer for decompression");
          out = ZSTD_outBuffer{current_reserved_space.iov_base, next_size, 0};
        }
        ret = ZSTD_decompressStream(dctx.get(), &out, &in);
        if (ZSTD_isError(ret)) {
          ZSTD_DCtx_reset(dctx.get(), ZSTD_reset_session_only);
          return ZstdError("zstd decompression failed", ret);
        }
        if ((in.pos == in.size) && ((ret == 0) || (out.pos < out.size))) {
          break;
        }
      }
    }
    if (ret != 0) {
      ZSTD_DCtx_reset(dctx.get(), ZSTD_reset_session_only);
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG, "truncated zstd compressed data");
    }

    RETURN_MSG_IF_ERR(
        CommitEVBuffer(decompressed_data, &current_reserved_space, out.pos),
        "unexpected error committing output buffer for decompression");
    return nullptr;  // success
  }

  static TRITONSERVER_Error* AllocEVBuffer(
      const size_t byte_size, evbuffer* evb,
      struct evbuffer_iovec* current_reserved_space)
  {
    // Reserve requested space in evbuffer...
    if ((evbuffer_reserve_space(evb, byte_size, current_reserved_space, 1) !=
         1) ||
        (current_reserved_space->iov_len < byte_size)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          std::string(
              "failed to reserve " + std::to_string(byte_size) +
              " bytes in evbuffer")
              .c_str());
    }
    return nullptr;  // success
  }

  static TRITONSERVER_Error* CommitEVBuffer(
      evbuffer* evb, struct evbuffer_iovec* current_reserved_space,
      const size_t filled_byte_size)
  {
    current_reserved_space->iov_len = filled_byte_size;
    if (evbuffer_commit_space(evb, current_reserved_space, 1) != 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL, "failed to commit allocated evbuffer");
    }
    current_reserved_space->iov_base = nullptr;
    return nullptr;  // success
  }
};

}}  // namespace triton::server
