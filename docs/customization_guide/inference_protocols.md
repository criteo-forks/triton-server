<!--
# Copyright 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Inference Protocols and APIs

Clients can communicate with Triton using either an [HTTP/REST
protocol](#httprest-and-grpc-protocols), a [GRPC
protocol](#httprest-and-grpc-protocols), or by an [in-process C
API](inprocess_c_api.md) or its
[C++ wrapper](https://github.com/triton-inference-server/developer_tools/tree/main/server).

## HTTP/REST and GRPC Protocols

Triton exposes both HTTP/REST and GRPC endpoints based on [standard
inference
protocols](https://github.com/kserve/kserve/tree/master/docs/predict-api/v2)
that have been proposed by the [KServe
project](https://github.com/kserve). To fully enable all capabilities
Triton also implements [HTTP/REST and GRPC
extensions](https://github.com/triton-inference-server/server/tree/main/docs/protocol)
to the KServe inference protocol. GRPC protocol also provides a
bi-directional streaming version of the inference RPC to allow a
sequence of inference requests/responses to be sent over a
GRPC stream. We typically recommend using the unary version for
inference requests. The streaming version should be used only if the
situation demands it. Some of such use cases can be:

* Assume a system with multiple Triton server instances running
  behind a Load Balancer. If a sequence of inference requests is
  needed to hit the same Triton server instance, a GRPC stream
  will hold a single connection throughout the lifetime and hence
  ensure the requests are delivered to the same Triton instance.
* If the order of requests/responses needs to be preserved over
  the network, a GRPC stream will ensure that the server receives
  the requests in the same order as they were sent from the
  client.

The HTTP/REST and GRPC protocols also provide endpoints to check
server and model health, metadata and statistics. Additional
endpoints allow model loading and unloading, and inferencing. See
the KServe and extension documentation for details.

### HTTP Options
Triton provides the following configuration options for server-client network transactions over HTTP protocol.

#### HTTP Reply Hand-off Tuning

Triton pauses each HTTP request on the completion thread and resumes it via a
callback handed to its connection's worker thread over the worker command pipe.
Under a reply burst the pipe can fill and the hand-off would be lost, leaking
the request's model references and wedging the model's next unload. To prevent
that, replies are batched per worker and a dropped hand-off is retried and
accounted for. The following environment variables tune that machinery:

| Variable | Default | Meaning |
| --- | --- | --- |
| `TRITON_HTTP_DEFER_RETRY_BUDGET_MS` | `3000` | Per-hand-off retry budget in ms before a reply is declared a leak. `0` disables retries (A/B control). |
| `TRITON_HTTP_REPLY_BATCH_MAX` | `256` | Max replies a single worker drain delivers before yielding back to the event loop. |
| `TRITON_HTTP_REPLY_QUEUE_MAX` | `12500` | Max queued replies per worker beyond which further hand-offs are dropped and reported, mirroring the old command-pipe bound. |
| `TRITON_HTTP_REPLY_DRAIN_CMDS_MAX` | `1` | Max drain commands a worker may have outstanding in its command pipe. evhtp places new connections on the worker with the fewest pipe commands, so this bounds the load a busy worker advertises: `1` only distinguishes busy from idle, larger values (e.g. `10`) restore a backlog proportional to the reply rate while still keeping the pipe far from full. |

Rejected or dropped hand-offs are reported on the metrics endpoint
(`nv_http_reply_handoff_retried`, `nv_http_reply_handoff_dropped`,
`nv_http_reply_batches`, `nv_http_reply_batched`, `nv_http_reply_queue_depth`)
and in the `FULL-SOCKETS` log lines, so fleets can alert on pods that will
leak and restart them before an unload wedges.

Note that `nv_inference_request_duration_us` stops when the backend's response
send returns, which for HTTP is the moment the reply is queued for its worker
thread. The time a reply then waits for that worker is reported separately by
the `nv_http_reply_handoff_wait_us` histogram; a growing tail there with normal
request durations means the HTTP frontend, not inference, is adding latency.

#### Compression

Triton allows the on-wire compression of request/response on HTTP through its clients. See [HTTP Compression](../client/README.md#compression) for more details.

#### Mapping Triton Server Error Codes to HTTP Status Codes

This table maps various Triton Server error codes to their corresponding HTTP status
codes. It can be used as a reference guide for understanding how Triton Server errors
are handled in HTTP responses.


| Triton Server Error Code                      | HTTP Status Code   | Description          |
| ----------------------------------------------| -------------------| ---------------------|
| `TRITONSERVER_ERROR_INTERNAL`                 | 500                | Internal Server Error|
| `TRITONSERVER_ERROR_NOT_FOUND`                | 404                | Not Found            |
| `TRITONSERVER_ERROR_UNAVAILABLE`              | 503                | Service Unavailable  |
| `TRITONSERVER_ERROR_UNSUPPORTED`              | 501                | Not Implemented      |
| `TRITONSERVER_ERROR_UNKNOWN`,<br>`TRITONSERVER_ERROR_INVALID_ARG`,<br>`TRITONSERVER_ERROR_ALREADY_EXISTS`,<br>`TRITONSERVER_ERROR_CANCELLED` | `400` | Bad Request (default for other errors)      |

### GRPC Options
Triton exposes various GRPC parameters for configuring the server-client network transactions. For usage of these options, refer to the output from `tritonserver --help`.

#### SSL/TLS

These options can be used to configure a secured channel for communication. The server-side options include:

* `--grpc-use-ssl`
* `--grpc-use-ssl-mutual`
* `--grpc-server-cert`
* `--grpc-server-key`
* `--grpc-root-cert`

For client-side documentation, see [Client-Side GRPC SSL/TLS](https://github.com/triton-inference-server/client/tree/main#ssltls)

For more details on overview of authentication in gRPC, refer [here](https://grpc.io/docs/guides/auth/).

#### Compression

Triton allows the on-wire compression of request/response messages by exposing following option on server-side:

* `--grpc-infer-response-compression-level`

For client-side documentation, see [Client-Side GRPC Compression](https://github.com/triton-inference-server/client/tree/main#compression-1)

Compression can be used to reduce the amount of bandwidth used in server-client communication. For more details, see [gRPC Compression](https://grpc.github.io/grpc/core/md_doc_compression.html).

#### GRPC KeepAlive

Triton exposes GRPC KeepAlive parameters with the default values for both
client and server described [here](https://github.com/grpc/grpc/blob/master/doc/keepalive.md).

These options can be used to configure the KeepAlive settings:

* `--grpc-keepalive-time`
* `--grpc-keepalive-timeout`
* `--grpc-keepalive-permit-without-calls`
* `--grpc-http2-max-pings-without-data`
* `--grpc-http2-min-recv-ping-interval-without-data`
* `--grpc-http2-max-ping-strikes`

For client-side documentation, see [Client-Side GRPC KeepAlive](https://github.com/triton-inference-server/client/blob/main/README.md#grpc-keepalive).

#### GRPC Status Codes

Triton implements GRPC error handling for streaming requests when a specific flag is enabled through headers. Upon encountering an error, Triton returns the appropriate GRPC error code and subsequently closes the stream.

* `triton_grpc_error` : The header value needs to be set to true while starting the stream.

GRPC status codes can be used for better visibility and monitoring. For more details, see [gRPC Status Codes](https://grpc.io/docs/guides/status-codes/)

For client-side documentation, see [Client-Side GRPC Status Codes](https://github.com/triton-inference-server/client/tree/main#GRPC-Status-Codes)

#### GRPC Inference Handler Threads

In general, using 2 threads per completion queue seems to give the best performance, see [gRPC Performance Best Practices](https://grpc.io/docs/guides/performance/#c). However, in cases where the performance bottleneck is at the request handling step (e.g. ensemble models), increasing the number of gRPC inference handler threads may lead to a higher throughput.

* `--grpc-infer-thread-count`: 2 by default.

Note: More threads don't always mean better performance.

### Limit Endpoint Access (BETA)

Triton users may want to restrict access to protocols or APIs that are
provided by the GRPC or HTTP endpoints of a server. For example, users
can provide one set of access credentials for inference APIs and
another for model control APIs such as model loading and unloading.

The following options can be specified to declare a restricted
protocol group (GRPC) or restricted API group (HTTP):

```
--grpc-restricted-protocol=<protocol_1>,<protocol_2>,...:<restricted-key>=<restricted-value>
--http-restricted-api=<API_1>,API_2>,...:<restricted-key>=<restricted-value>
```

When Vertex AI endpoint support is enabled, `--http-restricted-api`
also applies to redirected Vertex AI requests.
If Triton is built without `TRITON_ENABLE_HTTP`, Vertex AI falls back
to default unrestricted API settings.

The option can be specified multiple times to specifies multiple groups of
protocols or APIs with different restriction settings.

* `protocols / APIs` : A comma-separated list of protocols / APIs to be included in this
group. Note that currently a given protocol / API is not allowed to be included in
multiple groups. The following protocols / APIs are recognized:

  * `health` : Health endpoint defined for [HTTP/REST](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#health) and [GRPC](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#health-1). For GRPC endpoint, this value also exposes [GRPC health check protocol](https://github.com/triton-inference-server/common/blob/main/protobuf/health.proto).
  * `metadata` : Server / model metadata endpoints defined for [HTTP/REST](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#server-metadata) and [GRPC](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#server-metadata-1).
  * `inference` : Inference endpoints defined for [HTTP/REST](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference) and [GRPC](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md#inference-1).
  * `shared-memory` : [Shared-memory endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_shared_memory.md).
  * `model-config` : [Model configuration endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_configuration.md).
  * `model-repository` : [Model repository endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md).
  * `statistics` : [statistics endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_statistics.md).
  * `trace` : [trace endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_trace.md).
  * `logging` : [logging endpoint](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_logging.md).

* `restricted-key` : The GRPC / HTTP request header
to be checked when a request is received. The
completed header for GRPC will be in the form of
`triton-grpc-protocol-<restricted-key>`. The completed header for HTTP
will be in the form of `<restricted-key>`.

* `restricted-value` : The header value required to access the specified protocols.

#### Example

To start the server with a set of protocols and APIs restricted for
`admin` usage and the rest of the protocols and APIs left unrestricted
use the following command line arguments:


```
tritonserver --grpc-restricted-protocol=shared-memory,model-config,model-repository,statistics,trace:<admin-key>=<admin-value> \
             --http-restricted-api=shared-memory,model-config,model-repository,statistics,trace:<admin-key>=<admin-value> ...
```

GRPC requests to `admin` protocols require that an additional header
`triton-grpc-protocol-<admin-key>` is provided with value
`<admin-value>`. HTTP requests to `admin` APIs required that an
additional header `<admin-key>` is provided with value `<admin-value>`.
