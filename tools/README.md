# ORT Artifact Build

This directory contains `build_ort_from_triton_config.py`, the local entrypoint
for building ONNX Runtime artifacts with the same versions and CMake settings
that `build.py` would use for the Triton ONNX Runtime backend.

The script resolves Triton's build configuration, writes an
`ort-build-plan.json`, then invokes `triton-onnxruntime_backend/tools/build_ort_artifacts.py`
to build the `ort_target` CMake target and package the resulting
`onnxruntime/` payload.

Options before `--` configure the ORT artifact wrapper itself. Options after
`--` are forwarded to `build.py` so the ORT plan uses the same Triton build
configuration.

## Prerequisites

- Run commands from the `triton-server` checkout.
- Keep `triton-onnxruntime_backend` checked out next to `triton-server`, or pass
  `--backend-source-dir`.
- Keep `triton-common`, `triton-core`, and `triton-backend` checked out next to
  `triton-server`. The wrapper automatically passes those local directories to
  backend CMake `FetchContent`, which avoids fetching
  `https://github.com/triton-inference-server/common.git`.
- Use the same branch in both repositories, for example `standalone_ort_build`.
- Have Docker available. For rootless Docker, export `DOCKER_HOST`, for example:

```bash
export DOCKER_HOST=unix:///run/user/${UID}/docker.sock
```

## Generate a Plan Only

Use this first to inspect the resolved versions, repo tags, container version,
parallelism, CUDA architecture list, and backend CMake arguments without running
the ORT build.

```bash
./tools/build_ort_from_triton_config.py \
  --artifact-dir /tmp/triton-ort \
  --plan-only \
  -- \
  --enable-gpu \
  -j 8
```

The plan is written to `/tmp/triton-ort/ort-build-plan.json`.

## Build ORT Artifacts

Remove `--plan-only` to run the ORT-only build.

```bash
./tools/build_ort_from_triton_config.py \
  --artifact-dir /tmp/triton-ort \
  --backend-source-dir ../triton-onnxruntime_backend \
  -- \
  --enable-gpu \
  -j 8
```

By default, this configures the backend CMake build on the host, so the host
must provide `cmake`. To use the same Triton buildbase container flow as the
top-level container build, add `--build-in-triton-build-container`. This builds
the `tritonserver_buildbase` image from `build.py`'s generated
`Dockerfile.buildbase`, then runs the backend artifact builder inside that
container. The nested ORT Docker build uses Docker's `host` network mode by
default to avoid DNS issues in docker-in-docker builds; override it with
`--ort-docker-build-network` if needed.

```bash
./tools/build_ort_from_triton_config.py \
  --build-in-triton-build-container \
  --artifact-dir /tmp/triton-ort \
  --backend-source-dir ../triton-onnxruntime_backend \
  -- \
  --enable-gpu \
  -j 8
```

Expected outputs:

- `/tmp/triton-ort/ort-build-plan.json`
- `/tmp/triton-ort/ort-artifact-manifest.json`
- `/tmp/triton-ort/onnxruntime-<ort-version>-<container-version>-<config-hash>.tar.gz`

The tarball contains `onnxruntime/triton-ort-artifact.json`, which records the
resolved Triton build plan used to create the artifact, including
the `build.py` CUDA architecture list before ONNX Runtime backend conversion.
The backend owns the conversion from that Triton value to ORT
`CMAKE_CUDA_ARCHITECTURES`.

## Reuse the Artifact in a Triton Build

Use the unpacked `onnxruntime/` payload as `--ort-artifacts-dir` in a later
Triton build. The path must contain `include/`, `lib/libonnxruntime.so`, and
`triton-ort-artifact.json`.

```bash
mkdir -p /tmp/triton-ort/reuse
tar -xzf /tmp/triton-ort/onnxruntime-*.tar.gz -C /tmp/triton-ort/reuse

CUDA_ARCH_LIST="8.6" \
python ./build.py \
  --backend onnxruntime \
  --ort-artifacts-dir /tmp/triton-ort/reuse/onnxruntime \
  --enable-gpu \
  -j 8
```

The full Triton build validates the current `build.py` CUDA architecture list
and target machine against the artifact metadata before reusing the artifact.
Reuse fails when either value differs.

## Common Options

Control nested ORT build parallelism through the normal Triton build option:

```bash
./tools/build_ort_from_triton_config.py --artifact-dir /tmp/triton-ort -- -j 16
```

Build CPU-only ORT by omitting `--enable-gpu`:

```bash
./tools/build_ort_from_triton_config.py --artifact-dir /tmp/triton-ort -- -j 8
```

Use component checkouts from a non-standard location:

```bash
./tools/build_ort_from_triton_config.py \
  --artifact-dir /tmp/triton-ort \
  --component-source-dir common=/path/to/triton-common \
  --component-source-dir core=/path/to/triton-core \
  --component-source-dir backend=/path/to/triton-backend \
  -- \
  --enable-gpu \
  -j 8
```

Override an ONNX Runtime backend CMake argument through the forwarded
`build.py` options. For example, disable OpenVINO support:

```bash
./tools/build_ort_from_triton_config.py \
  --artifact-dir /tmp/triton-ort \
  -- \
  --enable-gpu \
  --override-backend-cmake-arg onnxruntime:TRITON_ENABLE_ONNXRUNTIME_OPENVINO=OFF \
  -j 8
```

Set CUDA architectures explicitly through the forwarded `build.py` option:

```bash
./tools/build_ort_from_triton_config.py \
  --artifact-dir /tmp/triton-ort \
  -- \
  --enable-gpu \
  --cuda-arch-list "8.0 8.6 9.0" \
  -j 8
```

If `--cuda-arch-list` is omitted, `build.py` falls back to `CUDA_ARCH_LIST`
from the environment when it is set.

The preferred path runs the ORT clone/build step as a non-root user inside the
generated Dockerfile. If an upstream ORT build assumption requires root, use the
explicit compatibility fallback:

```bash
./tools/build_ort_from_triton_config.py \
  --allow-root-ort-build \
  --artifact-dir /tmp/triton-ort \
  -- \
  --enable-gpu \
  -j 8
```
