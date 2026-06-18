#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys


SERVER_DIR = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE_DIR = SERVER_DIR.parent


def load_build_module():
    build_py = SERVER_DIR / "build.py"
    spec = importlib.util.spec_from_file_location("triton_build", build_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_value(repo_dir, args):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), *args], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return None


def find_backend_source(backend_source_dir, artifact_dir, github_organization, tag):
    if backend_source_dir is not None:
        return pathlib.Path(backend_source_dir).resolve()

    sibling = WORKSPACE_DIR / "triton-onnxruntime_backend"
    if sibling.exists():
        return sibling.resolve()

    clone_dir = artifact_dir / "src" / "onnxruntime_backend"
    if clone_dir.exists():
        return clone_dir.resolve()

    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    repo_url = "{}/onnxruntime_backend.git".format(github_organization.rstrip("/"))
    subprocess.run(
        [
            "git",
            "clone",
            "--recursive",
            "--single-branch",
            "--depth=1",
            "-b",
            tag,
            repo_url,
            str(clone_dir),
        ],
        check=True,
    )
    return clone_dir.resolve()


def ensure_onnxruntime_backend(build_args):
    found = False
    for idx, arg in enumerate(build_args):
        if arg == "--backend" and idx + 1 < len(build_args):
            backend = build_args[idx + 1].split(":", 1)[0]
            if backend != "onnxruntime":
                raise SystemExit(
                    "ORT artifact builds only support --backend onnxruntime; "
                    f"got --backend {build_args[idx + 1]}"
                )
            found = True
        if arg.startswith("--backend="):
            backend = arg.split("=", 1)[1].split(":", 1)[0]
            if backend != "onnxruntime":
                raise SystemExit(
                    "ORT artifact builds only support --backend onnxruntime; "
                    f"got {arg}"
                )
            found = True
    if found:
        return build_args
    return build_args + ["--backend", "onnxruntime"]

def normalize_cmake_args(args):
    normalized = []
    for arg in args:
        if arg == "..":
            continue
        if len(arg) >= 2 and arg[0] == '"' and arg[-1] == '"':
            arg = arg[1:-1]
        normalized.append(arg)
    return normalized


def config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def docker_mount_args(paths):
    mounts = []
    seen = []
    for path in paths:
        resolved = pathlib.Path(path).resolve()
        if resolved.is_file():
            resolved = resolved.parent
        if any(
            resolved == existing or resolved.is_relative_to(existing)
            for existing in seen
        ):
            continue
        seen.append(resolved)
        mounts += ["-v", f"{resolved}:{resolved}"]
    return mounts


def docker_env_args(names):
    env_args = []
    for name in names:
        if name in os.environ:
            env_args += ["-e", f"{name}={os.environ[name]}"]
    return env_args


def build_triton_buildbase(build, build_config, image):
    if build.FLAGS.no_container_build:
        raise SystemExit(
            "--build-in-triton-build-container cannot be used with --no-container-build"
        )
    if build.target_platform() == "windows":
        raise SystemExit("--build-in-triton-build-container is only supported on Linux")

    pathlib.Path(build.FLAGS.build_dir).mkdir(parents=True, exist_ok=True)
    build.create_build_dockerfiles(
        build_config["script_build_dir"],
        build_config["images"],
        build_config["backends"],
        build_config["repoagents"],
        build_config["caches"],
        build.FLAGS.endpoint,
    )

    dockerfile = pathlib.Path(build.FLAGS.build_dir) / "Dockerfile.buildbase"
    cmd = ["docker", "build", "-t", image, "-f", dockerfile]
    if not build.FLAGS.no_container_pull:
        cmd.append("--pull")
    cmd += [
        "--cache-from=tritonserver_buildbase",
        "--cache-from=tritonserver_buildbase_cache0",
        "--cache-from=tritonserver_buildbase_cache1",
        ".",
    ]
    subprocess.run([str(arg) for arg in cmd], cwd=SERVER_DIR, check=True)


def run_backend_builder_in_triton_build_container(
    build,
    backend_script,
    plan_output,
    artifact_dir,
    backend_source_dir,
    image,
):
    runargs = [
        "docker",
        "run",
        "--rm",
        # Use the host network so CMake FetchContent (e.g. googletest pulled by
        # triton-common tests) can resolve and reach github.com from inside the
        # buildbase container; the rootless default bridge has no working DNS.
        "--network",
        "host",
        "-w",
        str(backend_source_dir),
    ]
    runargs += build.docker_socket_runargs()
    runargs += docker_env_args(
        [
            "CUDA_ARCH_LIST",
            "DOCKER_BUILDKIT",
            "TRT_VERSION",
            "CMAKE_TOOLCHAIN_FILE",
            "VCPKG_TARGET_TRIPLET",
            "CCACHE_REMOTE_ONLY",
            "CCACHE_REMOTE_STORAGE",
        ]
    )
    if build.FLAGS.use_user_docker_config and os.path.exists(
        build.FLAGS.use_user_docker_config
    ):
        runargs += [
            "-v",
            "{}:/root/.docker/config.json".format(
                os.path.expanduser(build.FLAGS.use_user_docker_config)
            ),
        ]
    # WORKSPACE_DIR contains the local org mirror (passed as --github-organization),
    # so mounting it makes the mirror available for the backend's nested component
    # clones; no per-component mounts are needed.
    mount_paths = [WORKSPACE_DIR, artifact_dir, plan_output.parent, backend_source_dir]
    runargs += docker_mount_args(mount_paths)
    # The mirror is owned by the host user but git runs as root in the container;
    # mark it safe so FetchContent clones don't fail with "dubious ownership".
    inner_cmd = "git config --system --add safe.directory '*' && python3 '{}' --plan '{}'".format(
        backend_script, plan_output
    )
    runargs += [image, "sh", "-c", inner_cmd]
    subprocess.run([str(arg) for arg in runargs], check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build ONNX Runtime artifacts using Triton's resolved build configuration."
    )
    parser.add_argument(
        "--artifact-dir",
        type=pathlib.Path,
        default=SERVER_DIR / "build" / "ort-artifacts",
        help="Directory where the ORT plan, manifest and tarball will be written.",
    )
    parser.add_argument(
        "--plan-output",
        type=pathlib.Path,
        default=None,
        help="Path for the generated ORT build plan. Defaults to <artifact-dir>/ort-build-plan.json.",
    )
    parser.add_argument(
        "--backend-source-dir",
        type=pathlib.Path,
        default=None,
        help="Path to a triton-onnxruntime_backend checkout. Defaults to a sibling checkout.",
    )
    parser.add_argument(
        "--ort-docker-build-network",
        default="host",
        help=(
            "Docker network mode for the nested ONNXRuntime docker build. "
            "Defaults to host to avoid DNS issues in docker-in-docker builds. "
            "Use an empty value to keep Docker's default build network."
        ),
    )
    parser.add_argument(
        "--allow-root-ort-build",
        action="store_true",
        help="Allow the ORT Dockerfile to run the ORT build step as root.",
    )
    parser.add_argument(
        "--build-in-triton-build-container",
        action="store_true",
        help=(
            "Build the ORT artifact from inside the Triton buildbase container "
            "generated by build.py, instead of requiring CMake on the host."
        ),
    )
    parser.add_argument(
        "--triton-buildbase-image",
        default="tritonserver_buildbase",
        help="Docker image tag to use for the Triton buildbase container.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write the ORT build plan but do not invoke the backend artifact builder.",
    )
    wrapper_args, build_args = parser.parse_known_args()

    if build_args and build_args[0] == "--":
        build_args = build_args[1:]
    build_args = ensure_onnxruntime_backend(build_args)

    build = load_build_module()
    build_config = build.resolve_build_config(build.create_arg_parser().parse_args(build_args))

    library_paths = build_config["library_paths"]
    if (
        "onnxruntime" in library_paths
        or getattr(build.FLAGS, "ort_artifacts_dir", None) is not None
    ):
        raise SystemExit(
            "ORT artifact build must build ONNX Runtime; do not pass "
            "--library-paths=onnxruntime:* or --ort-artifacts-dir."
        )

    artifact_dir = wrapper_args.artifact_dir.resolve()
    backend_tag = build_config["backends"]["onnxruntime"]
    backend_source_dir = find_backend_source(
        wrapper_args.backend_source_dir,
        artifact_dir,
        build.FLAGS.github_organization,
        backend_tag,
    )
    cmake_args = normalize_cmake_args(
        build.backend_cmake_args(
            build_config["images"],
            build_config["components"],
            "onnxruntime",
            str(artifact_dir / "install"),
            library_paths,
        )
    )

    cuda_arch_list = build.FLAGS.cuda_arch_list

    if wrapper_args.ort_docker_build_network:
        cmake_args.append(
            "-DTRITON_ONNXRUNTIME_DOCKER_BUILD_NETWORK:STRING={}".format(
                wrapper_args.ort_docker_build_network
            )
        )
    cmake_args.append(
        "-DTRITON_ONNXRUNTIME_BUILD_AS_ROOT:BOOL={}".format(
            "ON" if wrapper_args.allow_root_ort_build else "OFF"
        )
    )

    source_config = {
        "backend_tag": backend_tag,
        "build_args": build_args,
        "build_parallel": build.FLAGS.build_parallel,
        "build_type": build.FLAGS.build_type,
        "cmake_args": cmake_args,
        "components": build_config["components"],
        "build_environment": (
            "triton-build-container"
            if wrapper_args.build_in_triton_build_container
            else "host"
        ),
        "cuda_arch_list": cuda_arch_list,
        "enable_gpu": build.FLAGS.enable_gpu,
        "github_organization": build.FLAGS.github_organization,
        "images": build_config["images"],
        "min_compute_capability": build.FLAGS.min_compute_capability,
        "ort_build_as_root": wrapper_args.allow_root_ort_build,
        "ort_docker_build_network": wrapper_args.ort_docker_build_network,
        "ort_openvino_version": build.FLAGS.ort_openvino_version,
        "ort_version": build.FLAGS.ort_version,
        "target_machine": build.target_machine(),
        "target_platform": build.target_platform(),
        "triton_buildbase_image": wrapper_args.triton_buildbase_image,
        "triton_container_version": build.FLAGS.triton_container_version,
        "upstream_container_version": build.FLAGS.upstream_container_version,
        "version": build.FLAGS.version,
    }
    source_config["config_hash"] = config_hash(source_config)

    plan = {
        "artifact_dir": str(artifact_dir),
        "artifact_name": "onnxruntime-{}-{}-{}.tar.gz".format(
            build.FLAGS.ort_version,
            build.FLAGS.triton_container_version,
            source_config["config_hash"],
        ),
        "backend_source_dir": str(backend_source_dir),
        "backend_git_commit": git_value(backend_source_dir, ["rev-parse", "HEAD"]),
        "server_git_commit": git_value(SERVER_DIR, ["rev-parse", "HEAD"]),
        "source_config": source_config,
    }

    plan_output = wrapper_args.plan_output or artifact_dir / "ort-build-plan.json"
    write_json(plan_output.resolve(), plan)
    print(f"Wrote ORT build plan to {plan_output.resolve()}")

    if wrapper_args.plan_only:
        return

    backend_script = backend_source_dir / "tools" / "build_ort_artifacts.py"
    if wrapper_args.build_in_triton_build_container:
        build_triton_buildbase(
            build, build_config, wrapper_args.triton_buildbase_image
        )
        run_backend_builder_in_triton_build_container(
            build,
            backend_script,
            plan_output.resolve(),
            artifact_dir,
            backend_source_dir,
            wrapper_args.triton_buildbase_image,
        )
    else:
        subprocess.run(
            [sys.executable, str(backend_script), "--plan", str(plan_output.resolve())],
            check=True,
        )


if __name__ == "__main__":
    main()
