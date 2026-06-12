#!/usr/bin/env python3
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import os
import platform
import pathlib
import subprocess
import sys
import tempfile
import unittest


SERVER_DIR = pathlib.Path(__file__).resolve().parents[1]
WRAPPER = SERVER_DIR / "tools" / "build_ort_from_triton_config.py"


def run(cmd, env=None, check=True):
    result = subprocess.run(
        [str(arg) for arg in cmd],
        cwd=SERVER_DIR,
        env=env,
        text=True,
        capture_output=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            "command failed with {}\ncmd: {}\nstdout:\n{}\nstderr:\n{}".format(
                result.returncode, " ".join(str(arg) for arg in cmd), result.stdout, result.stderr
            )
        )
    return result


def write_fake_artifact(path, cuda_arch_list):
    (path / "include").mkdir(parents=True)
    (path / "lib").mkdir()
    (path / "lib" / "libonnxruntime.so").touch()
    with (path / "triton-ort-artifact.json").open("w") as f:
        json.dump(
            {
                "plan": {
                    "source_config": {
                        "cuda_arch_list": cuda_arch_list,
                        "target_machine": platform.machine().lower(),
                    },
                },
            },
            f,
        )


class OrtArtifactWorkflowTest(unittest.TestCase):
    def test_plan_only_covers_common_parameters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            backend_source_dir = SERVER_DIR.parent / "triton-onnxruntime_backend"
            if not backend_source_dir.exists():
                backend_source_dir = SERVER_DIR
            components = {}
            for name in ("common", "core", "backend"):
                components[name] = tmp / f"triton-{name}"
                components[name].mkdir()
            run(
                [
                    sys.executable,
                    WRAPPER,
                    "--artifact-dir",
                    tmp / "ort",
                    "--backend-source-dir",
                    backend_source_dir,
                    "--no-local-component-repos",
                    "--component-source-dir",
                    f"common={components['common']}",
                    "--component-source-dir",
                    f"core={components['core']}",
                    "--component-source-dir",
                    f"backend={components['backend']}",
                    "--build-in-triton-build-container",
                    "--plan-only",
                    "--",
                    "--enable-gpu",
                    "--cuda-arch-list",
                    "8.6",
                    "--override-backend-cmake-arg",
                    "onnxruntime:TRITON_ENABLE_ONNXRUNTIME_OPENVINO=OFF",
                    "-j",
                    "6",
                ],
            )

            with (tmp / "ort" / "ort-build-plan.json").open() as f:
                plan = json.load(f)
            config = plan["source_config"]
            cmake_args = config["cmake_args"]

            self.assertEqual(config["build_parallel"], 6)
            self.assertEqual(config["build_environment"], "triton-build-container")
            self.assertEqual(config["ort_docker_build_network"], "host")
            self.assertEqual(
                config["component_source_dirs"],
                {name: str(path) for name, path in sorted(components.items())},
            )
            self.assertEqual(config["cuda_arch_list"], "8.6")
            self.assertTrue(config["enable_gpu"])
            self.assertEqual(
                sum("TRITON_CUDA_ARCH_LIST" in arg for arg in cmake_args), 1
            )
            self.assertIn("-DTRITON_ENABLE_ONNXRUNTIME_OPENVINO:BOOL=OFF", cmake_args)
            self.assertIn(
                "-DTRITON_ONNXRUNTIME_DOCKER_BUILD_NETWORK:STRING=host",
                cmake_args,
            )
            self.assertIn("-DTRITON_ONNXRUNTIME_BUILD_AS_ROOT:BOOL=OFF", cmake_args)
            for name in ("COMMON", "CORE", "BACKEND"):
                self.assertEqual(
                    sum(f"FETCHCONTENT_SOURCE_DIR_REPO-{name}" in arg for arg in cmake_args),
                    1,
                )

    def test_full_build_reuses_matching_artifact_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            artifact = tmp / "onnxruntime"
            write_fake_artifact(artifact, "8.6")

            env = os.environ.copy()
            env["CUDA_ARCH_LIST"] = "8.6"
            result = run(
                [
                    sys.executable,
                    SERVER_DIR / "build.py",
                    "--dryrun",
                    "--no-container-build",
                    "--no-core-build",
                    "--build-dir",
                    tmp / "build",
                    "--install-dir",
                    tmp / "install",
                    "--backend",
                    "onnxruntime",
                    "--ort-artifacts-dir",
                    artifact,
                    "--enable-gpu",
                ],
                env=env,
            )
            self.assertEqual(result.returncode, 0)

            with (tmp / "build" / "cmake_build").open() as f:
                cmake_build = f.read()
            self.assertIn("TRITON_ONNXRUNTIME_ARTIFACTS_PATH", cmake_build)
            self.assertIn(str(artifact), cmake_build)

            env["CUDA_ARCH_LIST"] = "8.0"
            result = run(
                [
                    sys.executable,
                    SERVER_DIR / "build.py",
                    "--dryrun",
                    "--no-container-build",
                    "--no-core-build",
                    "--build-dir",
                    tmp / "mismatch-build",
                    "--install-dir",
                    tmp / "mismatch-install",
                    "--backend",
                    "onnxruntime",
                    "--ort-artifacts-dir",
                    artifact,
                    "--enable-gpu",
                ],
                env=env,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("cuda_arch_list does not match", result.stderr)


if __name__ == "__main__":
    unittest.main()
