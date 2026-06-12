#!/usr/bin/env python3
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import os
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


if __name__ == "__main__":
    unittest.main()
