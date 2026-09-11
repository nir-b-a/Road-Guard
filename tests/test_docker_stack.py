"""Tests for the Docker packaging of Road Guard.

These are the guard rails for the container stack: the build context stays small,
the worker's dependency set is container-correct, the compose file wires the
services together the way the demo expects, and the images actually build.

Anything that talks to a real Docker daemon is marked ``@pytest.mark.docker`` and
skips cleanly when Docker is missing (CI without a daemon, a laptop with Docker
Desktop closed):

    pytest tests/test_docker_stack.py                    # everything available
    pytest tests/test_docker_stack.py -m "not docker"    # static checks only, fast
    pytest tests/test_docker_stack.py --rundocker-build  # + build the CPU image

The image build is slow (it pulls ~1 GB of wheels), so it sits behind
``--rundocker-build``; without the flag the suite still covers every static
property of the Dockerfiles, the entrypoint and the compose file.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Paths that must never reach a build context: the big auto-downloadable base
# weights, the CLRerNet checkpoint, the training-output detector, and the source
# trees (datasets, sample videos, per-run outputs) that dwarf the code itself.
FORBIDDEN_IN_CONTEXT = [
    "yolo11x.pt",
    "yolov8m.pt",
    "yolo11n.pt",
    "weights/clrernet_model_best_culane.pth",
    "models/v3_boxes_detector.pt",
    "UnLanedet",
    "datasets",
    "tests_videos",
    "validation_output",
    "demo_out",
    "backend",
    "frontend",
    "application",
    "_worker_jobs",
    "_quarantine_models",
]

# Modules worker.py -> main.process_video_with_models imports. If the allowlist
# drops one of these the image builds fine and then dies on the first job.
REQUIRED_IN_CONTEXT = [
    "worker.py",
    "worker_common.py",
    "lane_render.py",
    "main.py",
    "Constants.py",
    "cloud_env.py",
    "video_handler.py",
    "Objects/World.py",
    "speed_estimation/speed_estimator.py",
    "violations/event.py",
    "violations/export.py",
    "lpr/reader.py",
    "tools/fetch_models.py",
    "tools/ghost_mask.py",
    "tools/shoulder_violation.py",
    "worker/entrypoint.sh",
    "worker/verify_env.py",
    "requirements.txt",
    "requirements-worker.txt",
    "israeli_plates.pt",
    "weights/phase3_v3_yellowprotect.pt",
    "models/tire_yolo11n.pt",
]

# ~19 MB: the code plus the three committed weights (~18 MB of it). The ceiling
# leaves room for a few more modules but trips loudly if a weight or a dataset
# slips past the allowlist.
MAX_CONTEXT_MB = 40


# --------------------------------------------------------------------------- #
# Docker availability
# --------------------------------------------------------------------------- #
def _docker_ok() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True, timeout=30,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


requires_docker = pytest.mark.skipif(not _docker_ok(), reason="Docker daemon not available")


def _run(args, **kw):
    """Run a command in the repo root and return the CompletedProcess (text mode)."""
    return subprocess.run(
        args, cwd=REPO, capture_output=True, text=True,
        encoding="utf-8", errors="replace", **kw,
    )


# --------------------------------------------------------------------------- #
# Step 1 - build context
# --------------------------------------------------------------------------- #
class TestBuildContext:
    """The root .dockerignore is an allowlist; prove it lets the right files through."""

    def test_dockerignore_opens_with_the_catch_all(self):
        text = (REPO / ".dockerignore").read_text(encoding="utf-8")
        meaningful = [ln.strip() for ln in text.splitlines()
                      if ln.strip() and not ln.strip().startswith("#")]
        # `**`, not `*`: a bare `*` never matches a `/`, so nested files such as
        # weights/clrernet_model_best_culane.pth would survive it.
        assert meaningful[0] == "**", (
            "the allowlist must open with `**` so nested paths are excluded too; "
            f"found {meaningful[0]!r}"
        )

    @staticmethod
    @pytest.fixture(scope="class")
    def context_listing():
        """Materialise the real build context and return (relative path -> size, total).

        Copying the context into a throwaway image is the only way to see exactly
        what Docker would send; `docker build` reports a total but not a listing.
        """
        tag = "roadguard-ctx-probe:test"
        dockerfile = "FROM busybox\nCOPY . /ctx\n"
        build = _run(["docker", "build", "-q", "-f", "-", "-t", tag, "."], input=dockerfile)
        assert build.returncode == 0, f"context probe build failed:\n{build.stderr}"

        # BusyBox find has no -printf, so size + path come from a stat loop.
        listing = _run([
            "docker", "run", "--rm", tag, "sh", "-c",
            'find /ctx -type f | while read f; do echo "$(stat -c %s "$f") ${f#/ctx/}"; done',
        ])
        assert listing.returncode == 0, listing.stderr

        paths, total = {}, 0
        for line in listing.stdout.splitlines():
            if not line.strip():
                continue
            size, _, rel = line.partition(" ")
            paths[rel.replace("\\", "/")] = int(size)
            total += int(size)
        _run(["docker", "image", "rm", "-f", tag])
        return paths, total

    @requires_docker
    @pytest.mark.docker
    @pytest.mark.parametrize("needed", REQUIRED_IN_CONTEXT)
    def test_required_file_present(self, context_listing, needed):
        paths, _ = context_listing
        assert needed in paths, f"{needed} was excluded from the build context"

    @requires_docker
    @pytest.mark.docker
    @pytest.mark.parametrize("banned", FORBIDDEN_IN_CONTEXT)
    def test_forbidden_path_absent(self, context_listing, banned):
        paths, _ = context_listing
        leaked = [p for p in paths if p == banned or p.startswith(banned + "/")]
        assert not leaked, f"{banned} leaked into the build context: {leaked[:5]}"

    @requires_docker
    @pytest.mark.docker
    def test_context_stays_small(self, context_listing):
        _, total = context_listing
        mb = total / 1_000_000
        assert mb < MAX_CONTEXT_MB, (
            f"build context grew to {mb:.1f} MB (limit {MAX_CONTEXT_MB} MB) - "
            "something large slipped past the allowlist"
        )

    @requires_docker
    @pytest.mark.docker
    def test_no_compiled_python_or_caches(self, context_listing):
        paths, _ = context_listing
        junk = [p for p in paths
                if p.endswith(".pyc") or "__pycache__" in p or ".pytest_cache" in p]
        assert not junk, f"build cache artefacts leaked: {junk[:5]}"


# --------------------------------------------------------------------------- #
# Step 2 - worker dependency set
# --------------------------------------------------------------------------- #
def _parse_requirements(path: Path) -> dict[str, str]:
    """Map normalised distribution name -> full requirement line, ignoring comments."""
    out = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = line
        for sep in ("==", ">=", "<=", "~=", "<", ">", "!="):
            if sep in name:
                name = name.split(sep, 1)[0]
                break
        out[name.strip().lower().replace("_", "-")] = line
    return out


class TestWorkerRequirements:
    """requirements-worker.txt must be container-correct and must not drift from the
    proven native pins in requirements.txt."""

    @staticmethod
    @pytest.fixture(scope="class")
    def worker_reqs():
        return _parse_requirements(REPO / "requirements-worker.txt")

    @staticmethod
    @pytest.fixture(scope="class")
    def native_reqs():
        return _parse_requirements(REPO / "requirements.txt")

    def test_uses_headless_opencv(self, worker_reqs):
        # The headed build links libGL and wants an X server; the image has neither.
        assert "opencv-python-headless" in worker_reqs
        assert "opencv-python" not in worker_reqs, (
            "the headed opencv-python must not be requested - it and the headless "
            "build both provide `cv2` and would unpack over each other"
        )

    @pytest.mark.parametrize("excluded", [
        "paddleocr",       # ~600 MB of paddlepaddle for a reader that is not active
        "paddlepaddle",
        "easyocr",
        "albumentations",  # UnLanedet / training path only
        "imgaug",
        "scikit-image",
        "scikit-learn",
        "shapely",
        "ninja",
        "addict",
        "yapf",
        "yt-dlp",
    ])
    def test_offline_only_dependency_excluded(self, worker_reqs, excluded):
        assert excluded not in worker_reqs, (
            f"{excluded} is used only by offline tools/ scripts or the CLRerNet "
            "co-run path; it has no place in the worker image"
        )

    @pytest.mark.parametrize("required", [
        "numpy", "ultralytics", "lapx", "scipy", "matplotlib",
        "requests", "pandas", "pillow", "pyyaml", "tqdm",
        "boto3",        # worker.py R2 transport
        "fast-alpr",    # the active plate reader
        "onnxruntime",
    ])
    def test_runtime_dependency_present(self, worker_reqs, required):
        assert required in worker_reqs

    def test_torch_left_to_the_dockerfile(self, worker_reqs):
        # Pinning torch here would let pip resolve a default PyPI build and swap
        # the CUDA runtime out from under the GPU image.
        assert "torch" not in worker_reqs
        assert "torchvision" not in worker_reqs

    def test_pins_match_the_native_environment(self, worker_reqs, native_reqs):
        """Every shared package must carry the identical pin, so the container runs
        the same code as the .venv the pipeline was validated on."""
        drift = []
        for name, line in worker_reqs.items():
            # opencv is the one deliberate substitution; compare to its headed twin.
            native_name = "opencv-python" if name == "opencv-python-headless" else name
            if native_name not in native_reqs:
                continue
            want = native_reqs[native_name].replace("opencv-python", "opencv-python-headless")
            if want != line:
                drift.append(f"{name}: worker={line!r} native={native_reqs[native_name]!r}")
        assert not drift, (
            "requirements-worker.txt drifted from requirements.txt:\n" + "\n".join(drift)
        )


# --------------------------------------------------------------------------- #
# Steps 3 + 4 - the worker images
# --------------------------------------------------------------------------- #
class TestWorkerDockerfiles:
    """Static properties of the two Dockerfiles. These run without a daemon, so a
    broken Dockerfile is caught in seconds instead of after a 10-minute build."""

    @staticmethod
    @pytest.fixture(scope="class")
    def dockerfiles():
        return {
            "cpu": (REPO / "worker" / "Dockerfile.cpu").read_text(encoding="utf-8"),
            "gpu": (REPO / "worker" / "Dockerfile.gpu").read_text(encoding="utf-8"),
        }

    @pytest.mark.parametrize("flavour", ["cpu", "gpu"])
    def test_installs_torch_from_an_explicit_wheel_index(self, dockerfiles, flavour):
        """Plain `pip install torch` resolves the CUDA build on linux regardless of
        the base image, so both flavours must name their index."""
        text = dockerfiles[flavour]
        want = "download.pytorch.org/whl/cpu" if flavour == "cpu" else "download.pytorch.org/whl/cu118"
        assert want in text, f"{flavour} Dockerfile must pin torch to {want}"
        assert "torch==2.1.2" in text and "torchvision==0.16.2" in text

    @pytest.mark.parametrize("flavour", ["cpu", "gpu"])
    def test_removes_the_headed_opencv_ultralytics_drags_in(self, dockerfiles, flavour):
        text = dockerfiles[flavour]
        assert "pip uninstall -y opencv-python" in text
        assert "opencv-python-headless==4.10.0.84" in text
        assert "--no-deps" in text, (
            "reinstalling headless without --no-deps pulls the headed build straight back"
        )

    @pytest.mark.parametrize("flavour,expect", [("cpu", "cpu"), ("gpu", "cuda")])
    def test_runs_the_build_time_verification_gate(self, dockerfiles, flavour, expect):
        assert f"verify_env.py --expect {expect}" in dockerfiles[flavour]

    @pytest.mark.parametrize("flavour", ["cpu", "gpu"])
    def test_drops_to_a_non_root_user(self, dockerfiles, flavour):
        text = dockerfiles[flavour]
        assert "USER worker" in text
        # The chmod has to happen while still root, and before USER.
        assert text.index("chmod +x /app/worker/entrypoint.sh") < text.index("USER worker"), (
            "entrypoint.sh must be made executable before dropping privileges - a "
            "Windows checkout delivers it as 0644"
        )

    @pytest.mark.parametrize("flavour", ["cpu", "gpu"])
    def test_declares_the_cache_and_scratch_volumes(self, dockerfiles, flavour):
        text = dockerfiles[flavour]
        assert "/models-cache" in text and "/work" in text
        assert 'ENTRYPOINT ["/app/worker/entrypoint.sh"]' in text

    def test_gpu_image_uses_a_cu118_base(self, dockerfiles):
        # The pipeline is validated on torch 2.1.2+cu118, which is also the last
        # comfortable stop for the Pascal card this project develops on.
        assert "nvidia/cuda:11.8" in dockerfiles["gpu"]
        assert "-runtime-" in dockerfiles["gpu"], (
            "devel adds nvcc for nothing - no CUDA op is compiled in this image"
        )

    def test_cpu_and_gpu_agree_on_the_python_version(self, dockerfiles):
        assert "python:3.10-slim" in dockerfiles["cpu"]
        assert "python3.10" in dockerfiles["gpu"]


class TestEntrypoint:
    """The entrypoint is what keeps the 114 MB base weight out of the image."""

    @staticmethod
    @pytest.fixture(scope="class")
    def raw():
        return (REPO / "worker" / "entrypoint.sh").read_bytes()

    def test_has_unix_line_endings(self, raw):
        # A CRLF shebang makes the kernel look for "/bin/sh\r"; the container then
        # dies with a "no such file or directory" that names a file which exists.
        assert b"\r\n" not in raw, "entrypoint.sh must use LF endings (see .gitattributes)"

    def test_starts_with_a_shebang(self, raw):
        assert raw.startswith(b"#!/bin/sh")

    def test_relinks_after_fetch(self, raw):
        """ultralytics downloads to a temp file and os.replace()s it onto the target,
        replacing the symlink with a real file - so the move-back step is required."""
        text = raw.decode()
        assert 'mv "/app/$w" "$CACHE/$w"' in text
        assert text.index("fetch_models.py") < text.index('mv "/app/$w"')

    def test_puts_home_on_the_volume(self, raw):
        # fast_alpr caches ~13 MB of ONNX under $HOME/.cache; without this it
        # re-downloads on every container start.
        text = raw.decode()
        assert 'export HOME="$CACHE/home"' in text
        assert "YOLO_CONFIG_DIR" in text

    def test_offers_an_offline_escape_hatch(self, raw):
        assert "SKIP_MODEL_FETCH" in raw.decode()


@requires_docker
@pytest.mark.docker
@pytest.mark.docker_build
class TestBuiltImage:
    """Build the CPU image and assert on the artefact itself."""

    IMAGE = "roadguard-worker:cpu-test"

    @staticmethod
    @pytest.fixture(scope="class")
    def image():
        build = _run(["docker", "build", "-f", "worker/Dockerfile.cpu",
                      "-t", TestBuiltImage.IMAGE, "."])
        assert build.returncode == 0, f"CPU worker build failed:\n{build.stderr[-4000:]}"
        yield TestBuiltImage.IMAGE
        _run(["docker", "image", "rm", "-f", TestBuiltImage.IMAGE])

    def test_dependency_gate_passes_inside_the_image(self, image):
        res = _run(["docker", "run", "--rm", "--entrypoint", "python", image,
                    "worker/verify_env.py", "--expect", "cpu"])
        assert res.returncode == 0, res.stdout + res.stderr
        assert "All checks passed" in res.stdout

    def test_runs_as_non_root(self, image):
        res = _run(["docker", "run", "--rm", "--entrypoint", "id", image, "-u"])
        assert res.stdout.strip() == "1000", "the worker must not run as root"

    def test_large_weights_are_not_baked_into_a_layer(self, image):
        res = _run(["docker", "run", "--rm", "--entrypoint", "sh", image, "-c",
                    'find / -name "yolo11x.pt" -o -name "yolov8m.pt" -o -name "clrernet*" 2>/dev/null'])
        assert not res.stdout.strip(), f"large weights baked into the image: {res.stdout}"

    def test_committed_weights_are_present(self, image):
        res = _run(["docker", "run", "--rm", "--entrypoint", "sh", image, "-c",
                    "ls /app/israeli_plates.pt /app/weights/phase3_v3_yellowprotect.pt "
                    "/app/models/tire_yolo11n.pt"])
        assert res.returncode == 0, f"a committed weight is missing:\n{res.stderr}"

    def test_model_cache_survives_a_restart(self, image):
        """Cold start downloads yolo11x; the second start must link it from the volume."""
        vol = "roadguard-cache-pytest"
        _run(["docker", "volume", "rm", "-f", vol])
        try:
            cold = _run(["docker", "run", "--rm", "-v", f"{vol}:/models-cache", image,
                         "python", "-c", "print('boot')"])
            assert cold.returncode == 0, cold.stdout + cold.stderr
            assert "cached yolo11x.pt onto the volume" in cold.stdout

            warm = _run(["docker", "run", "--rm", "-v", f"{vol}:/models-cache", image,
                         "python", "-c", "print('boot')"])
            assert warm.returncode == 0, warm.stdout + warm.stderr
            assert "linked cached yolo11x.pt" in warm.stdout
            assert "Downloading" not in warm.stdout, "warm start re-downloaded the base weight"
        finally:
            _run(["docker", "volume", "rm", "-f", vol])


# --------------------------------------------------------------------------- #
# Step 5 - compose wiring
# --------------------------------------------------------------------------- #
@requires_docker
@pytest.mark.docker
class TestCompose:
    """Assert on the RESOLVED compose config rather than the YAML text: that is
    what catches a broken anchor merge or a profile that does not apply."""

    @staticmethod
    def _config(*profiles):
        cmd = ["docker", "compose"]
        for p in profiles:
            cmd += ["--profile", p]
        cmd += ["config", "--format", "json"]
        res = _run(cmd)
        assert res.returncode == 0, f"docker compose config failed:\n{res.stderr}"
        return json.loads(res.stdout)

    @staticmethod
    @pytest.fixture(scope="class")
    def default_cfg():
        return TestCompose._config()

    @staticmethod
    @pytest.fixture(scope="class")
    def cpu_cfg():
        return TestCompose._config("cpu")

    @staticmethod
    @pytest.fixture(scope="class")
    def gpu_cfg():
        return TestCompose._config("gpu")

    def test_default_up_is_the_three_service_demo(self, default_cfg):
        """A plain `docker compose up` must not drag in a 3 GB worker image."""
        assert set(default_cfg["services"]) == {"mongo", "backend", "frontend"}

    @pytest.mark.parametrize("profile,service", [("cpu", "worker-cpu"), ("gpu", "worker-gpu")])
    def test_profile_adds_only_its_own_worker(self, profile, service):
        cfg = TestCompose._config(profile)
        assert service in cfg["services"]
        other = "worker-gpu" if profile == "cpu" else "worker-cpu"
        assert other not in cfg["services"], (
            "both workers claim from the same queue - only one may be up at a time"
        )

    @pytest.mark.parametrize("profile,service", [("cpu", "worker-cpu"), ("gpu", "worker-gpu")])
    def test_worker_talks_to_the_backend_over_the_compose_network(self, profile, service):
        cfg = TestCompose._config(profile)
        env = cfg["services"][service]["environment"]
        # worker.env says localhost:5001, which is wrong inside the network; the
        # explicit value here wins because worker.py's loader uses setdefault.
        assert env["SERVER_URL"] == "http://backend:5000"

    @pytest.mark.parametrize("profile,service", [("cpu", "worker-cpu"), ("gpu", "worker-gpu")])
    def test_worker_waits_for_a_healthy_api(self, profile, service):
        cfg = TestCompose._config(profile)
        dep = cfg["services"][service]["depends_on"]["backend"]
        assert dep["condition"] == "service_healthy", (
            "service_started would let the worker poll an API whose Mongo "
            "connection is still opening"
        )

    @pytest.mark.parametrize("profile,service", [("cpu", "worker-cpu"), ("gpu", "worker-gpu")])
    def test_worker_mounts_named_volumes_for_cache_and_scratch(self, profile, service):
        cfg = TestCompose._config(profile)
        mounts = {v["target"]: v for v in cfg["services"][service]["volumes"]}
        assert set(mounts) == {"/models-cache", "/work"}
        for target, mount in mounts.items():
            assert mount["type"] == "volume", f"{target} must be a named volume, not a bind mount"

    @pytest.mark.parametrize("profile,service", [("cpu", "worker-cpu"), ("gpu", "worker-gpu")])
    def test_worker_inherits_the_shared_r2_credentials(self, profile, service):
        """The anchor merge must actually apply; a silent merge failure would leave
        the worker with no R2 keys and every job would fail at download."""
        cfg = TestCompose._config(profile)
        env = cfg["services"][service]["environment"]
        for key in ("R2_ACCOUNT_ID", "R2_BUCKET", "R2_ENDPOINT"):
            assert key in env, f"{key} missing - backend/.env was not merged into {service}"

    def test_backend_has_a_healthcheck_on_the_health_route(self, default_cfg):
        joined = " ".join(default_cfg["services"]["backend"]["healthcheck"]["test"])
        assert "/api/health" in joined
        # alpine has no curl, and adding one purely for a healthcheck is waste.
        assert "curl" not in joined

    def test_backend_still_waits_for_mongo(self, default_cfg):
        cond = default_cfg["services"]["backend"]["depends_on"]["mongo"]["condition"]
        assert cond == "service_healthy"

    def test_gpu_worker_requests_all_devices(self, gpu_cfg):
        gpus = gpu_cfg["services"]["worker-gpu"].get("gpus")
        assert gpus, "the gpu profile must request NVIDIA devices"
        # compose normalises `gpus: all` to count -1.
        assert gpus[0]["count"] == -1

    def test_cpu_worker_requests_no_devices(self, cpu_cfg):
        assert not cpu_cfg["services"]["worker-cpu"].get("gpus"), (
            "the cpu profile exists precisely so it runs without the NVIDIA Container Toolkit"
        )

    def test_the_two_workers_share_one_definition(self, cpu_cfg, gpu_cfg):
        """Everything except build, name, profile and gpus must be identical, so the
        two flavours cannot drift apart."""
        cpu = dict(cpu_cfg["services"]["worker-cpu"])
        gpu = dict(gpu_cfg["services"]["worker-gpu"])
        for key in ("build", "container_name", "profiles", "gpus"):
            cpu.pop(key, None)
            gpu.pop(key, None)
        assert cpu == gpu


# --------------------------------------------------------------------------- #
# Step 6 - the stack actually runs
# --------------------------------------------------------------------------- #
@requires_docker
@pytest.mark.docker
@pytest.mark.docker_build
class TestStackEndToEnd:
    """Bring the real stack up with the cpu profile and assert the pieces reach each
    other. Everything above verifies configuration; this verifies behaviour.

    It uses its own compose project name so it cannot disturb a stack the developer
    already has running, and tears down its containers and volumes afterwards.
    """

    PROJECT = "roadguard-e2e-pytest"

    @staticmethod
    def _compose(*args, timeout=1800):
        return _run(["docker", "compose", "-p", TestStackEndToEnd.PROJECT,
                     "--profile", "cpu", *args], timeout=timeout)

    @staticmethod
    @pytest.fixture(scope="class")
    def stack():
        up = TestStackEndToEnd._compose("up", "-d", "--build")
        if up.returncode != 0:
            TestStackEndToEnd._compose("down", "-v")
            pytest.fail(f"stack failed to come up:\n{up.stderr[-4000:]}")
        try:
            # `up -d` returns once the dependency conditions are met, but the worker
            # still has to load a 114 MB model before it polls. Poll for that rather
            # than sleeping a fixed amount.
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                logs = TestStackEndToEnd._compose("logs", "worker-cpu", timeout=120)
                if "polling" in logs.stdout:
                    break
                time.sleep(5)
            yield
        finally:
            TestStackEndToEnd._compose("down", "-v", timeout=300)

    def test_every_service_is_running(self, stack):
        res = self._compose("ps", "--format", "json")
        # compose emits one JSON object per line here, not a JSON array.
        services = {}
        for line in res.stdout.splitlines():
            if line.strip():
                row = json.loads(line)
                services[row["Service"]] = row["State"]
        assert set(services) == {"mongo", "backend", "frontend", "worker-cpu"}
        assert all(state == "running" for state in services.values()), services

    def test_mongo_and_backend_report_healthy(self, stack):
        res = self._compose("ps", "--format", "json")
        health = {}
        for line in res.stdout.splitlines():
            if line.strip():
                row = json.loads(line)
                health[row["Service"]] = row.get("Health", "")
        # The worker gates on these two; if either is not healthy it never starts.
        assert health["mongo"] == "healthy"
        assert health["backend"] == "healthy"

    def test_health_endpoint_reports_a_connected_database(self, stack):
        res = _run(["docker", "compose", "-p", self.PROJECT, "--profile", "cpu",
                    "exec", "-T", "backend", "node", "-e",
                    "require('http').get('http://127.0.0.1:5000/api/health',"
                    "r=>{let d='';r.on('data',c=>d+=c);r.on('end',()=>"
                    "{console.log(r.statusCode);console.log(d)})})"])
        assert res.returncode == 0, res.stderr
        assert res.stdout.splitlines()[0].strip() == "200"
        assert json.loads(res.stdout.split("\n", 1)[1])["data"]["mongo"] == "connected"

    def test_worker_reaches_the_poll_loop(self, stack):
        """The end-to-end assertion: the worker resolved its models, loaded YOLO and
        is polling the API by its compose service name."""
        logs = self._compose("logs", "worker-cpu")
        assert "All models present and verified" in logs.stdout
        assert "polling http://backend:5000" in logs.stdout

    def test_worker_claims_jobs_through_the_compose_network(self, stack):
        """A 200 on /api/internal/next-job proves DNS, routing and the shared
        INTERNAL_TOKEN all line up - not just that the worker process is alive."""
        logs = self._compose("logs", "backend")
        assert "/api/internal/next-job -> 200" in logs.stdout

    def test_worker_logs_no_errors(self, stack):
        logs = self._compose("logs", "worker-cpu").stdout.lower()
        for bad in ("traceback", "connection refused", "modulenotfounderror"):
            assert bad not in logs, f"worker logged {bad!r}"
