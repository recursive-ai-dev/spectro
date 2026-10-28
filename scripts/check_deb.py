#!/usr/bin/env python3
"""Verify a Spectro Debian package without installing or changing system files."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile


REQUIRED = {
    "usr/bin/spectro", "usr/bin/spectro-gui",
    "usr/lib/spectro/ssgs/__init__.py", "usr/lib/spectro/ssgs/__main__.py",
    "usr/lib/spectro/ssgs/model.py", "usr/lib/spectro/ssgs/cli.py",
    "usr/lib/spectro/ssgs/workflows.py", "usr/lib/spectro/ssgs/desktop.py",
    "usr/lib/spectro/ssgs/libssgs_native.so",
    "usr/share/applications/spectro.desktop",
    "usr/share/icons/hicolor/scalable/apps/spectro.svg",
    "usr/share/doc/spectro/LICENSE", "usr/share/doc/spectro/copyright",
    "usr/share/doc/spectro/INSTALL.md", "usr/share/man/man1/spectro.1.gz",
}


def run(command, *, cwd=None, env=None):
    result = subprocess.run([str(value) for value in command], check=True, cwd=cwd,
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()


def check_archive(archive):
    package = run(["dpkg-deb", "--field", archive, "Package"])
    if package != "spectro":
        raise ValueError(f"Expected package spectro, found {package!r}")
    version = run(["dpkg-deb", "--field", archive, "Version"])
    architecture = run(["dpkg-deb", "--field", archive, "Architecture"])
    depends = run(["dpkg-deb", "--field", archive, "Depends"])
    for requirement in ("python3", "python3-numpy", "python3-scipy", "python3-matplotlib",
                        "python3-soundfile", "python3-pywt", "python3-safetensors", "python3-tk", "libc6"):
        names = [item.strip().split(" ", 1)[0] for item in depends.split(",")]
        if requirement not in names:
            raise ValueError(f"Missing runtime dependency: {requirement}")
    if architecture == "all":
        raise ValueError("The compiled native backend requires a machine architecture")
    contents = subprocess.run(["dpkg-deb", "--fsys-tarfile", str(archive)], check=True,
                              stdout=subprocess.PIPE).stdout
    names = set()
    with tarfile.open(fileobj=io.BytesIO(contents), mode="r:") as payload:
        for member in payload.getmembers():
            name = member.name.removeprefix("./")
            if name in ("", "."):
                continue
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "usr":
                raise ValueError(f"Unexpected archive path: {member.name}")
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise ValueError(f"Unexpected archive entry type: {name}")
            if member.uid or member.gid:
                raise ValueError(f"Archive entry is not owned by root: {name}")
            if member.mode & 0o022:
                raise ValueError(f"Archive entry is group/world writable: {name}")
            if any(part in {"__pycache__", ".venv", "training_001", "training_002", ".git"}
                   for part in path.parts) or path.suffix in {".pyc", ".mp3", ".wav"}:
                raise ValueError(f"Unwanted build/development file in package: {name}")
            if member.isfile():
                names.add(name)
        missing = REQUIRED - names
        if missing:
            raise ValueError("Package is missing required files: " + ", ".join(sorted(missing)))
    print(f"Archive OK: {package} {version} ({architecture}); {len(names)} files")
    return version, architecture


def check_extracted(archive, *, interpreter, skip_runtime):
    with tempfile.TemporaryDirectory(prefix="spectro-package-check-") as temporary:
        work = Path(temporary)
        extracted = work / "root"
        run(["dpkg-deb", "--raw-extract", archive, extracted])
        sums = (extracted / "DEBIAN/md5sums").read_text(encoding="utf-8")
        for line in sums.splitlines():
            expected, name = line.split("  ", 1)
            file_path = extracted / name
            if not file_path.is_relative_to(extracted) or ".." in Path(name).parts:
                raise ValueError(f"Invalid checksum path: {name}")
            if hashlib.md5(file_path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Payload checksum mismatch: {name}")
        for name in ("spectro", "spectro-gui"):
            launcher = extracted / "usr/bin" / name
            if launcher.stat().st_mode & 0o111 != 0o111:
                raise ValueError(f"Launcher is not executable: {name}")
            if not launcher.read_text().startswith("#!/usr/bin/python3\n"):
                raise ValueError(f"Launcher does not use distribution Python: {name}")
        if shutil.which("desktop-file-validate"):
            run(["desktop-file-validate", extracted / "usr/share/applications/spectro.desktop"])
        library = extracted / "usr/lib/spectro/ssgs/libssgs_native.so"
        if not library.read_bytes().startswith(b"\x7fELF"):
            raise ValueError("Native library is not ELF")
        print("Payload checksums and desktop integration OK")
        if skip_runtime:
            return
        machine_architecture = run(["dpkg", "--print-architecture"])
        package_architecture = run(["dpkg-deb", "--field", archive, "Architecture"])
        if package_architecture != machine_architecture:
            raise ValueError("Runtime checks require the package's architecture; use --skip-runtime to inspect only")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(extracted / "usr/lib/spectro")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["MPLBACKEND"] = "Agg"
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            environment[variable] = "1"
        launcher = extracted / "usr/bin/spectro"
        command = [interpreter, launcher]
        print(run(command + ["--version"], cwd=work, env=environment))
        doctor = json.loads(run(command + ["doctor", "--json"], cwd=work, env=environment))
        if doctor.get("backend") != "native":
            raise ValueError("Installed package did not activate its native backend")
        # Exercise loading through the installed package rather than an editable
        # source tree; package path and .so are asserted in an isolated process.
        run([interpreter, "-c", (
            "import pathlib, ssgs, ssgs.native; "
            "assert pathlib.Path(ssgs.__file__).is_relative_to(pathlib.Path(__import__('sys').argv[1])); "
            "assert ssgs.native._lib is not None; import ssgs.desktop"
        ), extracted], cwd=work, env=environment)
        demo = work / "demo"
        run(command + ["demo", "--output-dir", demo], cwd=work, env=environment)
        models = list(demo.glob("*.npz")) + list(demo.glob("*.safetensors"))
        audio = list(demo.glob("*.wav"))
        if not models or not audio:
            raise ValueError("Demo did not produce both a model and WAV audio")
        json.loads(run(command + ["inspect", models[0], "--json"], cwd=work, env=environment))
        rendered = work / "rendered.wav"
        run(command + ["generate", models[0], "--output", rendered,
                       "--duration", "0.25", "--sample-rate", "16000", "--seed", "7"], cwd=work, env=environment)
        run([interpreter, "-c", (
            "import soundfile as sf, numpy as np, sys; "
            "audio, rate = sf.read(sys.argv[1]); "
            "assert rate == 16000 and len(audio) == 4000; "
            "assert np.isfinite(audio).all() and np.max(np.abs(audio)) > 0"
        ), rendered], cwd=work, env=environment)
        print("Runtime OK: native backend, desktop import, demo training, model inspect, and exact-length WAV generation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", nargs="+", type=Path)
    parser.add_argument("--python", default="/usr/bin/python3", help="Runtime interpreter (default: /usr/bin/python3)")
    parser.add_argument("--skip-runtime", action="store_true", help="Check structure and checksums without importing application dependencies")
    args = parser.parse_args(argv)
    # Preserve virtualenv symlinks: resolving .venv/bin/python to /usr/bin/python3
    # would discard the selected virtual environment and its installed packages.
    interpreter = str(Path(args.python).absolute()) if Path(args.python).is_file() else shutil.which(args.python)
    try:
        if not args.skip_runtime and interpreter is None:
            raise ValueError(f"Runtime interpreter not found: {args.python}")
        for archive in args.packages:
            archive = archive.resolve(strict=True)
            check_archive(archive)
            check_extracted(archive, interpreter=interpreter, skip_runtime=args.skip_runtime)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"check_deb: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr.strip(), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
