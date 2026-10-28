#!/usr/bin/env python3
"""Build a native Debian package without root, pip, or application dependencies.

The package uses the distribution Python and apt-managed runtime dependencies.
Only tracked-style application sources and explicit integration assets are
staged; training recordings, local environments, and generated data are excluded.
"""

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DEPENDENCIES = [
    "python3 (>= 3.9)",
    "python3-numpy (>= 1.21)",
    "python3-scipy (>= 1.7)",
    "python3-matplotlib (>= 3.4)",
    "python3-soundfile (>= 0.10)",
    "python3-pywt (>= 1.1)",
    "python3-safetensors (>= 0.4)",
    "python3-tk",
    "libsndfile1",
]


def run(arguments, *, cwd=None, capture=False, env=None):
    result = subprocess.run(
        [str(value) for value in arguments], cwd=cwd, env=env,
        check=True, text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else None


def project_version():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = text.split("[project]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"', project, re.MULTILINE)
    if not match:
        raise ValueError("No project version in pyproject.toml")
    return match.group(1)


def source_epoch():
    supplied = os.environ.get("SOURCE_DATE_EPOCH")
    if supplied is not None:
        epoch = int(supplied)
    elif (ROOT / ".git").exists() and shutil.which("git"):
        epoch = int(run(["git", "log", "-1", "--format=%ct"], cwd=ROOT, capture=True))
    else:
        epoch = 1792540800  # Stable timestamp fallback for source archives.
    if epoch < 0:
        raise ValueError("SOURCE_DATE_EPOCH must be non-negative")
    return epoch


def put(source, destination, mode=0o644):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(mode)


def put_text(destination, value, mode=0o644):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(value, encoding="utf-8")
    destination.chmod(mode)


def native_dependencies(native_library, work):
    # dpkg-shlibdeps consults the build machine's symbols metadata to derive the
    # actual minimum libc version; never guess based on the distribution name.
    put_text(work / "debian/control", """Source: spectro
Section: sound
Priority: optional
Maintainer: Recursive AI Devs <psychcoherence@gmail.com>
Build-Depends: libc6-dev

Package: spectro
Architecture: any
Description: Spectral-state guided audio synthesis workbench
""")
    output = run(["dpkg-shlibdeps", "-O", "-e" + str(native_library.relative_to(work))], cwd=work, capture=True)
    for line in output.splitlines():
        if line.startswith("shlibs:Depends="):
            return line.split("=", 1)[1]
    raise RuntimeError("dpkg-shlibdeps did not report native runtime dependencies")


def normalize_tree(directory, epoch):
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_dir():
            path.chmod(0o755)
        os.utime(path, (epoch, epoch), follow_symlinks=False)
    directory.chmod(0o755)
    os.utime(directory, (epoch, epoch))


def build(args):
    for tool in ("dpkg", "dpkg-deb", "dpkg-shlibdeps", "cc"):
        if not shutil.which(tool):
            raise RuntimeError(f"Required build tool is missing: {tool}; install build-essential dpkg-dev")
    version = args.version or project_version()
    if not re.fullmatch(r"[0-9][A-Za-z0-9.+~]*", version):
        raise ValueError("Version must be a Debian-compatible upstream version (for example 2.0.0)")
    if not re.fullmatch(r"[A-Za-z0-9.+~]+", args.revision):
        raise ValueError("Debian revision contains unsupported characters")
    deb_version = f"{version}-{args.revision}"
    run(["dpkg", "--validate-version", deb_version])
    architecture = run(["dpkg", "--print-architecture"], capture=True)
    epoch = source_epoch()
    output_directory = Path(args.output_dir).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    filename = f"spectro_{deb_version}_{architecture}.deb"
    output = output_directory / filename
    checksum_file = output_directory / (filename + ".sha256")
    manifest_file = output_directory / (filename + ".json")
    if not args.force:
        for destination in (output, checksum_file, manifest_file):
            if destination.exists():
                raise FileExistsError(f"Output already exists: {destination}; use --force to replace it")

    with tempfile.TemporaryDirectory(prefix=".spectro-deb-", dir=output_directory) as temporary:
        work = Path(temporary)
        stage = work / "debian/spectro"
        (stage / "DEBIAN").mkdir(parents=True)
        package = stage / "usr/lib/spectro/ssgs"
        package.mkdir(parents=True)
        for source in sorted((ROOT / "src/ssgs").rglob("*.py")):
            if "__pycache__" not in source.parts:
                put(source, package / source.relative_to(ROOT / "src/ssgs"))

        native = package / "libssgs_native.so"
        run([
            "cc", "-shared", "-fPIC", "-O3", "-fstack-protector-strong",
            "-D_FORTIFY_SOURCE=2", "-Wl,-z,relro,-z,now", "-Wl,--build-id=sha1",
            "-o", native, ROOT / "src/ssgs/ssgs_native.c", "-lm",
        ])
        native.chmod(0o644)
        if shutil.which("strip"):
            run(["strip", "--strip-unneeded", native])
        native_depends = native_dependencies(native, work)

        for name in ("spectro", "spectro-gui"):
            put(ROOT / "packaging" / name, stage / "usr/bin" / name, 0o755)
        put(ROOT / "packaging/spectro.desktop", stage / "usr/share/applications/spectro.desktop")
        put(ROOT / "packaging/spectro.svg", stage / "usr/share/icons/hicolor/scalable/apps/spectro.svg")
        if shutil.which("desktop-file-validate"):
            run(["desktop-file-validate", stage / "usr/share/applications/spectro.desktop"])
        docs = stage / "usr/share/doc/spectro"
        for source, target in (
            ("LICENSE", "LICENSE"), ("README.md", "README.md"),
            ("docs/INSTALL.md", "INSTALL.md"), ("packaging/copyright", "copyright"),
        ):
            put(ROOT / source, docs / target)
        date = datetime.fromtimestamp(epoch, timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
        changelog = (
            f"spectro ({deb_version}) unstable; urgency=medium\n\n"
            "  * Desktop audio workbench, supported CLI, and native Debian package.\n\n"
            f" -- Recursive AI Devs <psychcoherence@gmail.com>  {date}\n"
        )
        (docs / "changelog.Debian.gz").write_bytes(gzip.compress(changelog.encode(), mtime=0))
        man = stage / "usr/share/man/man1/spectro.1.gz"
        man.parent.mkdir(parents=True)
        man.write_bytes(gzip.compress((ROOT / "packaging/spectro.1").read_bytes(), mtime=0))

        installed_size = (sum(p.stat().st_size for p in stage.rglob("*") if p.is_file()) + 1023) // 1024
        depends = ", ".join(PYTHON_DEPENDENCIES + [native_depends])
        control = f"""Package: spectro
Version: {deb_version}
Section: sound
Priority: optional
Architecture: {architecture}
Maintainer: Recursive AI Devs <psychcoherence@gmail.com>
Installed-Size: {installed_size}
Depends: {depends}
Recommends: xdg-utils
Homepage: https://github.com/recursive-ai-dev/spectro
Description: Spectral-state guided generative audio workbench
 Train spectral models from recordings, generate new audio, and inspect
 waveforms and spectrograms in a local desktop workbench or command line.
 Includes compiled numerical kernels, deterministic seeded workflows,
 portable model exports, and an offline synthetic demo.
"""
        put_text(stage / "DEBIAN/control", control)
        sums = []
        for path in sorted((stage / "usr").rglob("*")):
            if path.is_file():
                sums.append(f"{hashlib.md5(path.read_bytes()).hexdigest()}  {path.relative_to(stage)}\n")
        put_text(stage / "DEBIAN/md5sums", "".join(sums))
        normalize_tree(stage, epoch)
        environment = dict(os.environ, SOURCE_DATE_EPOCH=str(epoch))
        temporary_deb = work / filename
        run(["dpkg-deb", "--root-owner-group", "-Zxz", "-z6", "--build", stage, temporary_deb], env=environment)
        run(["dpkg-deb", "--info", temporary_deb])
        sha256 = hashlib.sha256(temporary_deb.read_bytes()).hexdigest()
        manifest = {
            "package": "spectro", "version": deb_version, "architecture": architecture,
            "sha256": sha256, "source_date_epoch": epoch,
            "native_dependencies": native_depends, "depends": depends,
        }
        temporary_checksum = work / "checksum"
        temporary_checksum.write_text(f"{sha256}  {filename}\n", encoding="utf-8")
        temporary_manifest = work / "manifest"
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_deb, output)
        os.replace(temporary_checksum, checksum_file)
        os.replace(temporary_manifest, manifest_file)
    print(f"Built {output}\nSHA256 {sha256}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "dist"), help="Artifact directory (default: dist)")
    parser.add_argument("--version", help="Override the upstream version read from pyproject.toml")
    parser.add_argument("--revision", default="1", help="Debian revision (default: 1)")
    parser.add_argument("--force", action="store_true", help="Replace an existing package and its checksum/manifest")
    args = parser.parse_args(argv)
    try:
        build(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"build_deb: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
