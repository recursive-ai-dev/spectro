# Install Spectro

Spectro provides a desktop audio workbench, the `spectro` command, and native DSP
kernels. The Debian package uses the system Python and distribution packages.
No Python environment or compiler is needed on the installed machine.

## Install the Debian package

From the repository directory, install the package built for your machine's
architecture:

```bash
sudo apt install ./dist/spectro_2.0.0-1_amd64.deb
```

`apt` installs dependencies from your enabled repositories. Launch **Spectro**
from the desktop application menu, run `spectro-gui`, or use the command line:

```bash
spectro --version
spectro doctor
spectro demo --output-dir "$HOME/spectro-demo"
spectro train ./recordings --output my-model.npz --states 16 --iterations 10
spectro generate my-model.npz --output result.wav --duration 5 --seed 42
spectro adapt my-model.npz ./new-recordings --output updated.npz --adaptation-rate 0.25
spectro inspect my-model.npz --json
```

The desktop requires a graphical session. CLI commands work without a display.
WAV, FLAC, OGG, AIFF and other formats supported by the installed libsndfile can
be used for training; available codecs vary by distribution. Generated audio is
written as WAV. Existing outputs require an explicit overwrite option.

The package built in this workspace is **amd64**, built on **Ubuntu 26.04**.
Its control file declares the actual minimum libc version required by the
compiled kernels. Inspect the package's requirements before using it on another
release:

```bash
dpkg-deb --field dist/spectro_2.0.0-1_amd64.deb Depends Architecture
```

Debian 13 and Ubuntu 26.04 provide the named Python dependencies, including
`python3-safetensors`. Other releases need all of those dependencies in their
enabled apt repositories. Build on the oldest release you intend to support;
a package built against a newer libc may not install on an older distribution.
For another architecture, build on that architecture. Cross compilation is not
configured by this project.

To remove the installed program:

```bash
sudo apt remove spectro
```

Your recordings, exported models, and generated audio remain in the locations
you selected. The package installs application files under `/usr/lib/spectro`,
launchers under `/usr/bin`, and documentation under `/usr/share/doc/spectro`.

## Build the package

On Debian or Ubuntu, install the build tools once:

```bash
sudo apt install build-essential dpkg-dev python3
```

Build without root:

```bash
make deb
# Equivalent:
python3 scripts/build_deb.py
```

The builder compiles the native kernels, derives their shared-library
dependencies with `dpkg-shlibdeps`, stages the application and desktop assets,
and writes a `.deb`, `.deb.sha256`, and `.deb.json` manifest into `dist/`.
It does not install anything. Recordings and sample training directories are
excluded. `desktop-file-validate` runs when available.

Use `--output-dir DIRECTORY`, `--version 2.0.1`, or `--revision 2` to select a
different artifact destination or package version. Replacing an existing
artifact requires `--force`. `SOURCE_DATE_EPOCH` controls archive timestamps;
the default is the most recent Git commit timestamp, or a fixed fallback for a
source archive. With identical source, compiler, tools and epoch, builds are
reproducible.

```bash
python3 scripts/build_deb.py --output-dir ./dist --force
cd dist
sha256sum --check spectro_2.0.0-1_amd64.deb.sha256
```

## Verify without installing

Structural checks validate package metadata, permissions, archive contents,
desktop integration, and payload checksums. Runtime checks extract into a
temporary directory and exercise the installed CLI, native backend, synthetic
training, model inspection, and WAV output. They do not modify system files.

```bash
python3 scripts/check_deb.py dist/spectro_2.0.0-1_amd64.deb
# Or use a project environment that has the runtime dependencies installed:
python3 scripts/check_deb.py dist/spectro_2.0.0-1_amd64.deb --python .venv/bin/python
# Structural checks only:
python3 scripts/check_deb.py dist/spectro_2.0.0-1_amd64.deb --skip-runtime
```

The runtime interpreter must provide NumPy, SciPy, Matplotlib, SoundFile,
PyWavelets, safetensors, and tkinter. `make check-package` uses the interpreter
selected by the Makefile's `PYTHON` variable.

## Develop from source

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
make
.venv/bin/spectro gui
.venv/bin/python -m pytest -q
```

If your distribution supplies `python3-venv` separately, install that package
first. Tkinter is supplied by the distribution's `python3-tk` package. The native
kernels are optional in source installs; `make` enables them. Debian packages
always include the compiled kernels.

Packaging references: [dpkg-deb](https://manpages.debian.org/bookworm/dpkg/dpkg-deb.1.en.html),
[dpkg-shlibdeps](https://manpages.debian.org/bookworm/dpkg-dev/dpkg-shlibdeps.1.en.html),
[Debian safetensors package](https://packages.debian.org/trixie/python3-safetensors).
