# Build the compiled SSGS numeric kernels.
# Usage: make   (produces src/ssgs/libssgs_native.so, loadable by ssgs.native)

PYTHON ?= python3
CC ?= gcc
CFLAGS ?= -shared -fPIC -O3

NATIVE_SRC := src/ssgs/ssgs_native.c
NATIVE_LIB := src/ssgs/libssgs_native.so

$(NATIVE_LIB): $(NATIVE_SRC)
	$(CC) $(CFLAGS) -o $@ $(NATIVE_SRC) -lm

.PHONY: all clean rebuild test deb check-package
all: $(NATIVE_LIB)

test: $(NATIVE_LIB)
	$(PYTHON) -m pytest -q

deb:
	$(PYTHON) scripts/build_deb.py --force

check-package: deb
	$(PYTHON) scripts/check_deb.py dist/spectro_*.deb --python $(PYTHON)

clean:
	rm -f $(NATIVE_LIB)

rebuild: clean
	$(MAKE) $(NATIVE_LIB)
