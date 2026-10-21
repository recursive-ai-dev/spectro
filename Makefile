# Build the compiled SSGS numeric kernels.
# Usage: make   (produces src/ssgs/libssgs_native.so, loadable by ssgs.native)

CC ?= gcc
CFLAGS ?= -shared -fPIC -O3

NATIVE_SRC := src/ssgs/ssgs_native.c
NATIVE_LIB := src/ssgs/libssgs_native.so

$(NATIVE_LIB): $(NATIVE_SRC)
	$(CC) $(CFLAGS) -o $@ $(NATIVE_SRC) -lm

.PHONY: clean rebuild
clean:
	rm -f $(NATIVE_LIB)

rebuild: clean
	$(MAKE) $(NATIVE_LIB)
