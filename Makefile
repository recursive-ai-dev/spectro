# Build the compiled SSGS numeric kernels.
# Usage: make   (produces libssgs_native.so, loadable by ssgs_native.py)

CC ?= gcc
CFLAGS ?= -shared -fPIC -O3

libssgs_native.so: ssgs_native.c
	$(CC) $(CFLAGS) -o $@ ssgs_native.c -lm

.PHONY: clean rebuild
clean:
	rm -f libssgs_native.so

rebuild: clean
	$(MAKE) libssgs_native.so
