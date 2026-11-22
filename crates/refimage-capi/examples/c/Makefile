# Build the example against an installed refimage (crates/refimage-capi/install.sh):
#
#   make                                   # PKG_CONFIG_PATH=<prefix>/lib/pkgconfig if not in /usr/local
#   ./stream && ./stream_static
#
# `stream` links the shared library (with an rpath to it, so a non-system prefix runs as
# is); `stream_static` links the archive plus the system libraries it needs, which the
# pkg-config file lists as Libs.private.

CC         ?= cc
CFLAGS     ?= -std=c11 -O2 -Wall -Wextra
PKG_CONFIG ?= pkg-config

# Looked up when a build needs them (not for `make clean`).
REFIMAGE_CFLAGS   = $(shell $(PKG_CONFIG) --cflags refimage)
REFIMAGE_LIBS     = $(shell $(PKG_CONFIG) --libs refimage)
REFIMAGE_LIBDIR   = $(shell $(PKG_CONFIG) --variable=libdir refimage)
# Libs.private: what --static adds over the plain --libs.
REFIMAGE_PRIVATE  = $(filter-out $(REFIMAGE_LIBS),$(shell $(PKG_CONFIG) --static --libs refimage))

ifneq ($(filter-out clean,$(or $(MAKECMDGOALS),all)),)
ifeq ($(shell $(PKG_CONFIG) --exists refimage && echo yes),)
$(error pkg-config cannot find refimage: install it, or set PKG_CONFIG_PATH=<prefix>/lib/pkgconfig)
endif
endif

.PHONY: all run clean

all: stream stream_static

stream: stream.c
	$(CC) $(CFLAGS) $(REFIMAGE_CFLAGS) $< -o $@ $(REFIMAGE_LIBS) -Wl,-rpath,$(REFIMAGE_LIBDIR) $(LDFLAGS)

stream_static: stream.c
	$(CC) $(CFLAGS) $(REFIMAGE_CFLAGS) $< -o $@ $(REFIMAGE_LIBDIR)/librefimage_c.a $(REFIMAGE_PRIVATE) $(LDFLAGS)

run: all
	./stream
	./stream_static

clean:
	rm -f stream stream_static last.fits
