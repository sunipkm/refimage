#!/bin/sh
# Build and install refimage's C API: refimage.h, librefimage_c (shared, versioned by
# ABI, plus the unversioned development symlink), librefimage_c.a, a pkg-config file
# and a CMake package.
#
#   crates/refimage-capi/install.sh [PREFIX]          # default /usr/local
#   DESTDIR=/tmp/stage crates/refimage-capi/install.sh /usr
#
# Linux and macOS. On Windows, run `cargo build --release -p refimage-capi` and take
# target/release/refimage_c.dll (+ refimage_c.dll.lib import library), refimage_c.lib
# (static) and include/refimage.h.
set -eu

prefix=${1:-/usr/local}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)

cargo build --release -p refimage-capi --manifest-path "$root/Cargo.toml"
target=${CARGO_TARGET_DIR:-$root/target}/release

# The workspace version (every crate inherits it; a release stamps the tag's version there).
version=$(awk '/^\[workspace\.package\]/ { in_pkg = 1; next } /^\[/ { in_pkg = 0 }
    in_pkg && /^version *=/ { gsub(/[" ]/, "", $0); sub(/^version=/, ""); print; exit }' "$root/Cargo.toml")
if [ -z "$version" ]; then
    echo "install.sh: no version in [workspace.package] of $root/Cargo.toml" >&2
    exit 1
fi
abi=$(sed -n 's/^const ABI: &str = "\(.*\)";$/\1/p' "$here/build.rs")
dest=${DESTDIR:-}$prefix

case "$(uname -s)" in
Darwin)
    shared=librefimage_c.dylib
    versioned=librefimage_c.$abi.dylib
    private="-framework CoreFoundation -lm"
    ;;
*)
    shared=librefimage_c.so
    versioned=librefimage_c.so.$abi
    private="-lgcc_s -lutil -lrt -lpthread -ldl -lm"
    ;;
esac

install -d "$dest/include" "$dest/lib/pkgconfig" "$dest/lib/cmake/refimage"
install -m 644 "$here/include/refimage.h" "$dest/include/refimage.h"
install -m 644 "$target/librefimage_c.a" "$dest/lib/librefimage_c.a"
install -m 755 "$target/$shared" "$dest/lib/$versioned"
ln -sf "$versioned" "$dest/lib/$shared"
if [ "$(uname -s)" = Darwin ]; then
    # Absolute install name, so programs find the library without an rpath.
    install_name_tool -id "$prefix/lib/$versioned" "$dest/lib/$versioned"
fi

sed -e "s|@PREFIX@|$prefix|" -e "s|@VERSION@|$version|" -e "s|@LIBS_PRIVATE@|$private|" \
    "$here/cmake/refimage.pc.in" >"$dest/lib/pkgconfig/refimage.pc"
sed -e "s|@SHARED@|$versioned|" -e "s|@SONAME@|$versioned|" \
    "$here/cmake/refimageConfig.cmake.in" >"$dest/lib/cmake/refimage/refimageConfig.cmake"
sed -e "s|@VERSION@|${version%%-*}|" \
    "$here/cmake/refimageConfigVersion.cmake.in" >"$dest/lib/cmake/refimage/refimageConfigVersion.cmake"

echo "installed refimage $version (ABI $abi) into $dest"
