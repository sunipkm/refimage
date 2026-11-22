//! Give the shared library an ABI-versioned name so programs linked against one ABI
//! never load an incompatible one: `librefimage_c.so.<ABI>` on Linux and other ELF
//! systems, `@rpath/librefimage_c.<ABI>.dylib` on macOS. Keep `ABI` equal to
//! `REF_ABI_VERSION` in src/lib.rs (a unit test checks).

const ABI: &str = "1";

fn main() {
    println!("cargo:rustc-env=REFIMAGE_SONAME_ABI={ABI}");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let family = std::env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();
    if os == "macos" || os == "ios" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/librefimage_c.{ABI}.dylib");
    } else if family == "unix" {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,librefimage_c.so.{ABI}");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
