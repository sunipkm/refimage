//! `include/refimage.h` must match what cbindgen generates from the sources.
//!
//! Regenerate it with `REFIMAGE_BLESS=1 cargo test -p refimage-capi --test header`.

use std::path::Path;

#[test]
fn header_is_up_to_date() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut config = cbindgen::Config::from_file(dir.join("cbindgen.toml")).expect("cbindgen.toml");
    config.after_includes = config
        .after_includes
        .map(|s| s.replace("@VERSION@", env!("CARGO_PKG_VERSION")));
    let mut generated = Vec::new();
    cbindgen::Builder::new()
        .with_crate(dir)
        .with_config(config)
        .generate()
        .expect("cbindgen failed")
        .write(&mut generated);
    let generated = String::from_utf8(generated).expect("header is UTF-8");

    let path = dir.join("include/refimage.h");
    if std::env::var_os("REFIMAGE_BLESS").is_some() {
        std::fs::write(&path, &generated).expect("write include/refimage.h");
        return;
    }
    let committed = std::fs::read_to_string(&path).unwrap_or_default();
    assert!(
        committed == generated,
        "include/refimage.h is stale; regenerate it with \
         `REFIMAGE_BLESS=1 cargo test -p refimage-capi --test header`"
    );
}
