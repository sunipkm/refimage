//! Compile `tests/c/test_api.c` against the freshly built `librefimage_c` (static and
//! shared) with the system C compiler, and run it. With GCC/Clang on Linux the
//! static build also runs under AddressSanitizer + LeakSanitizer + UBSan, so a leaked
//! handle or buffer fails the test.
//!
//! Set `CC` to choose the compiler. Skipped when no compiler is found, and on
//! Windows (where `cl.exe` needs a developer prompt; CI covers it separately).

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;

fn crate_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// `target/<profile>/deps`, next to this test binary: where `cargo test` writes the
/// `librefimage_c.{a,so,dylib}` it just built. (The copies in `target/<profile>` are
/// only refreshed by `cargo build`, so they can be stale.)
fn lib_dir() -> PathBuf {
    let exe = std::env::current_exe().expect("test executable path");
    exe.parent().expect("deps dir").to_path_buf()
}

fn compiler() -> Option<String> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    Command::new(&cc)
        .arg("--version")
        .output()
        .ok()?
        .status
        .success()
        .then_some(cc)
}

fn system_libs() -> &'static [&'static str] {
    if cfg!(target_os = "macos") {
        &[
            "-framework",
            "CoreFoundation",
            "-framework",
            "Security",
            "-lm",
        ]
    } else {
        &["-lgcc_s", "-lutil", "-lrt", "-lpthread", "-ldl", "-lm"]
    }
}

fn build(cc: &str, out: &Path, link: &[String], extra: &[&str]) -> bool {
    let status = Command::new(cc)
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-g", "-O1"])
        .args(extra)
        .arg("-I")
        .arg(crate_dir().join("include"))
        .arg(crate_dir().join("tests/c/test_api.c"))
        .arg("-o")
        .arg(out)
        .args(link)
        .args(system_libs())
        .status()
        .expect("run the C compiler");
    status.success()
}

fn run(exe: &Path, envs: &[(&str, &str)]) {
    let tmp = std::env::temp_dir();
    let out = Command::new(exe)
        .arg(&tmp)
        .envs(envs.iter().copied())
        .output()
        .expect("run the C test");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "{} failed ({})\n--- stdout\n{stdout}\n--- stderr\n{stderr}",
        exe.display(),
        out.status
    );
    assert!(stdout.contains("all C API checks passed"), "{stdout}");
}

#[test]
fn c_test_program_static() {
    let Some(cc) = compiler() else {
        eprintln!("no C compiler; skipping");
        return;
    };
    let lib = lib_dir().join("librefimage_c.a");
    assert!(lib.exists(), "{} was not built", lib.display());
    let dir = lib_dir();
    let link = [lib.display().to_string()];

    let plain = dir.join("capi_test_static");
    assert!(
        build(&cc, &plain, &link, &[]),
        "compiling test_api.c failed"
    );
    run(&plain, &[]);

    // Sanitizers, where the toolchain supports them.
    if cfg!(target_os = "linux") {
        let san = dir.join("capi_test_asan");
        let flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"];
        if build(&cc, &san, &link, &flags) {
            run(
                &san,
                &[
                    ("ASAN_OPTIONS", "detect_leaks=1"),
                    ("UBSAN_OPTIONS", "halt_on_error=1"),
                ],
            );
        } else {
            eprintln!("sanitizers unavailable; skipped the ASan build");
        }
    }
}

#[test]
fn c_test_program_shared() {
    let Some(cc) = compiler() else {
        eprintln!("no C compiler; skipping");
        return;
    };
    let dir = lib_dir();
    let exe = dir.join("capi_test_shared");
    let link = [
        format!("-L{}", dir.display()),
        "-lrefimage_c".into(),
        format!("-Wl,-rpath,{}", dir.display()),
    ];
    // Make sure the linker picks the shared library, not the archive next to it.
    let shared = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let built = dir.join(format!("librefimage_c.{shared}"));
    assert!(built.exists());
    // The library's soname / install name carries the ABI version (see build.rs);
    // provide that name next to cargo's unversioned file, as an install would.
    let abi = env!("REFIMAGE_SONAME_ABI");
    let versioned = if cfg!(target_os = "macos") {
        dir.join(format!("librefimage_c.{abi}.dylib"))
    } else {
        dir.join(format!("librefimage_c.so.{abi}"))
    };
    let _ = std::fs::remove_file(&versioned);
    std::os::unix::fs::symlink(&built, &versioned).expect("symlink the versioned library");
    let link: Vec<String> = if cfg!(target_os = "macos") {
        link.to_vec()
    } else {
        let mut l = vec!["-Wl,-Bdynamic".to_string()];
        l.extend(link);
        l
    };
    assert!(build(&cc, &exe, &link, &[]), "compiling test_api.c failed");
    run(&exe, &[]);
}

/// Compile `src` (C, or C++ when `cxx`) as an object file with `-Werror`; `true` on success.
fn compiles(cc: &str, name: &str, src: &str, flags: &[&str]) -> bool {
    let dir = lib_dir();
    let file = dir.join(name);
    std::fs::write(&file, src).expect("write snippet");
    Command::new(cc)
        .args(["-Werror", "-c", "-o"])
        .arg(dir.join(format!("{name}.o")))
        .args(flags)
        .arg("-I")
        .arg(crate_dir().join("include"))
        .arg(&file)
        .output()
        .expect("run the compiler")
        .status
        .success()
}

const USES_STATUS: &str = "#include <refimage.h>\n\
    int f(void) { RefPipeline *p = 0; return ref_pipeline_new(&p) == REF_OK; }\n";
const IGNORES_STATUS: &str = "#include <refimage.h>\n\
    void f(void) { RefPipeline *p = 0; ref_pipeline_new(&p); }\n";

#[test]
fn ignored_status_is_a_compile_error() {
    let Some(cc) = compiler() else {
        eprintln!("no C compiler; skipping");
        return;
    };
    assert!(compiles(&cc, "uses_status.c", USES_STATUS, &["-std=c11"]));
    assert!(
        !compiles(&cc, "ignores_status.c", IGNORES_STATUS, &["-std=c11"]),
        "ignoring a RefStatus compiled cleanly: _MustUse is not in effect"
    );
    // C++17: [[nodiscard]] (also checks the header's extern "C" wrapper).
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    if Command::new(&cxx)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
    {
        assert!(compiles(
            &cxx,
            "uses_status.cpp",
            USES_STATUS,
            &["-std=c++17", "-x", "c++"]
        ));
        assert!(!compiles(
            &cxx,
            "ignores_status.cpp",
            IGNORES_STATUS,
            &["-std=c++17", "-x", "c++"]
        ));
    }
}

#[test]
fn null_into_nonnull_is_a_compile_error_with_clang() {
    let Some(cc) = compiler() else {
        eprintln!("no C compiler; skipping");
        return;
    };
    let version = Command::new(&cc)
        .arg("--version")
        .output()
        .expect("cc --version");
    if !String::from_utf8_lossy(&version.stdout).contains("clang") {
        eprintln!("{cc} is not Clang; _Nonnull is Clang-only, skipping");
        return;
    }
    let ok = "#include <refimage.h>\n\
        int f(const RefPipeline *p) { size_t n; return ref_pipeline_len(p, &n); }\n";
    let null = "#include <refimage.h>\n\
        int f(void) { size_t n; return ref_pipeline_len(0, &n); }\n";
    let nullable_ok = "#include <refimage.h>\n\
        void f(void) { ref_pipeline_free(0); }\n";
    assert!(compiles(
        &cc,
        "nonnull_ok.c",
        ok,
        &["-std=c11", "-Wnullability-completeness"]
    ));
    assert!(compiles(&cc, "nullable_ok.c", nullable_ok, &["-std=c11"]));
    assert!(
        !compiles(&cc, "nonnull_null.c", null, &["-std=c11"]),
        "passing NULL to a _Nonnull parameter compiled cleanly"
    );
}
