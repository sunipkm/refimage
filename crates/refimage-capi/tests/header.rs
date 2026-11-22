//! `include/refimage.h` must match what cbindgen generates from the sources, plus the
//! nullability pass below.
//!
//! Regenerate it with `REFIMAGE_BLESS=1 cargo test -p refimage-capi --test header`.
//!
//! # Nullability
//!
//! The Rust functions take raw pointers so they can report `NULL` as `REF_ERR_NULL`
//! instead of it being undefined behaviour, which means cbindgen cannot tell which
//! pointers may be `NULL`. This pass marks every pointer in a function signature or
//! struct field `_Nonnull`, except these, which get `_Nullable`:
//!
//! - a parameter whose function docs say "`name` may be `NULL`";
//! - a struct field whose docs say "May be `NULL`";
//! - the argument of a one-parameter `*_free` function;
//! - `buf` and `out_len` of a string-out function (one with a `buf_len` parameter);
//! - the inner pointer of a handle out-parameter (`RefImage *_Nullable *_Nonnull
//!   out`): the handle slot is required, the handle in it may be `NULL`.
//!
//! So the docs and the annotations cannot disagree. A doc that names a parameter the
//! function does not have fails the test.

use std::collections::BTreeSet;
use std::path::Path;

const NONNULL: &str = "_Nonnull";
const NULLABLE: &str = "_Nullable";
const HANDLES: [&str; 3] = ["RefImage", "RefPipeline", "RefRunner"];

/// Names declared nullable in a doc comment: every "`name` may be `NULL`".
fn doc_nullable(doc: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut rest = doc;
    while let Some(at) = rest.find("` may be `NULL`") {
        let before = &rest[..at];
        if let Some(open) = before.rfind('`') {
            names.insert(before[open + 1..].to_owned());
        }
        rest = &rest[at + 1..];
    }
    names
}

/// `type_part` (everything before the declared name, e.g. `"const RefImage **"`)
/// with a qualifier after each `*`; `qual(i, n)` picks it for star `i` of `n`, where
/// the last star is the outermost pointer (the parameter or field itself).
fn qualify(type_part: &str, qual: impl Fn(usize, usize) -> &'static str) -> String {
    let n = type_part.matches('*').count();
    let mut out = String::new();
    let mut i = 0;
    for c in type_part.trim_end().chars() {
        out.push(c);
        if c == '*' {
            out.push_str(qual(i, n));
            out.push(' ');
            i += 1;
        }
    }
    if n == 0 {
        out.push(' ');
    }
    out
}

/// Split `"const RefImage *img"` into (`"const RefImage *"`, `"img"`).
fn split_decl(decl: &str) -> (&str, &str) {
    let decl = decl.trim();
    let start = decl
        .rfind(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .map_or(0, |i| i + 1);
    (&decl[..start], &decl[start..])
}

fn base_type(type_part: &str) -> &str {
    type_part
        .trim_end_matches(|c: char| c == '*' || c.is_whitespace())
        .trim_start_matches("const ")
        .trim()
}

fn annotate_function(line: &str, doc: &str) -> Result<String, String> {
    let open = line.find('(').ok_or("no '('")?;
    let close = line.rfind(')').ok_or("no ')'")?;
    let (ret, name) = split_decl(&line[..open]);
    let params = &line[open + 1..close];

    let mut nullable = doc_nullable(doc);
    let parts: Vec<&str> = if params == "void" {
        vec![]
    } else {
        params.split(", ").collect()
    };
    let names: Vec<&str> = parts.iter().map(|p| split_decl(p).1).collect();
    if name.ends_with("_free") && parts.len() == 1 {
        nullable.insert(names[0].to_owned());
    }
    if names.contains(&"buf_len") {
        nullable.insert("buf".into());
        nullable.insert("out_len".into());
    }

    let mut out_params = Vec::new();
    for p in &parts {
        let (ty, pname) = split_decl(p);
        let outer = if nullable.remove(pname) {
            NULLABLE
        } else {
            NONNULL
        };
        if ty.contains('*') {
            let inner = if HANDLES.contains(&base_type(ty)) {
                NULLABLE
            } else {
                NONNULL
            };
            out_params.push(format!(
                "{}{pname}",
                qualify(ty, |i, n| if i + 1 == n { outer } else { inner })
            ));
        } else if outer == NULLABLE {
            return Err(format!(
                "{name}: docs say `{pname}` may be NULL, but it is not a pointer"
            ));
        } else {
            out_params.push(p.to_string());
        }
    }
    if let Some(stray) = nullable.into_iter().next() {
        return Err(format!(
            "{name}: docs say `{stray}` may be NULL, but there is no such parameter"
        ));
    }
    let ret = if ret.contains('*') {
        qualify(ret, |_, _| NONNULL)
    } else {
        ret.to_owned()
    };
    let params = if out_params.is_empty() {
        "void".to_owned()
    } else {
        out_params.join(", ")
    };
    Ok(format!("{ret}{name}({params}){}", &line[close + 1..]))
}

fn annotate_field(line: &str, doc: &str) -> Result<String, String> {
    let indent = &line[..line.len() - line.trim_start().len()];
    let decl = line.trim().trim_end_matches(';');
    let (ty, name) = split_decl(decl);
    let nullable = doc.to_ascii_lowercase().contains("may be `null`");
    if !ty.contains('*') {
        if nullable {
            return Err(format!(
                "field {name}: docs say it may be NULL, but it is not a pointer"
            ));
        }
        return Ok(line.to_owned());
    }
    let q = if nullable { NULLABLE } else { NONNULL };
    Ok(format!("{indent}{}{name};", qualify(ty, |_, _| q)))
}

/// Add `_Nonnull` / `_Nullable` to every pointer declarator.
fn annotate(header: &str) -> Result<String, String> {
    let mut out = String::with_capacity(header.len() * 11 / 10);
    let (mut doc, mut in_doc, mut in_struct) = (String::new(), false, false);
    for line in header.lines() {
        let t = line.trim();
        let mut emitted = line.to_owned();
        if t.starts_with("/**") {
            doc.clear();
            in_doc = !t.ends_with("*/");
            doc.push_str(t.trim_start_matches("/**").trim_end_matches("*/").trim());
            doc.push(' ');
        } else if in_doc {
            if t.ends_with("*/") {
                in_doc = false;
            } else {
                doc.push_str(t.trim_start_matches('*').trim());
                doc.push(' ');
            }
        } else if t.is_empty() {
        } else {
            if t.starts_with("typedef struct") && t.ends_with('{') {
                in_struct = true;
            } else if in_struct && t.starts_with('}') {
                in_struct = false;
            } else if in_struct && t.ends_with(';') {
                emitted = annotate_field(line, &doc)?;
            } else if !line.starts_with(' ')
                && t.ends_with(");")
                && (t.contains(" ref_") || t.contains("*ref_"))
            {
                emitted = annotate_function(line, &doc)?;
            }
            doc.clear();
        }
        out.push_str(&emitted);
        out.push('\n');
    }
    Ok(out)
}

fn generate() -> String {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    // No version in the header: it would change with every release (the version comes from
    // the git tag). `ref_version()` reports it at run time; REF_ABI_VERSION is the contract.
    let config = cbindgen::Config::from_file(dir.join("cbindgen.toml")).expect("cbindgen.toml");
    let mut generated = Vec::new();
    cbindgen::Builder::new()
        .with_crate(dir)
        .with_config(config)
        .generate()
        .expect("cbindgen failed")
        .write(&mut generated);
    let generated = String::from_utf8(generated).expect("header is UTF-8");
    annotate(&generated).unwrap_or_else(|e| panic!("nullability pass: {e}"))
}

#[test]
fn header_is_up_to_date() {
    let generated = generate();
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("include/refimage.h");
    if std::env::var_os("REFIMAGE_BLESS").is_some() {
        std::fs::write(&path, &generated).expect("write include/refimage.h");
        return;
    }
    // Line endings aside: a Windows checkout may have converted the file to CRLF.
    let committed = std::fs::read_to_string(&path)
        .unwrap_or_default()
        .replace("\r\n", "\n");
    assert!(
        committed == generated,
        "include/refimage.h is stale; regenerate it with \
         `REFIMAGE_BLESS=1 cargo test -p refimage-capi --test header`"
    );
}

#[test]
fn every_pointer_is_annotated() {
    for line in generate().lines() {
        let t = line.trim();
        if t.starts_with('*') || t.starts_with("/*") || t.starts_with('#') || t.starts_with("//") {
            continue; // comments and preprocessor
        }
        let stars = t.matches('*').count();
        let quals = t.matches(NONNULL).count() + t.matches(NULLABLE).count();
        assert_eq!(stars, quals, "unannotated pointer in: {t}");
    }
}

#[test]
fn nullability_rules() {
    let f = |l: &str, d: &str| annotate_function(l, d).unwrap();
    assert_eq!(
        f(
            "_MustUse RefStatus ref_image_clone(const RefImage *img, RefImage **out);",
            ""
        ),
        "_MustUse RefStatus ref_image_clone(const RefImage *_Nonnull img, \
         RefImage *_Nullable *_Nonnull out);"
    );
    assert_eq!(
        f("void ref_image_free(RefImage *img);", ""),
        "void ref_image_free(RefImage *_Nullable img);"
    );
    assert_eq!(
        f(
            "RefStatus ref_x(const char *key, const char *comment);",
            "Set it. `comment` may be `NULL`."
        ),
        "RefStatus ref_x(const char *_Nonnull key, const char *_Nullable comment);"
    );
    assert_eq!(
        f(
            "RefStatus ref_s(char *buf, size_t buf_len, size_t *out_len);",
            ""
        ),
        "RefStatus ref_s(char *_Nullable buf, size_t buf_len, size_t *_Nullable out_len);"
    );
    assert_eq!(
        f(
            "RefStatus ref_d(RefImage *img, void **data, size_t *nbytes);",
            "`nbytes` may be `NULL`."
        ),
        "RefStatus ref_d(RefImage *_Nonnull img, void *_Nonnull *_Nonnull data, \
         size_t *_Nullable nbytes);"
    );
    assert_eq!(
        f("const char *ref_version(void);", ""),
        "const char *_Nonnull ref_version(void);"
    );
    assert!(annotate_function("RefStatus ref_y(int a);", "`b` may be `NULL`.").is_err());
    assert_eq!(
        annotate_field(
            "  const void *data;",
            "First row. May be `NULL` when unused."
        )
        .unwrap(),
        "  const void *_Nullable data;"
    );
    assert_eq!(
        annotate_field("  size_t width;", "Width.").unwrap(),
        "  size_t width;"
    );
}
