//! Compilation of Stan programs into model executables.
//!
//! 1. Read the source; check it with stanc-rs (diagnostics, `#include` files, `--info`).
//! 2. Compute the cache key from every material input (source, included files, compiler
//!    options, compiler and library revisions, the runtime crates' sources, the Rust toolchain,
//!    the target, the profile, and the Rust flags).
//! 3. Under the entry's lock: reuse a complete entry with that key, or generate the Rust module
//!    with stanc-rs, write a Cargo project around it, and build it with Cargo.
//! 4. Check that the executable exists, record its hash, and write the entry manifest last.
//!
//! Compilation never parses Stan itself and never builds anything but the generated project.
//! Building runs the local Rust toolchain, so a Stan program from an untrusted source should be
//! compiled and run in an isolated environment: cmdstan-rs provides no sandbox.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use cmdstan_runner::output::{hex, sha256_file};
use cmdstan_runner::{ExitCode, Failure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::cache::{Cache, Entry, read_manifest};

/// The stanc-rs revision of the compiler, the runtime, and the adapter (see Cargo.toml).
pub const STANC_RS_REV: &str = "242398e7bbd1754a514f221d31078254afb108f8";
/// The stan-model-rs revision (runtime and stanc-rs adapter).
pub const STAN_MODEL_RS_REV: &str = "685c03b39b1cd1477f8259f7508c69d5fcda2f48";
/// The stanmath-rs revision that stanc-rs and stan-model-rs pin.
pub const STANMATH_RS_REV: &str = "04dc87184161bf29cbc035a32eb0407b14169900";
/// The CmdStan release whose argument grammar and output layout are the reference.
pub const CMDSTAN_REFERENCE: &str = "2.40.0";

/// Cargo build profile of model executables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    /// Optimized (`cargo build --release`).
    Release,
    /// Unoptimized with debug assertions; much slower sampling.
    Dev,
}

impl Profile {
    fn name(self) -> &'static str {
        match self {
            Profile::Release => "release",
            Profile::Dev => "dev",
        }
    }
}

/// Compilation settings.
#[derive(Clone, Debug)]
pub struct CompileOptions {
    /// The Stan program.
    pub source: PathBuf,
    /// Artifact name (default: the file stem).
    pub name: Option<String>,
    /// stanc model name (default: `<stem>_model`, as stanc3).
    pub model_name: Option<String>,
    /// `#include` search directories.
    pub include_paths: Vec<PathBuf>,
    /// Accept functions declared without a definition (they cannot be built: stanc-rs has no
    /// mechanism to link user code yet).
    pub allow_undefined: bool,
    /// Build profile.
    pub profile: Profile,
    /// Pass `--offline` to Cargo (no network access; dependencies must already be fetched).
    pub offline: bool,
    /// Rebuild even if a complete entry exists.
    pub force: bool,
    /// Show Cargo's output.
    pub verbose: bool,
}

/// A built model.
#[derive(Clone, Debug)]
pub struct Artifact {
    /// Artifact name.
    pub name: String,
    /// The model executable inside the cache.
    pub executable: PathBuf,
    /// The cache entry directory.
    pub entry_dir: PathBuf,
    /// True if the entry existed and nothing was built.
    pub cached: bool,
    /// The entry manifest.
    pub manifest: Value,
}

/// The cmdstan-rs source tree that holds the runner crate generated projects depend on:
/// `$CMDSTAN_RS_HOME`, else the workspace this binary was built from.
pub fn runtime_root() -> Result<PathBuf, Failure> {
    // ponytail: generated projects use the runner crate by path, so an installed binary still
    // needs the source checkout; publish cmdstan-runner (or vendor it into the cache) to lift
    // this.
    let root = match std::env::var_os("CMDSTAN_RS_HOME") {
        Some(d) if !d.is_empty() => PathBuf::from(d),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."),
    };
    let runner = root.join("crates/cmdstan-runner/Cargo.toml");
    if !runner.exists() {
        return Err(Failure::usage(format!(
            "the cmdstan-rs source tree was not found at {} (no crates/cmdstan-runner); set \
             CMDSTAN_RS_HOME to the cmdstan-rs checkout",
            root.display()
        )));
    }
    let root = absolute_path(&root)?;
    // The code generator is compiled into this binary; the runtime is built from this tree.
    // They must be the same revisions.
    let lock = root.join("Cargo.lock");
    let text = std::fs::read_to_string(&lock).map_err(|e| Failure::io("cannot read", &lock, &e))?;
    for (what, rev) in [
        ("stanc-rs", STANC_RS_REV),
        ("stan-model-rs", STAN_MODEL_RS_REV),
    ] {
        if !text.contains(rev) {
            return Err(Failure::usage(format!(
                "the cmdstan-rs source tree at {} pins another {what} revision than this \
                 cmdstan-rs binary ({rev}); rebuild cmdstan-rs from that tree or point \
                 CMDSTAN_RS_HOME at the matching checkout",
                root.display()
            )));
        }
    }
    Ok(root)
}

/// An absolute path without `..`: canonical (symbolic links resolved) on Unix; on Windows
/// `std::path::absolute`, which avoids the `\\?\` verbatim prefix that Cargo and linkers
/// handle poorly.
pub fn absolute_path(p: &Path) -> Result<PathBuf, Failure> {
    let r = if cfg!(windows) {
        std::path::absolute(p).and_then(|a| a.metadata().map(|_| a))
    } else {
        p.canonicalize()
    };
    r.map_err(|e| {
        let code = if e.kind() == std::io::ErrorKind::NotFound {
            ExitCode::NotFound
        } else {
            ExitCode::Usage
        };
        Failure::new(code, format!("{}: {e}", p.display()))
    })
}

/// The Rust toolchain: `rustc -vV`.
///
/// Run in `dir`: the cache, where Cargo builds, so a `rust-toolchain.toml` near the caller's
/// working directory does not select a different compiler than the one that builds.
pub fn rustc_version(dir: &Path) -> Result<String, Failure> {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let out = Command::new(&rustc)
        .arg("-vV")
        .current_dir(dir)
        .output()
        .map_err(|e| {
            Failure::new(
                ExitCode::Build,
                format!(
                    "cannot run {} ({e}); install Rust from https://rustup.rs",
                    Path::new(&rustc).display()
                ),
            )
        })?;
    if !out.status.success() {
        return Err(Failure::new(ExitCode::Build, "rustc -vV failed"));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The host target triple from `rustc -vV`.
pub fn host_triple(rustc_vv: &str) -> String {
    rustc_vv
        .lines()
        .find_map(|l| l.strip_prefix("host: "))
        .unwrap_or("unknown")
        .trim()
        .to_string()
}

/// A Cargo-safe name: ASCII letters, digits, and `_`, not starting with a digit.
pub fn sanitize(name: &str) -> String {
    let mut s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    // A leading digit is not a valid crate name; Windows reserves device names.
    const RESERVED: [&str; 22] = [
        "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
        "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
    ];
    if s.is_empty() || s.starts_with(|c: char| c.is_ascii_digit()) || RESERVED.contains(&s.as_str())
    {
        s.insert(0, 'm');
    }
    s
}

fn stanc_options(o: &CompileOptions) -> stanc_rs::Options {
    stanc_rs::Options {
        model_name: o.model_name.clone(),
        include_paths: o.include_paths.clone(),
        allow_undefined: o.allow_undefined,
    }
}

/// The result of checking a program.
pub struct Checked {
    /// Source text.
    pub text: String,
    /// stanc-rs warnings.
    pub warnings: Vec<String>,
    /// `--info` JSON.
    pub info: String,
    /// Included files with their contents' hashes.
    pub included: Vec<(String, String)>,
}

/// Reads and checks a program with stanc-rs.
pub fn check(o: &CompileOptions) -> Result<Checked, Failure> {
    let text = std::fs::read_to_string(&o.source).map_err(|e| {
        let code = if e.kind() == std::io::ErrorKind::NotFound {
            ExitCode::NotFound
        } else {
            ExitCode::Usage
        };
        Failure::new(code, format!("model source {}: {e}", o.source.display()))
    })?;
    let name = o.source.to_string_lossy().into_owned();
    let checked = stanc_rs::check_source(&name, &text, &stanc_options(o))
        .map_err(|e| Failure::new(ExitCode::Compile, e))?;
    let mut included = Vec::new();
    for f in &checked.included_files {
        let h = sha256_file(Path::new(f))
            .map_err(|e| Failure::io("cannot read included file", Path::new(f), &e))?;
        included.push((f.clone(), h));
    }
    Ok(Checked {
        info: checked.info(),
        warnings: checked.warnings,
        included,
        text,
    })
}

/// Hashes the sources of a crate directory (`Cargo.toml` and everything under `src/`), in a
/// fixed order.
fn hash_crate(h: &mut Sha256, dir: &Path) -> Result<(), Failure> {
    fn walk(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
        for e in std::fs::read_dir(dir)? {
            let p = e?.path();
            if p.is_dir() {
                walk(&p, files)?;
            } else {
                files.push(p);
            }
        }
        Ok(())
    }
    let mut files = vec![dir.join("Cargo.toml")];
    let src = dir.join("src");
    walk(&src, &mut files).map_err(|e| Failure::io("cannot read the runtime sources", &src, &e))?;
    files.sort();
    // An unreadable source would make two different builds share a key: fail instead.
    for f in files {
        let bytes = std::fs::read(&f).map_err(|e| Failure::io("cannot read", &f, &e))?;
        let rel = f.strip_prefix(dir).unwrap_or(&f);
        h.update(rel.to_string_lossy().as_bytes());
        h.update([0]);
        h.update(&bytes);
        h.update([0]);
    }
    Ok(())
}

/// The crates generated projects build from local paths: the runner and the unpublished
/// sibling workspaces it uses. Their sources are part of the cache key because Cargo cannot
/// pin a path dependency.
fn path_dependency_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs = vec![root.join("crates/cmdstan-runner")];
    let parent = root.join("..");
    for d in [
        "stan-services-rs/crates/stan-services",
        "stan-inference-rs/crates/stan-inference",
        "stan-inference-rs/crates/stan-inference-diagnostics",
        "stan-inference-rs/crates/stan-inference-stan-model",
    ] {
        dirs.push(parent.join(d));
    }
    dirs
}

/// The `main.rs` of every generated project. Part of the cache key: a change to it changes
/// every executable.
const MAIN_RS: &str = "// Generated by cmdstan-rs. Do not edit.\n\
#[allow(clippy::all, non_snake_case, non_camel_case_types, unused, dead_code)]\n\
mod model {\n\
\x20   #[allow(unused_imports)]\n\
\x20   use cmdstan_runner::stanc_rs_runtime;\n\
\x20   include!(\"model.rs\");\n\
}\n\n\
fn main() {\n\
\x20   cmdstan_runner::run(\n\
\x20       cmdstan_runner::BuildInfo {\n\
\x20           build_json: include_str!(\"../build_info.json\"),\n\
\x20           stanc_info_json: include_str!(\"../stanc_info.json\"),\n\
\x20       },\n\
\x20       model::Model::new,\n\
\x20   )\n\
}\n";

/// Version of the generated `Cargo.toml` layout (part of the cache key).
const CARGO_TOML_LAYOUT: u32 = 1;

/// Whether an environment variable can change what Cargo or rustc builds. Every such variable
/// is part of the cache key. Variables that Cargo sets for the process that runs cmdstan-rs
/// (under `cargo run` or `cargo test`), locations, and output settings are not.
fn is_build_env(k: &str) -> bool {
    const IGNORED: [&str; 12] = [
        "CARGO_PKG_",
        "CARGO_MANIFEST_",
        "CARGO_BIN_",
        "CARGO_CRATE_",
        "CARGO_TARGET_TMPDIR",
        "CARGO_PRIMARY_PACKAGE",
        "CARGO_HOME",
        "CARGO_TERM_",
        "CARGO_MAKEFLAGS",
        "CARGO_TARGET_DIR",
        "RUSTUP_HOME",
        "RUST_LOG",
    ];
    (k.starts_with("CARGO_") || k.starts_with("RUST"))
        && !IGNORED.iter().any(|p| k.starts_with(p))
        && !matches!(
            k,
            "CARGO" | "RUST_BACKTRACE" | "RUST_RECURSION_COUNT" | "RUST_MIN_STACK"
        )
}

/// The Cargo configuration files that apply to a build run inside the cache: those in
/// `$CARGO_HOME` (the build runs with the generated project as its working directory, so a
/// caller's project configuration does not apply).
fn cargo_home_config() -> Vec<(String, String)> {
    let home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cargo")))
        .or_else(|| std::env::var_os("USERPROFILE").map(|h| PathBuf::from(h).join(".cargo")));
    let Some(home) = home else {
        return Vec::new();
    };
    ["config.toml", "config"]
        .iter()
        .map(|f| home.join(f))
        .filter_map(|f| {
            std::fs::read(&f)
                .ok()
                .map(|b| (f.display().to_string(), hex(&Sha256::digest(&b))))
        })
        .collect()
}

/// `rustc --print cfg` with the Rust flags in effect: the target features it reports include
/// those `-C target-cpu=native` selects on this machine.
/// Only flags from the environment are seen (Cargo's precedence: `CARGO_ENCODED_RUSTFLAGS`,
/// `RUSTFLAGS`, `CARGO_BUILD_RUSTFLAGS`); `target-cpu=native` set in a Cargo configuration file
/// is not reflected.
fn rustc_cfg(dir: &Path) -> Result<String, Failure> {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let var = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let flags: Vec<String> = if let Some(f) = var("CARGO_ENCODED_RUSTFLAGS") {
        f.split('\x1f').map(str::to_string).collect()
    } else {
        var("RUSTFLAGS")
            .or_else(|| var("CARGO_BUILD_RUSTFLAGS"))
            .unwrap_or_default()
            .split_whitespace()
            .map(str::to_string)
            .collect()
    };
    let out = Command::new(&rustc)
        .arg("--print")
        .arg("cfg")
        .args(flags)
        .current_dir(dir)
        .output()
        .map_err(|e| {
            Failure::new(
                ExitCode::Build,
                format!("cannot run rustc --print cfg: {e}"),
            )
        })?;
    if !out.status.success() {
        return Err(Failure::new(
            ExitCode::Build,
            format!(
                "rustc --print cfg failed with the Rust flags in effect: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ),
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Whether `s` is a Stan identifier (letters, digits, and underscores, starting with a letter).
fn is_stan_identifier(s: &str) -> bool {
    s.starts_with(|c: char| c.is_ascii_alphabetic())
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Every material input of a build, as JSON (recorded in the manifest), and its SHA-256.
fn cache_key(
    o: &CompileOptions,
    dir: &Path,
    name: &str,
    checked: &Checked,
    stanc_model_name: &str,
    root: &Path,
    rustc_vv: &str,
) -> Result<(Value, String), Failure> {
    let mut deps = Sha256::new();
    for d in path_dependency_dirs(root) {
        hash_crate(&mut deps, &d)?;
    }
    // The workspace manifests of the sibling path dependencies set dependency versions, lints,
    // and features.
    let parent = root.join("..");
    for w in [
        "stan-services-rs/Cargo.toml",
        "stan-inference-rs/Cargo.toml",
    ] {
        let f = parent.join(w);
        let bytes = std::fs::read(&f).map_err(|e| Failure::io("cannot read", &f, &e))?;
        deps.update(w.as_bytes());
        deps.update([0]);
        deps.update(&bytes);
    }
    let lock = root.join("Cargo.lock");
    let lock_hash = sha256_file(&lock).map_err(|e| Failure::io("cannot read", &lock, &e))?;
    let mut env: Vec<(String, String)> = std::env::vars_os()
        .filter_map(|(k, v)| {
            let k = k.into_string().ok()?;
            // Only a hash of each value: variables such as registry tokens must never reach a
            // manifest, which is embedded in every executable.
            is_build_env(&k).then(|| (k, hex(&Sha256::digest(v.as_encoded_bytes()))))
        })
        .collect();
    env.sort();
    let inputs = json!({
        "cmdstan_rs": env!("CARGO_PKG_VERSION"),
        // The artifact name and the source path are recorded in the executable.
        "name": name,
        "source_path": o.source.display().to_string(),
        "main_rs_sha256": hex(&Sha256::digest(MAIN_RS.as_bytes())),
        "cargo_toml_layout": CARGO_TOML_LAYOUT,
        "source_sha256": hex(&Sha256::digest(checked.text.as_bytes())),
        "included": checked.included.iter().map(|(p, h)| json!({"path": p, "sha256": h})).collect::<Vec<_>>(),
        "stanc_options": {
            "model_name": stanc_model_name,
            "include_paths": o.include_paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
            "allow_undefined": o.allow_undefined,
        },
        "stanc_rs": STANC_RS_REV,
        "stan_model_rs": STAN_MODEL_RS_REV,
        "stanmath_rs": STANMATH_RS_REV,
        "path_dependencies_sha256": hex(&deps.finalize()),
        "cargo_lock_sha256": lock_hash,
        "rustc": rustc_vv,
        "profile": o.profile.name(),
        "build_env": env,
        "cargo_home_config": cargo_home_config(),
        "rustc_cfg": rustc_cfg(dir)?,
        "features": [],
    });
    let key = hex(&Sha256::digest(inputs.to_string().as_bytes()));
    Ok((inputs, key))
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// A TOML basic string (JSON string escapes are valid TOML escapes).
fn toml_str(s: &str) -> String {
    serde_json::to_string(s).expect("strings serialize")
}

/// Runs `cargo build` on the generated project and returns the path of the built executable,
/// as Cargo reports it (so a configured `build.target` or target directory is honored).
fn cargo_build(
    o: &CompileOptions,
    cache: &Cache,
    project: &Path,
    pkg: &str,
) -> Result<PathBuf, Failure> {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut cmd = Command::new(&cargo);
    cmd.arg("build")
        .arg("--manifest-path")
        .arg(project.join("Cargo.toml"))
        .arg("--message-format=json-render-diagnostics")
        .env("CARGO_TARGET_DIR", cache.target_dir())
        // Only the cache key's configuration applies, not a `.cargo/config.toml` of the
        // directory cmdstan-rs was started from.
        .current_dir(project)
        .stdout(Stdio::piped());
    if o.profile == Profile::Release {
        cmd.arg("--release");
    }
    if o.offline {
        cmd.arg("--offline");
    }
    if o.verbose {
        cmd.stderr(Stdio::inherit());
    }
    let out = cmd
        .output()
        .map_err(|e| Failure::new(ExitCode::Build, format!("cannot run cargo: {e}")))?;
    let log = String::from_utf8_lossy(&out.stderr).into_owned();
    if !out.status.success() {
        let log_path = project.join("build.log");
        let note = if o.verbose {
            "(the build output was shown on the terminal)\n"
        } else {
            ""
        };
        let _ = std::fs::write(&log_path, format!("{note}{log}"));
        // The last lines carry the error; the full log stays next to the project.
        let tail: Vec<&str> = log.lines().rev().take(60).collect();
        let tail: Vec<&str> = tail.into_iter().rev().collect();
        return Err(Failure::new(
            ExitCode::Build,
            format!(
                "cargo build failed for the generated project {} (log: {}):\n{}",
                project.display(),
                log_path.display(),
                tail.join("\n")
            ),
        ));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let built = stdout
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|m| m["reason"] == "compiler-artifact" && m["target"]["name"] == pkg)
        .find_map(|m| m["executable"].as_str().map(PathBuf::from));
    match built {
        Some(p) if p.is_file() => Ok(p),
        _ => Err(Failure::new(
            ExitCode::Build,
            format!("cargo reported success but no executable for {pkg}"),
        )),
    }
}

/// Compiles (or reuses) a model.
pub fn compile(o: &CompileOptions, cache: &Cache) -> Result<Artifact, Failure> {
    // One identity per source file, whatever directory it was compiled from.
    let source = absolute_path(&o.source)
        .map_err(|f| Failure::new(f.code, format!("model source {}", f.message)))?;
    // Include directories are recorded absolute, so a run by name resolves them from anywhere.
    let include_paths = o
        .include_paths
        .iter()
        .map(|p| {
            absolute_path(p).map_err(|f| Failure::usage(format!("include path {}", f.message)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let o = &CompileOptions {
        source,
        include_paths,
        ..o.clone()
    };
    if let Some(m) = &o.model_name
        && !is_stan_identifier(m)
    {
        return Err(Failure::usage(format!(
            "--model-name {m:?} is not a Stan identifier (letters, digits, and underscores, \
             starting with a letter)"
        )));
    }
    let checked = check(o)?;
    for w in &checked.warnings {
        eprintln!("{w}");
    }
    let stem = o
        .source
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    let name = o.name.clone().unwrap_or(stem);
    if name.is_empty() || name.contains(['/', '\\']) || name == "." || name == ".." {
        return Err(Failure::usage(format!("invalid model name {name:?}")));
    }
    let stanc_model_name = o
        .model_name
        .clone()
        .unwrap_or_else(|| stanc_rs::default_model_name(&o.source.to_string_lossy()));
    let root = runtime_root()?;
    cache.ensure()?;
    let rustc_vv = rustc_version(&cache.root)?;
    let (inputs, key) = cache_key(
        o,
        &cache.root,
        &name,
        &checked,
        &stanc_model_name,
        &root,
        &rustc_vv,
    )?;
    // Builds share the cache; `cache clear` waits for them.
    let _cache_guard = cache.lock(false)?;
    let entry_name = format!("{}-{}", sanitize(&name), &key[..16]);
    let entry_dir = cache.models_dir().join(&entry_name);

    // One build per entry at a time; concurrent compilations of the same model wait here, then
    // find the finished entry.
    let lock_path = cache.locks_dir().join(format!("{entry_name}.lock"));
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)
        .map_err(|e| Failure::io("cannot open the cache lock", &lock_path, &e))?;
    lock.lock()
        .map_err(|e| Failure::io("cannot lock", &lock_path, &e))?;

    if !o.force
        && let Some(m) = read_manifest(&entry_dir)
    {
        let entry = Entry {
            dir: entry_dir.clone(),
            manifest: m,
        };
        if entry.manifest["status"] == "success"
            && entry.key() == key
            && let Ok(exe) = entry.verified_executable()
        {
            return Ok(Artifact {
                name,
                executable: exe,
                entry_dir,
                cached: true,
                manifest: entry.manifest,
            });
        }
        eprintln!(
            "warning: the cache entry {} is incomplete or modified; rebuilding",
            entry_dir.display()
        );
    }

    // Build in a staging directory; the entry is replaced only after a successful build, so a
    // failed rebuild never destroys a working executable.
    // The entry lock serializes builds of this entry, so the staging name needs no process id
    // (and an interrupted build's leftovers are reused, not accumulated).
    let staging = cache.models_dir().join(format!(".staging-{entry_name}"));
    if staging.exists() {
        std::fs::remove_dir_all(&staging).map_err(|e| Failure::io("cannot clear", &staging, &e))?;
    }
    let started = Instant::now();
    let rust = stanc_rs::compile_source(
        &o.source.to_string_lossy(),
        &checked.text,
        &stanc_options(o),
    )
    .map_err(|e| Failure::new(ExitCode::Compile, e))?;
    let codegen_seconds = started.elapsed().as_secs_f64();
    // Included files are read again by the code generator: refuse to publish a build whose
    // includes changed after they were hashed into the key.
    for (f, h) in &checked.included {
        if sha256_file(Path::new(f)).ok().as_deref() != Some(h.as_str()) {
            return Err(Failure::usage(format!(
                "the included file {f} changed during compilation; compile again"
            )));
        }
    }

    let pkg = format!("cmdstan_model_{}_{}", sanitize(&name), &key[..12]);
    let src = staging.join("src");
    let io = |p: &Path, e: std::io::Error| Failure::io("cannot write", p, &e);
    std::fs::create_dir_all(&src).map_err(|e| io(&src, e))?;
    let write = |p: PathBuf, s: &str| std::fs::write(&p, s).map_err(|e| io(&p, e));
    write(src.join("model.rs"), &rust.rust)?;
    write(src.join("main.rs"), MAIN_RS)?;
    let runner = root.join("crates/cmdstan-runner");
    let opt = match o.profile {
        Profile::Release => "[profile.release]\nopt-level = 3\ndebug = false\n",
        Profile::Dev => "",
    };
    write(
        staging.join("Cargo.toml"),
        &format!(
            "# Generated by cmdstan-rs. Do not edit.\n\
             [package]\n\
             name = \"{pkg}\"\n\
             version = \"0.1.0\"\n\
             edition = \"2024\"\n\
             publish = false\n\n\
             [[bin]]\n\
             name = \"{pkg}\"\n\
             path = \"src/main.rs\"\n\n\
             [dependencies]\n\
             cmdstan-runner = {{ path = {} }}\n\n\
             # A standalone project, never part of an enclosing workspace.\n\
             [workspace]\n\n\
             {opt}",
            toml_str(&runner.to_string_lossy())
        ),
    )?;
    // The workspace lock pins every registry and git dependency to the versions cmdstan-rs
    // was developed with; Cargo adds the model package and drops what it does not need. The
    // resulting lock is recorded in the manifest.
    let lock_src = root.join("Cargo.lock");
    std::fs::copy(&lock_src, staging.join("Cargo.lock"))
        .map_err(|e| Failure::io("cannot copy", &lock_src, &e))?;
    write(staging.join("model.stan"), &checked.text)?;
    write(staging.join("stanc_info.json"), &checked.info)?;
    let target = std::env::var("CARGO_BUILD_TARGET")
        .ok()
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| host_triple(&rustc_vv));
    let build_info = json!({
        "name": name,
        "stanc_model_name": stanc_model_name,
        "source": {
            "path": o.source.display().to_string(),
            "sha256": inputs["source_sha256"],
            "included": inputs["included"],
        },
        "cache_key": key,
        "inputs": inputs,
        "compiler": format!("stanc-rs 0.1.0 ({STANC_RS_REV})"),
        "runtime": format!("stan-model-rs 0.1.0 ({STAN_MODEL_RS_REV}) with stanc-rs-runtime ({STANC_RS_REV})"),
        "math": format!("stanmath-rs ({STANMATH_RS_REV})"),
        "inference": "stan-inference 0.1.0 (path dependency, see inputs.path_dependencies_sha256)",
        "services": "stan-services 0.1.0 (path dependency, see inputs.path_dependencies_sha256)",
        "cmdstan_rs": env!("CARGO_PKG_VERSION"),
        "cmdstan_reference": CMDSTAN_REFERENCE,
        "target": target,
        "profile": o.profile.name(),
        "rustc": rustc_vv.lines().next().unwrap_or(""),
    });
    write(
        staging.join("build_info.json"),
        &serde_json::to_string_pretty(&build_info).expect("serializes"),
    )?;

    eprintln!(
        "building {name} ({}; the first build compiles the math library and takes minutes)",
        o.profile.name()
    );
    let build_started = Instant::now();
    let built = match cargo_build(o, cache, &staging, &pkg) {
        Ok(b) => b,
        Err(f) => {
            // Keep the failed project and its log where `cache` commands can find it.
            let failed = cache.models_dir().join(format!(".failed-{entry_name}"));
            let _ = std::fs::remove_dir_all(&failed);
            let msg = if std::fs::rename(&staging, &failed).is_ok() {
                f.message.replace(
                    &staging.display().to_string(),
                    &failed.display().to_string(),
                )
            } else {
                f.message
            };
            return Err(Failure::new(f.code, msg));
        }
    };
    let build_seconds = build_started.elapsed().as_secs_f64();
    let bin_dir = staging.join("bin");
    std::fs::create_dir_all(&bin_dir).map_err(|e| io(&bin_dir, e))?;
    let exe_file = format!("{}{}", sanitize(&name), std::env::consts::EXE_SUFFIX);
    let staged_exe = bin_dir.join(&exe_file);
    std::fs::copy(&built, &staged_exe).map_err(|e| Failure::io("cannot copy", &built, &e))?;
    let exe_hash =
        sha256_file(&staged_exe).map_err(|e| Failure::io("cannot read", &staged_exe, &e))?;
    let generated_lock = staging.join("Cargo.lock");
    let mut manifest = build_info;
    manifest["status"] = json!("success");
    manifest["artifact"] = json!({"file": exe_file, "sha256": exe_hash});
    manifest["generated_cargo_lock_sha256"] = json!(sha256_file(&generated_lock).ok());
    manifest["built_unix"] = json!(unix_now());
    manifest["timing"] =
        json!({"codegen_seconds": codegen_seconds, "cargo_seconds": build_seconds});
    manifest["stanc_warnings"] = json!(checked.warnings);
    // The manifest is written last: an entry without one is incomplete.
    write(
        staging.join("manifest.json"),
        &serde_json::to_string_pretty(&manifest).expect("serializes"),
    )?;
    // Replace: move the old entry aside, move the new one in, then delete the old one.
    let old = cache.models_dir().join(format!(".replaced-{entry_name}"));
    let _ = std::fs::remove_dir_all(&old);
    let had_old = entry_dir.exists();
    if had_old {
        std::fs::rename(&entry_dir, &old)
            .map_err(|e| Failure::io("cannot replace the cache entry", &entry_dir, &e))?;
    }
    if let Err(e) = std::fs::rename(&staging, &entry_dir) {
        if had_old {
            let _ = std::fs::rename(&old, &entry_dir);
        }
        return Err(io(&entry_dir, e));
    }
    if had_old {
        let _ = std::fs::remove_dir_all(&old);
    }
    drop(lock);
    Ok(Artifact {
        name,
        executable: entry_dir.join("bin").join(&exe_file),
        entry_dir,
        cached: false,
        manifest,
    })
}

/// Copies a built executable to `dest` (a standalone model executable).
pub fn export(a: &Artifact, dest: &Path, overwrite: bool) -> Result<PathBuf, Failure> {
    let mut dest = dest.to_path_buf();
    if dest.is_dir() {
        dest = dest.join(format!(
            "{}{}",
            sanitize(&a.name),
            std::env::consts::EXE_SUFFIX
        ));
    } else if !std::env::consts::EXE_SUFFIX.is_empty()
        && !dest
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case(std::env::consts::EXE_EXTENSION))
    {
        // Windows runs only `.exe` files: append the suffix (also to dotted names such as
        // `model.v1`).
        let mut s = dest.into_os_string();
        s.push(std::env::consts::EXE_SUFFIX);
        dest = PathBuf::from(s);
    }
    // Written to a temporary file and published without replacing an existing file unless
    // `overwrite` (no window in which another process's file is clobbered).
    let mut src = std::fs::File::open(&a.executable)
        .map_err(|e| Failure::io("cannot read", &a.executable, &e))?;
    cmdstan_runner::output::write_file(&dest, overwrite, |w| {
        std::io::copy(&mut src, w).map(|_| ())
    })?;
    let perms = std::fs::metadata(&a.executable)
        .map_err(|e| Failure::io("cannot read", &a.executable, &e))?
        .permissions();
    std::fs::set_permissions(&dest, perms).map_err(|e| Failure::io("cannot write", &dest, &e))?;
    Ok(dest)
}
