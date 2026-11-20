//! Resolving a model argument to an executable and running it.
//!
//! A model argument is a .stan`` file (compiled first, or taken from the cache), a path to a
//! model executable, or the name of a compiled model in the cache. The unified commands
//! translate their options into the executable's CmdStan arguments and run it as a child
//! process with structured arguments (no shell). The child's exit code is returned unchanged.

use std::path::{Path, PathBuf};
use std::process::Command;

use cmdstan_runner::{ExitCode, Failure};
use serde_json::Value;

use crate::cache::Cache;
use crate::compile::{self, CompileOptions, Profile};

/// A model ready to run.
#[derive(Clone, Debug)]
pub struct Resolved {
    /// Artifact name (used for default output names).
    pub name: String,
    /// The executable.
    pub executable: PathBuf,
}

/// Options for compiling a .stan`` argument on the fly.
#[derive(Clone, Debug, Default)]
pub struct OnTheFly {
    /// `#include` directories.
    pub include_paths: Vec<PathBuf>,
    /// Pass `--offline` to Cargo.
    pub offline: bool,
    /// Build the dev profile.
    pub dev: bool,
}

/// Resolves a model argument.
pub fn resolve(model: &str, cache: &Cache, fly: &OnTheFly) -> Result<Resolved, Failure> {
    let path = Path::new(model);
    if path.extension().is_some_and(|e| e == "stan") {
        let a = compile::compile(
            &CompileOptions {
                source: path.to_path_buf(),
                name: None,
                model_name: None,
                include_paths: fly.include_paths.clone(),
                allow_undefined: false,
                profile: if fly.dev {
                    Profile::Dev
                } else {
                    Profile::Release
                },
                offline: fly.offline,
                force: false,
                verbose: false,
            },
            cache,
        )?;
        return Ok(Resolved {
            name: a.name,
            executable: a.executable,
        });
    }
    // A path to an executable must look like a path (`./m`, `dir/m`, absolute). A bare name
    // always means the cache: a file named like the model in the working directory (such as
    // CmdStan's C++ executable next to `m.stan`) is never run by accident.
    let looks_like_path = path.is_absolute() || path.components().count() > 1;
    if looks_like_path {
        if !path.is_file() {
            return Err(Failure::new(
                ExitCode::NotFound,
                format!("{} is not an executable file", path.display()),
            ));
        }
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".into());
        // The absolute path: `Command` would search PATH for a bare name such as `model`,
        // which could run a different program than the file checked here.
        let executable = path
            .canonicalize()
            .map_err(|e| Failure::io("cannot resolve", path, &e))?;
        return Ok(Resolved { name, executable });
    }
    let entry = cache.find(model)?;
    // When the recorded source still exists, a name resolves exactly like the .stan path with
    // the entry's settings: the key is recomputed, so a changed source, toolchain, or runtime
    // gives a fresh build instead of a stale executable.
    let src = Path::new(entry.source());
    if src.is_file() {
        let inputs = &entry.manifest["inputs"]["stanc_options"];
        let include_paths = inputs["include_paths"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(PathBuf::from)
            .collect();
        let a = compile::compile(
            &CompileOptions {
                source: src.to_path_buf(),
                name: Some(entry.name().to_string()),
                model_name: entry.manifest["stanc_model_name"]
                    .as_str()
                    .map(str::to_string),
                include_paths,
                allow_undefined: inputs["allow_undefined"].as_bool().unwrap_or(false),
                profile: if entry.manifest["profile"] == "dev" {
                    Profile::Dev
                } else {
                    Profile::Release
                },
                offline: fly.offline,
                force: false,
                verbose: false,
            },
            cache,
        )?;
        return Ok(Resolved {
            name: a.name,
            executable: a.executable,
        });
    }
    eprintln!(
        "warning: the source {} of {model} no longer exists; running the cached build without \
         checking it against the current toolchain and runtime",
        src.display()
    );
    Ok(Resolved {
        name: entry.name().to_string(),
        executable: entry.verified_executable()?,
    })
}

/// Runs a model executable with `args` and returns its exit code. stdout and stderr are
/// inherited, so progress (stderr) and results (stdout) reach the user unchanged.
///
/// On Unix this process is replaced by the model executable (`exec`): every signal sent to
/// cmdstan-rs reaches the model, and no orphan is left when cmdstan-rs is terminated.
#[cfg(unix)]
pub fn run(exe: &Path, args: &[String]) -> Result<i32, Failure> {
    use std::os::unix::process::CommandExt;
    // `exec` returns only on failure.
    let e = Command::new(exe).args(args).exec();
    Err(Failure::new(
        ExitCode::NotFound,
        format!("cannot run {}: {e}", exe.display()),
    ))
}

/// Runs a model executable as a child process and returns its exit code.
#[cfg(not(unix))]
pub fn run(exe: &Path, args: &[String]) -> Result<i32, Failure> {
    // The child receives Ctrl+C from the terminal and stops its chains cleanly; this process
    // only waits for it.
    let _ = ctrlc::set_handler(|| {});
    let status = Command::new(exe).args(args).status().map_err(|e| {
        Failure::new(
            ExitCode::NotFound,
            format!("cannot run {}: {e}", exe.display()),
        )
    })?;
    Ok(exit_code(status))
}

/// The exit code of a finished child. A child killed by a signal exits `128 + signal`, as a
/// shell reports it: 130 for an interrupt, 139 for a segmentation fault, and so on.
fn exit_code(status: std::process::ExitStatus) -> i32 {
    if let Some(c) = status.code() {
        return c;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            return 128 + sig;
        }
    }
    ExitCode::Internal.code()
}

/// Runs a model executable and captures its stdout.
pub fn run_capture(exe: &Path, args: &[String]) -> Result<(i32, String), Failure> {
    let out = Command::new(exe)
        .args(args)
        .stderr(std::process::Stdio::inherit())
        .output()
        .map_err(|e| {
            Failure::new(
                ExitCode::NotFound,
                format!("cannot run {}: {e}", exe.display()),
            )
        })?;
    Ok((
        exit_code(out.status),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    ))
}
