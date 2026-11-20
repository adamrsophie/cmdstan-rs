//! The model build cache.
//!
//! Layout of a cache root:
//!
//! ```text
//! `<root>/`
//! ├── .cmdstan-rs-cache        marker: only a directory with this file is ever cleared
//! ├── target/                  shared Cargo target directory (the math library builds once)
//! ├── `locks/<entry>.lock`       one lock file per entry, held while the entry is built
//! └── `models/<name>-<key16>/`   one entry per distinct set of compilation inputs
//!     ├── Cargo.toml, Cargo.lock, src/main.rs, src/model.rs
//!     ├── model.stan           copy of the compiled source
//!     ├── build_info.json      provenance embedded in the executable
//!     ├── stanc_info.json      stanc-rs --info output
//!     ├── `bin/<name>[`.exe]     the model executable
//!     └── manifest.json        written last; its presence marks a complete entry
//! ```
//!
//! The entry key is a SHA-256 over every input that changes the executable (see
//! `compile::cache_key`), never the file name alone.

use std::path::{Path, PathBuf};

use std::fs::File;

use cmdstan_runner::output::sha256_file;
use cmdstan_runner::{ExitCode, Failure};
use serde_json::Value;

const MARKER: &str = ".cmdstan-rs-cache";

/// Fails unless `p` is a real directory (not a symbolic link) or does not exist.
fn no_symlink(p: &Path) -> Result<(), Failure> {
    match std::fs::symlink_metadata(p) {
        Ok(m) if m.file_type().is_symlink() || !m.is_dir() => Err(Failure::usage(format!(
            "{} is not a plain directory (a symbolic link or a file); refusing to use it as \
             part of the cache",
            p.display()
        ))),
        _ => Ok(()),
    }
}

/// A cache root.
#[derive(Clone, Debug)]
pub struct Cache {
    /// The root directory.
    pub root: PathBuf,
}

/// A complete cache entry.
#[derive(Clone, Debug)]
pub struct Entry {
    /// Entry directory.
    pub dir: PathBuf,
    /// The entry's manifest.
    pub manifest: Value,
}

impl Entry {
    /// Model name (the artifact name, not stanc's `<name>_model`).
    pub fn name(&self) -> &str {
        self.manifest["name"].as_str().unwrap_or("")
    }

    /// The cache key.
    pub fn key(&self) -> &str {
        self.manifest["cache_key"].as_str().unwrap_or("")
    }

    /// The model executable: always `bin/<file>` inside the entry, whatever path the manifest
    /// records, so a manifest cannot point outside the cache.
    pub fn executable(&self) -> PathBuf {
        let recorded = Path::new(self.manifest["artifact"]["file"].as_str().unwrap_or(""));
        let file = recorded.file_name().unwrap_or_default();
        self.dir.join("bin").join(file)
    }

    /// The executable, after checking that its SHA-256 matches the manifest.
    pub fn verified_executable(&self) -> Result<PathBuf, Failure> {
        let exe = self.executable();
        let want = self.manifest["artifact"]["sha256"].as_str();
        let have = sha256_file(&exe).ok();
        match (want, have.as_deref()) {
            (Some(w), Some(h)) if w == h => Ok(exe),
            _ => Err(Failure::new(
                ExitCode::NotFound,
                format!(
                    "the cache entry {} has no intact executable (missing, or its hash differs \
                     from the manifest); recompile the model",
                    self.dir.display()
                ),
            )),
        }
    }

    /// Source path recorded at build time.
    pub fn source(&self) -> &str {
        self.manifest["source"]["path"].as_str().unwrap_or("")
    }

    /// Build completion time (seconds since the Unix epoch).
    pub fn built(&self) -> f64 {
        self.manifest["built_unix"].as_f64().unwrap_or(0.0)
    }
}

impl Cache {
    /// The cache at `dir`, else `$CMDSTAN_RS_CACHE`, else the platform cache directory.
    pub fn locate(dir: Option<&Path>) -> Result<Self, Failure> {
        let root = match dir {
            Some(d) => d.to_path_buf(),
            None => match std::env::var_os("CMDSTAN_RS_CACHE") {
                Some(d) if !d.is_empty() => PathBuf::from(d),
                _ => default_root().ok_or_else(|| {
                    Failure::usage("no cache directory: set CMDSTAN_RS_CACHE or pass --cache-dir")
                })?,
            },
        };
        Ok(Cache { root })
    }

    /// Creates the root and its marker if needed. A nonempty directory without the marker is
    /// refused, so that a mistyped `--cache-dir` never turns a user directory into a cache.
    pub fn ensure(&self) -> Result<(), Failure> {
        let marker = self.root.join(MARKER);
        if !marker.exists() {
            if self.root.exists() {
                let nonempty = std::fs::read_dir(&self.root)
                    .map_err(|e| Failure::io("cannot read the cache directory", &self.root, &e))?
                    .next()
                    .is_some();
                // The marker may have appeared while the directory was read.
                if nonempty && !marker.exists() {
                    return Err(Failure::usage(format!(
                        "{} is not empty and is not a cmdstan-rs cache (no {MARKER} file); \
                         choose another --cache-dir",
                        self.root.display()
                    )));
                }
            }
            std::fs::create_dir_all(&self.root)
                .map_err(|e| Failure::io("cannot create the cache directory", &self.root, &e))?;
            // The marker is created first and exclusively, so a concurrent first use sees it
            // (and the subdirectories below are created by whichever process gets there).
            match File::create_new(&marker) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(Failure::io("cannot write", &marker, &e)),
            }
            let _ = std::fs::write(&marker, "cmdstan-rs build cache\n");
        }
        for d in [self.models_dir(), self.locks_dir(), self.target_dir()] {
            no_symlink(&d)?;
            std::fs::create_dir_all(&d)
                .map_err(|e| Failure::io("cannot create the cache directory", &d, &e))?;
        }
        Ok(())
    }

    /// Takes the cache-wide lock: shared for builds (several models build at once), exclusive
    /// for clearing (no build is running while entries or the target directory are removed).
    pub fn lock(&self, exclusive: bool) -> Result<File, Failure> {
        let p = self.locks_dir().join("cache.lock");
        let f = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&p)
            .map_err(|e| Failure::io("cannot open the cache lock", &p, &e))?;
        if exclusive { f.lock() } else { f.lock_shared() }
            .map_err(|e| Failure::io("cannot lock", &p, &e))?;
        Ok(f)
    }

    /// Entry directories.
    pub fn models_dir(&self) -> PathBuf {
        self.root.join("models")
    }

    /// Lock files.
    pub fn locks_dir(&self) -> PathBuf {
        self.root.join("locks")
    }

    /// Shared Cargo target directory.
    pub fn target_dir(&self) -> PathBuf {
        self.root.join("target")
    }

    /// Every complete entry, newest first.
    pub fn entries(&self) -> Vec<Entry> {
        let mut out = Vec::new();
        let Ok(rd) = std::fs::read_dir(self.models_dir()) else {
            return out;
        };
        for e in rd.flatten() {
            let dir = e.path();
            // Staging, failed, and replaced builds are dot directories, never entries.
            if e.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            if let Some(manifest) = read_manifest(&dir)
                && manifest["status"] == "success"
            {
                out.push(Entry { dir, manifest });
            }
        }
        out.sort_by(|a, b| b.built().total_cmp(&a.built()));
        out
    }

    /// The newest entry for model `name`. Entries built from different source files with the
    /// same name are ambiguous: the caller must pass the .stan`` path instead.
    pub fn find(&self, name: &str) -> Result<Entry, Failure> {
        let hits: Vec<Entry> = self
            .entries()
            .into_iter()
            .filter(|e| e.name() == name || (name.len() >= 8 && e.key().starts_with(name)))
            .collect();
        let Some(first) = hits.first() else {
            return Err(Failure::new(
                ExitCode::NotFound,
                format!(
                    "no compiled model named {name:?} in the cache {}; compile it first \
                     (cmdstan-rs compile {name}.stan) or pass the .stan file",
                    self.root.display()
                ),
            ));
        };
        let mut sources: Vec<&str> = hits.iter().map(Entry::source).collect();
        sources.sort_unstable();
        sources.dedup();
        if sources.len() > 1 {
            return Err(Failure::usage(format!(
                "the name {name:?} matches models compiled from different files ({}); pass the \
                 .stan file instead",
                sources.join(", ")
            )));
        }
        // An optimized build is preferred over a newer `--dev` build of the same model.
        Ok(hits
            .iter()
            .find(|e| e.manifest["profile"] == "release")
            .unwrap_or(first)
            .clone())
    }

    /// Removes entries (all, or those named `name`) and, with `all`, the shared target
    /// directory. Only directories inside this cache are touched.
    pub fn clear(&self, name: Option<&str>, all: bool) -> Result<Vec<PathBuf>, Failure> {
        if !self.root.join(MARKER).exists() {
            return Err(Failure::usage(format!(
                "{} is not a cmdstan-rs cache (no {MARKER} file); nothing was removed",
                self.root.display()
            )));
        }
        // Only plain directories of this cache are ever traversed or removed.
        let models = self.models_dir();
        no_symlink(&models)?;
        no_symlink(&self.target_dir())?;
        no_symlink(&self.locks_dir())?;
        let _guard = self.lock(true)?;
        let mut removed = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&models) {
            for e in rd.flatten() {
                let dir = e.path();
                // Never follow a symbolic link out of the cache.
                let Ok(meta) = std::fs::symlink_metadata(&dir) else {
                    continue;
                };
                if !meta.is_dir() {
                    continue;
                }
                let matches = match name {
                    None => true,
                    Some(n) => read_manifest(&dir).is_some_and(|m| m["name"] == n),
                };
                if matches {
                    std::fs::remove_dir_all(&dir)
                        .map_err(|e| Failure::io("cannot remove", &dir, &e))?;
                    removed.push(dir);
                }
            }
        }
        if all && name.is_none() {
            let t = self.target_dir();
            if t.exists() {
                std::fs::remove_dir_all(&t).map_err(|e| Failure::io("cannot remove", &t, &e))?;
                removed.push(t);
            }
        }
        Ok(removed)
    }
}

/// The manifest of an entry directory, if it has one.
pub fn read_manifest(dir: &Path) -> Option<Value> {
    let text = std::fs::read_to_string(dir.join("manifest.json")).ok()?;
    serde_json::from_str(&text).ok()
}

fn default_root() -> Option<PathBuf> {
    let env = |k: &str| {
        std::env::var_os(k)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    if cfg!(windows) {
        env("LOCALAPPDATA").map(|d| d.join("cmdstan-rs"))
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library/Caches/cmdstan-rs"))
    } else {
        env("XDG_CACHE_HOME")
            .or_else(|| env("HOME").map(|h| h.join(".cache")))
            .map(|d| d.join("cmdstan-rs"))
    }
}
