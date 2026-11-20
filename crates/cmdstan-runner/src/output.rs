//! Output files of the methods whose results stan-services returns in memory (optimization,
//! ADVI, Pathfinder, Laplace, standalone generated quantities, log densities), the adapted
//! metric files, and the run manifest.
//!
//! Sampling CSV files are written by stan-services itself; this module only chooses their
//! location. The tables written here follow the Stan CSV layout of the corresponding CmdStan
//! method (configuration comments, one header line, rows of values formatted like C's `%g`),
//! but cmdstan-rs does not claim that every comment line matches CmdStan.
//!
//! Every file is first written to a uniquely named temporary file in the target directory and
//! then published: without `--overwrite` by a hard link that fails if the target exists, so an
//! existing file is never replaced; with `--overwrite` by a rename.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use stan_services::output::format::format_g;

use crate::exit::{ExitCode, Failure};

/// A table of named columns with configuration comments.
pub struct Table<'a> {
    /// `(key, value)` comment lines written before the header, as `# key = value`.
    pub config: &'a [(String, String)],
    /// Column names.
    pub columns: &'a [String],
    /// Rows; every row has `columns.len()` values.
    pub rows: &'a [Vec<f64>],
    /// Comment lines written after the rows (timing, notes).
    pub trailer: &'a [String],
    /// Significant digits.
    pub sig_figs: usize,
}

impl Table<'_> {
    fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        for (k, v) in self.config {
            writeln!(w, "# {k} = {}", one_line(v))?;
        }
        writeln!(w, "{}", self.columns.join(","))?;
        for row in self.rows {
            debug_assert_eq!(row.len(), self.columns.len());
            let cells: Vec<String> = row.iter().map(|&x| format_g(x, self.sig_figs)).collect();
            writeln!(w, "{}", cells.join(","))?;
        }
        for t in self.trailer {
            writeln!(w, "# {}", one_line(t))?;
        }
        Ok(())
    }
}

fn one_line(s: &str) -> String {
    s.replace(['\n', '\r'], " ")
}

/// Fails if `path` exists and overwriting was not requested. Called before a method runs, so
/// that a long computation is not wasted on an output that cannot be written.
pub fn preflight(path: &Path, overwrite: bool) -> Result<(), Failure> {
    if !overwrite && path.exists() {
        return Err(Failure::new(
            ExitCode::Output,
            format!(
                "{} exists; pass --overwrite to replace it, or choose another output file",
                path.display()
            ),
        ));
    }
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)
            .map_err(|e| Failure::io("cannot create the output directory", dir, &e))?;
    }
    Ok(())
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(target: &Path) -> PathBuf {
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".into());
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    target.with_file_name(format!(".{name}.{}.{n}.tmp", std::process::id()))
}

/// Writes `contents` to `target` through a temporary file.
pub fn write_file(
    target: &Path,
    overwrite: bool,
    contents: impl FnOnce(&mut BufWriter<File>) -> io::Result<()>,
) -> Result<(), Failure> {
    preflight(target, overwrite)?;
    let tmp = temp_path(target);
    let result = (|| {
        let f = File::create_new(&tmp)?;
        let mut w = BufWriter::new(f);
        contents(&mut w)?;
        let f = w.into_inner().map_err(|e| e.into_error())?;
        f.sync_all()?;
        publish(&tmp, target, overwrite)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result.map_err(|e| Failure::io("cannot write", target, &e))
}

fn publish(tmp: &Path, target: &Path, overwrite: bool) -> io::Result<()> {
    if overwrite {
        return std::fs::rename(tmp, target);
    }
    match std::fs::hard_link(tmp, target) {
        Ok(()) => std::fs::remove_file(tmp),
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Err(e),
        // File systems without hard links (exFAT, some network mounts): create the target
        // exclusively, which still never replaces an existing file, and copy into it.
        Err(_) => {
            let mut out = File::create_new(target)?;
            io::copy(&mut File::open(tmp)?, &mut out)?;
            out.sync_all()?;
            std::fs::remove_file(tmp)
        }
    }
}

/// Writes a table as a Stan CSV file.
pub fn write_table(target: &Path, overwrite: bool, table: &Table<'_>) -> Result<(), Failure> {
    write_file(target, overwrite, |w| table.write_to(w))
}

/// Writes JSON (pretty printed) to `target`.
pub fn write_json(
    target: &Path,
    overwrite: bool,
    value: &serde_json::Value,
) -> Result<(), Failure> {
    write_file(target, overwrite, |w| {
        serde_json::to_writer_pretty(&mut *w, value).map_err(io::Error::other)?;
        writeln!(w)
    })
}

/// A JSON number, or for a nonfinite value the string `"NaN"`, `"Inf"`, or `"-Inf"` (JSON has
/// no NaN or infinity), as stan-services writes them.
pub fn num(x: f64) -> serde_json::Value {
    stan_services::output::metadata::num(x)
}

/// `dir/stem<suffix>.ext` for an output file `dir/stem.ext`.
pub fn sibling(file: &Path, suffix: &str, ext: &str) -> PathBuf {
    let stem = file
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".into());
    file.with_file_name(format!("{stem}{suffix}.{ext}"))
}

/// Hex SHA-256 of a file, for provenance (never the file's contents).
pub fn sha256_file(path: &Path) -> io::Result<String> {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    let mut f = File::open(path)?;
    io::copy(&mut f, &mut h)?;
    Ok(hex(&h.finalize()))
}

/// Lowercase hexadecimal.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
