//! Posterior summaries of Stan CSV files from `sample`.
//!
//! Files are read with stan-services' reader and every estimate comes from stan-services'
//! `summarize_draws` (stan-inference's rank-normalized R-hat, bulk and tail ESS, and MCSE) and
//! stan-inference's E-BFMI. This module only selects and checks the files, counts sampler
//! flags, and formats the results.
//!
//! The summary refuses incomplete files, files from other methods, files of different models or
//! columns, chains of different lengths, and the same file given twice. For a results
//! directory, the chains are those listed in the run's `*_metadata.json` (stale files from
//! earlier runs are ignored, missing files are errors). Failed chains make it fail unless
//! `allow_partial` is set, and are then reported as excluded.

use std::path::{Path, PathBuf};

use cmdstan_runner::output::num;
use cmdstan_runner::{ExitCode, Failure};
use serde_json::{Value, json};
use stan_services::diagnostics::summarize_draws;
use stan_services::{StanCsv, Thresholds, VariableSummary, read_stan_csv};

/// What to summarize.
pub struct SummaryOptions {
    /// Files or directories.
    pub inputs: Vec<PathBuf>,
    /// Summarize the completed chains even if some chains failed.
    pub allow_partial: bool,
    /// Output format.
    pub format: Format,
    /// Significant digits in the table.
    pub sig_figs: usize,
}

/// Output format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Aligned text.
    Table,
    /// One JSON document.
    Json,
    /// CSV.
    Csv,
}

struct ChainFile {
    path: PathBuf,
    csv: StanCsv,
    /// This chain's entry in the run metadata (stan-services `*_metadata.json`), if known.
    meta: Option<Value>,
}

/// The run in a results directory, from its stan-services metadata: the files of every chain
/// it wrote, and each chain's metadata. Files in the directory that the run did not write
/// (left over from an earlier run with more chains, or from another run) are not part of it.
struct RunFiles {
    metadata: PathBuf,
    /// `(csv file, chain metadata)` of completed chains.
    completed: Vec<(PathBuf, Value)>,
    /// Chains that failed or were cancelled.
    failed: Vec<String>,
}

fn run_in_dir(dir: &Path) -> Result<Option<RunFiles>, Failure> {
    let rd = std::fs::read_dir(dir).map_err(|e| Failure::io("cannot read directory", dir, &e))?;
    let mut metas: Vec<PathBuf> = rd
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with("_metadata.json"))
        })
        .collect();
    metas.sort();
    let Some(metadata) = metas.first().cloned() else {
        return Ok(None);
    };
    if metas.len() > 1 {
        return Err(Failure::usage(format!(
            "{} holds several runs ({}); pass the CSV files of one run",
            dir.display(),
            metas
                .iter()
                .map(|m| m.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    let text = std::fs::read_to_string(&metadata)
        .map_err(|e| Failure::io("cannot read", &metadata, &e))?;
    let v: Value = serde_json::from_str(&text)
        .map_err(|e| Failure::usage(format!("{} is not valid JSON: {e}", metadata.display())))?;
    let mut completed = Vec::new();
    let mut failed = Vec::new();
    for c in v["chains"].as_array().into_iter().flatten() {
        let id = c["chain_id"].to_string();
        if c["outcome"]["status"] != "completed" {
            failed.push(format!(
                "chain {id} {}",
                c["outcome"]["status"]
                    .as_str()
                    .unwrap_or("did not complete")
            ));
            continue;
        }
        let csv = c["files"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(Path::new)
            .find(|f| f.extension().is_some_and(|e| e == "csv"))
            .and_then(Path::file_name)
            .map(|f| dir.join(f));
        match csv {
            Some(f) if f.is_file() => completed.push((f, c.clone())),
            Some(f) => {
                return Err(Failure::new(
                    ExitCode::NotFound,
                    format!(
                        "chain {id} of the run in {} completed, but its file {} is missing",
                        metadata.display(),
                        f.display()
                    ),
                ));
            }
            None => failed.push(format!("chain {id} has no output file")),
        }
    }
    // Chains that never started are not in `chains`: compare with the configured number.
    let configured = v["config"]["chains"].as_u64().unwrap_or(0) as usize;
    let seen = v["chains"].as_array().map_or(0, Vec::len);
    if configured > seen {
        failed.push(format!("{} configured chains never ran", configured - seen));
    }
    Ok(Some(RunFiles {
        metadata,
        completed,
        failed,
    }))
}

fn collect(opts: &SummaryOptions) -> Result<(Vec<ChainFile>, Vec<String>), Failure> {
    let mut files = Vec::new();
    let mut partial: Vec<String> = Vec::new();
    let mut explicit = Vec::new();
    let mut metas: Vec<Option<Value>> = Vec::new();
    for input in &opts.inputs {
        if input.is_dir()
            && let Some(run) = run_in_dir(input)?
        {
            eprintln!("run {}", run.metadata.display());
            partial.extend(run.failed);
            for (f, m) in run.completed {
                files.push((f, true));
                metas.push(Some(m));
            }
        } else if input.is_dir() {
            let rd = std::fs::read_dir(input)
                .map_err(|e| Failure::io("cannot read directory", input, &e))?;
            let mut paths: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
            paths.sort();
            for p in paths {
                let n = p
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if n.ends_with(".csv.partial") {
                    partial.push(p.display().to_string());
                } else if n.ends_with(".csv") {
                    files.push((p, false));
                    metas.push(None);
                }
            }
        } else if input.is_file() {
            files.push((input.clone(), true));
            metas.push(None);
            explicit.push(input.clone());
        } else {
            return Err(Failure::new(
                ExitCode::NotFound,
                format!("{} does not exist", input.display()),
            ));
        }
    }
    let mut chains = Vec::new();
    for ((path, given), meta) in files.into_iter().zip(metas) {
        let csv = read_stan_csv(&path).map_err(|e| {
            Failure::usage(format!(
                "{} is not a readable Stan CSV file: {e}",
                path.display()
            ))
        })?;
        let method = csv.config_value("method").unwrap_or("sample").to_string();
        if method != "sample" {
            if given {
                return Err(Failure::usage(format!(
                    "{} comes from method {method}, not sample; MCMC summaries do not apply",
                    path.display()
                )));
            }
            // Other outputs in a results directory (generated quantities, optimization) are
            // not chains of this run.
            continue;
        }
        chains.push(ChainFile { path, csv, meta });
    }
    if chains.is_empty() {
        return Err(Failure::new(
            ExitCode::NotFound,
            "no Stan CSV files from sample were found",
        ));
    }
    Ok((chains, partial))
}

fn check(chains: &[ChainFile]) -> Result<(), Failure> {
    let first = &chains[0];
    let model = first.csv.config_value("model");
    for c in chains {
        if !c.csv.is_complete() {
            return Err(Failure::usage(format!(
                "{} is incomplete ({}); its draws are not a posterior sample",
                c.path.display(),
                c.csv.status.as_deref().unwrap_or("missing timing or draws")
            )));
        }
        if c.csv.config_value("model") != model {
            return Err(Failure::usage(format!(
                "{} and {} come from different models; summarize one model at a time",
                first.path.display(),
                c.path.display()
            )));
        }
        if c.csv.columns != first.csv.columns {
            return Err(Failure::usage(format!(
                "{} and {} have different columns",
                first.path.display(),
                c.path.display()
            )));
        }
        if c.csv.draws().is_empty() {
            return Err(Failure::usage(format!(
                "{} has no post-warmup draws",
                c.path.display()
            )));
        }
    }
    // The same file given twice would double count its draws.
    let mut paths: Vec<PathBuf> = chains
        .iter()
        .map(|c| c.path.canonicalize().unwrap_or_else(|_| c.path.clone()))
        .collect();
    let n = paths.len();
    paths.sort();
    paths.dedup();
    if paths.len() != n {
        return Err(Failure::usage("the same file was given twice"));
    }
    // Two chains with identical draws (two runs with the same seed and id) would make R-hat look
    // perfect and double the ESS.
    for (i, a) in chains.iter().enumerate() {
        if let Some(b) = chains[i + 1..]
            .iter()
            .find(|b| b.csv.draws() == a.csv.draws())
        {
            return Err(Failure::usage(format!(
                "{} and {} contain identical draws (the same chain twice, or two runs with the \
                 same seed and id); they are not independent chains",
                a.path.display(),
                b.path.display()
            )));
        }
    }
    // R-hat and ESS need chains of equal length.
    let len = first.csv.draws().len();
    if let Some(c) = chains.iter().find(|c| c.csv.draws().len() != len) {
        return Err(Failure::usage(format!(
            "{} has {} draws but {} has {len}; chains must have the same number of draws",
            c.path.display(),
            c.csv.draws().len(),
            first.path.display()
        )));
    }
    Ok(())
}

fn column(c: &ChainFile, j: usize) -> Vec<f64> {
    c.csv.draws().iter().map(|r| r[j]).collect()
}

/// Sampler flags of one chain. Divergences, tree depth hits, and E-BFMI come from the run's
/// `*_metadata.json`, which covers every post-warmup transition, when the file was selected
/// through it (a results directory); otherwise they are counted over the retained draws, which with thinning misses
/// transitions that were not saved (stan-dev/cmdstan#1185).
struct ChainFlags {
    path: PathBuf,
    id: Option<String>,
    draws: usize,
    /// Whether the counts cover every post-warmup transition (from the metadata).
    all_transitions: bool,
    divergent: Option<usize>,
    max_treedepth_hits: Option<usize>,
    ebfmi: Option<f64>,
}

fn flags(c: &ChainFile) -> ChainFlags {
    let count = |name: &str, pred: &dyn Fn(f64) -> bool| {
        c.csv
            .column_index(name)
            .map(|j| column(c, j).into_iter().filter(|&x| pred(x)).count())
    };
    let max_depth: Option<f64> = c.csv.config_value("max_depth").and_then(|v| v.parse().ok());
    ChainFlags {
        path: c.path.clone(),
        id: c.csv.config_value("id").map(str::to_string),
        draws: c.csv.draws().len(),
        all_transitions: false,
        divergent: count("divergent__", &|x| x != 0.0),
        max_treedepth_hits: max_depth.and_then(|m| count("treedepth__", &|x| x >= m)),
        ebfmi: c
            .csv
            .column_index("energy__")
            .map(|j| stan_inference::diagnostics::ebfmi(&column(c, j))),
    }
}

/// Runs the summary command; returns the text to print on stdout.
/// Returns the text to print and whether a convergence or sampler check failed.
pub fn summary(opts: &SummaryOptions) -> Result<(String, bool), Failure> {
    let (chains, partial) = collect(opts)?;
    check(&chains)?;
    let mut warnings = Vec::new();
    if !partial.is_empty() {
        let list = partial.clone();
        if !opts.allow_partial {
            return Err(Failure::new(
                ExitCode::Partial,
                format!(
                    "failed or cancelled chains left incomplete files ({}); the run is not \
                     complete. Pass --allow-partial to summarize the completed chains only",
                    list.join(", ")
                ),
            ));
        }
        warnings.push(format!(
            "summarizing {} completed chains; excluded incomplete chains: {}",
            chains.len(),
            list.join(", ")
        ));
    }
    let columns = &chains[0].csv.columns;
    let vars: Vec<(usize, &String)> = columns
        .iter()
        .enumerate()
        .filter(|(_, n)| *n == "lp__" || !n.ends_with("__"))
        .collect();
    let summaries: Vec<VariableSummary> = vars
        .iter()
        .map(|&(j, name)| {
            let cols: Vec<Vec<f64>> = chains.iter().map(|c| column(c, j)).collect();
            let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
            summarize_draws(name, &refs)
        })
        .collect();
    let mut chain_flags: Vec<ChainFlags> = chains.iter().map(flags).collect();
    // Counts over every post-warmup transition, from the metadata of the run that wrote the
    // file (only when the file was selected through that metadata).
    for (f, c) in chain_flags.iter_mut().zip(&chains) {
        if let Some(m) = &c.meta {
            let d = &m["sampler_diagnostics"];
            let n = |k: &str| d[k].as_u64().map(|x| x as usize);
            f.divergent = n("divergent").or(f.divergent);
            f.max_treedepth_hits = n("max_treedepth_hits").or(f.max_treedepth_hits);
            f.ebfmi = d["ebfmi"].as_f64().or(f.ebfmi);
            f.all_transitions = true;
        }
    }
    if chains[0].csv.config_value("algorithm") == Some("fixed_param") {
        warnings.push(
            "fixed_param output: the draws do not come from a Markov chain exploring the \
             posterior, so R-hat and ESS describe only the generated quantities' random draws"
                .into(),
        );
    }
    let t = Thresholds::default();
    let k = chains.len() as f64;
    // Whether a convergence or sampler check failed (for `--strict`).
    let mut flagged = false;
    for s in &summaries {
        if s.rhat.is_some_and(|r| r > t.rhat) {
            flagged = true;
            warnings.push(format!(
                "{}: R-hat {:.3} exceeds {}",
                s.name,
                s.rhat.unwrap_or(f64::NAN),
                t.rhat
            ));
        }
        if s.ess_bulk < t.ess_per_chain * k || s.ess_tail < t.ess_per_chain * k {
            flagged = true;
            warnings.push(format!(
                "{}: bulk ESS {:.0} or tail ESS {:.0} is below {:.0}",
                s.name,
                s.ess_bulk,
                s.ess_tail,
                t.ess_per_chain * k
            ));
        }
        if s.nonfinite_draws > 0 {
            flagged = true;
            warnings.push(format!("{}: {} nonfinite draws", s.name, s.nonfinite_draws));
        }
        // A constant column has no defined R-hat or ESS and needs none; any other undefined
        // diagnostic is a failed check, never a pass.
        let undefined = !s.rhat.is_some_and(f64::is_finite)
            || !s.ess_bulk.is_finite()
            || !s.ess_tail.is_finite();
        if undefined && s.sd != 0.0 && s.nonfinite_draws == 0 {
            flagged = true;
            warnings.push(format!("{}: R-hat or ESS could not be computed", s.name));
        }
    }
    for f in &chain_flags {
        let id = f.id.clone().unwrap_or_else(|| f.path.display().to_string());
        let over = if f.all_transitions {
            "post-warmup transitions"
        } else {
            "retained draws"
        };
        if let Some(d) = f.divergent.filter(|&d| d > 0) {
            warnings.push(format!("chain {id}: {d} divergent {over}"));
            flagged = true;
        }
        if let Some(h) = f.max_treedepth_hits.filter(|&h| h > 0) {
            warnings.push(format!("chain {id}: {h} {over} hit the maximum tree depth"));
            flagged = true;
        }
        if let Some(e) = f.ebfmi.filter(|&e| e < t.ebfmi) {
            warnings.push(format!("chain {id}: E-BFMI {e:.3} is below {}", t.ebfmi));
            flagged = true;
        }
    }
    if chains.len() == 1 {
        warnings.push(
            "one chain: R-hat cannot detect chains stuck in different regions; run several chains"
                .into(),
        );
    }
    warnings.push(
        "passing these checks does not show that the chains converged; failing them is a reason \
         to investigate"
            .into(),
    );
    let out = match opts.format {
        Format::Json => {
            let v = json!({
                "model": chains[0].csv.config_value("model"),
                "chains": chain_flags.iter().map(|f| json!({
                    "file": f.path.display().to_string(),
                    "id": f.id,
                    "draws": f.draws,
                    "divergent": f.divergent,
                    "max_treedepth_hits": f.max_treedepth_hits,
                    "counted_over": if f.all_transitions { "post-warmup transitions" } else { "retained draws" },
                    "ebfmi": f.ebfmi.map(num),
                })).collect::<Vec<_>>(),
                "excluded": partial,
                "variables": summaries.iter().map(|s| json!({
                    "name": s.name,
                    "mean": num(s.mean),
                    "sd": num(s.sd),
                    "q5": num(s.q5),
                    "median": num(s.median),
                    "q95": num(s.q95),
                    "mcse_mean": num(s.mcse_mean),
                    "mcse_sd": num(s.mcse_sd),
                    "ess_bulk": num(s.ess_bulk),
                    "ess_tail": num(s.ess_tail),
                    "rhat": s.rhat.map(num),
                    "split_rhat_single_chain": s.split_rhat_single_chain.map(num),
                    "nonfinite_draws": s.nonfinite_draws,
                })).collect::<Vec<_>>(),
                "warnings": warnings,
                "notes": [
                    "divergences, tree depth hits, and E-BFMI cover every post-warmup transition when the chains were selected through the run metadata (see counted_over); otherwise only the retained draws, which with thinning miss the dropped transitions",
                ],
            });
            serde_json::to_string_pretty(&v).expect("serializes") + "\n"
        }
        Format::Csv => {
            let mut out = String::from(
                "variable,mean,sd,q5,median,q95,mcse_mean,mcse_sd,ess_bulk,ess_tail,rhat\n",
            );
            let g = |x: f64| stan_services::output::format::format_g(x, opts.sig_figs);
            for s in &summaries {
                out.push_str(&format!(
                    "{},{},{},{},{},{},{},{},{},{},{}\n",
                    s.name,
                    g(s.mean),
                    g(s.sd),
                    g(s.q5),
                    g(s.median),
                    g(s.q95),
                    g(s.mcse_mean),
                    g(s.mcse_sd),
                    g(s.ess_bulk),
                    g(s.ess_tail),
                    s.rhat.map(g).unwrap_or_else(|| "NA".into())
                ));
            }
            for w in &warnings {
                eprintln!("warning: {w}");
            }
            out
        }
        Format::Table => table(&summaries, &chain_flags, &warnings, opts.sig_figs),
    };
    Ok((out, flagged))
}

fn table(s: &[VariableSummary], flags: &[ChainFlags], warnings: &[String], sig: usize) -> String {
    let g = |x: f64| stan_services::output::format::format_g(x, sig);
    let head = [
        "variable",
        "mean",
        "sd",
        "q5",
        "median",
        "q95",
        "mcse_mean",
        "mcse_sd",
        "ess_bulk",
        "ess_tail",
        "rhat",
    ];
    let mut rows: Vec<Vec<String>> = vec![head.iter().map(|h| h.to_string()).collect()];
    for v in s {
        rows.push(vec![
            v.name.clone(),
            g(v.mean),
            g(v.sd),
            g(v.q5),
            g(v.median),
            g(v.q95),
            g(v.mcse_mean),
            g(v.mcse_sd),
            format!("{:.0}", v.ess_bulk),
            format!("{:.0}", v.ess_tail),
            v.rhat
                .map(|r| format!("{r:.3}"))
                .unwrap_or_else(|| "NA".into()),
        ]);
    }
    let widths: Vec<usize> = (0..head.len())
        .map(|j| rows.iter().map(|r| r[j].len()).max().unwrap_or(0))
        .collect();
    let mut out = String::new();
    for r in &rows {
        let cells: Vec<String> = r
            .iter()
            .enumerate()
            .map(|(j, c)| {
                if j == 0 {
                    format!("{c:<w$}", w = widths[j])
                } else {
                    format!("{c:>w$}", w = widths[j])
                }
            })
            .collect();
        out.push_str(cells.join("  ").trim_end());
        out.push('\n');
    }
    out.push_str(&format!(
        "\n{} chains, {} retained draws\n",
        flags.len(),
        flags.iter().map(|f| f.draws).sum::<usize>()
    ));
    for f in flags {
        let opt = |x: Option<usize>| x.map(|v| v.to_string()).unwrap_or_else(|| "NA".into());
        out.push_str(&format!(
            "  chain {}: {} draws, divergent {}, max tree depth hits {}, E-BFMI {}\n",
            f.id.as_deref().unwrap_or("?"),
            f.draws,
            opt(f.divergent),
            opt(f.max_treedepth_hits),
            f.ebfmi
                .map(|e| format!("{e:.3}"))
                .unwrap_or_else(|| "NA".into()),
        ));
    }
    if !warnings.is_empty() {
        out.push('\n');
        for w in warnings {
            out.push_str(&format!("warning: {w}\n"));
        }
    }
    out
}

/// Summarizes the manifest and services metadata of a run directory or manifest file.
pub fn inspect_run(path: &Path, as_json: bool) -> Result<String, Failure> {
    let mut manifests = Vec::new();
    let mut metadata = Vec::new();
    let mut push = |p: PathBuf| {
        let n = p
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        if n.ends_with("_manifest.json") {
            manifests.push(p);
        } else if n.ends_with("_metadata.json") {
            metadata.push(p);
        }
    };
    if path.is_dir() {
        let rd = std::fs::read_dir(path).map_err(|e| Failure::io("cannot read", path, &e))?;
        let mut v: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        v.sort();
        v.into_iter().for_each(&mut push);
    } else if path.is_file() {
        push(path.to_path_buf());
    } else {
        return Err(Failure::new(
            ExitCode::NotFound,
            format!("{} does not exist", path.display()),
        ));
    }
    if manifests.is_empty() {
        return Err(Failure::new(
            ExitCode::NotFound,
            format!("no *_manifest.json run manifest in {}", path.display()),
        ));
    }
    let read = |p: &Path| -> Result<Value, Failure> {
        let t = std::fs::read_to_string(p).map_err(|e| Failure::io("cannot read", p, &e))?;
        serde_json::from_str(&t).map_err(|e| Failure::usage(format!("{}: {e}", p.display())))
    };
    let mut docs = Vec::new();
    for m in &manifests {
        docs.push(json!({"file": m.display().to_string(), "manifest": read(m)?}));
    }
    for m in &metadata {
        docs.push(json!({"file": m.display().to_string(), "services_metadata": read(m)?}));
    }
    if as_json {
        return Ok(serde_json::to_string_pretty(&docs).expect("serializes") + "\n");
    }
    let mut out = String::new();
    for d in &docs {
        let Some(m) = d.get("manifest") else { continue };
        let b = &m["build"];
        out.push_str(&format!(
            "run manifest {}\n",
            d["file"].as_str().unwrap_or("")
        ));
        let line = |k: &str, v: &Value| {
            format!(
                "  {k:<18} {}\n",
                v.as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| v.to_string())
            )
        };
        out.push_str(&line("model", &m["model"]));
        out.push_str(&line("method", &m["method"]));
        out.push_str(&line("exit status", &m["exit"]["status"]));
        out.push_str(&line("seed", &m["seed"]));
        out.push_str(&line("first chain id", &m["id"]));
        out.push_str(&line("source", &b["source"]["path"]));
        out.push_str(&line("source sha256", &b["source"]["sha256"]));
        out.push_str(&line("data sha256", &m["data"]["sha256"]));
        out.push_str(&line("compiler", &b["compiler"]));
        out.push_str(&line("runtime", &b["runtime"]));
        out.push_str(&line("math", &b["math"]));
        out.push_str(&line("rustc", &b["rustc"]));
        out.push_str(&line("target", &b["target"]));
        out.push_str(&line("profile", &b["profile"]));
        out.push_str(&line("producer", &m["producer"]));
        out.push_str("  arguments:\n");
        if let Some(a) = m["arguments"].as_object() {
            for (k, v) in a {
                out.push_str(&format!("    {k} = {}\n", v.as_str().unwrap_or("")));
            }
        }
        if let Some(files) = m["files"].as_array() {
            out.push_str("  files:\n");
            for f in files {
                out.push_str(&format!("    {}\n", f.as_str().unwrap_or("")));
            }
        }
        if let Some(w) = m["warnings"].as_array().filter(|w| !w.is_empty()) {
            out.push_str("  warnings:\n");
            for x in w {
                out.push_str(&format!("    {}\n", x.as_str().unwrap_or("")));
            }
        }
    }
    Ok(out)
}
