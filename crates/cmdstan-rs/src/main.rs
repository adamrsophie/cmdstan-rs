//! `cmdstan-rs`: compile Stan programs into native Rust model executables and run Bayesian
//! inference with them.
//!
//! The command line is a thin layer: compilation is `compile`, model runs translate options into
//! the CmdStan arguments of the model executable (whose runtime is `cmdstan-runner`), and
//! summaries read Stan CSV files with stan-services.

mod cache;
mod compile;
mod exec;
mod summary;

use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand, ValueEnum};
use cmdstan_runner::{ExitCode, Failure};
use serde_json::Value;

use crate::cache::Cache;
use crate::compile::{CompileOptions, Profile};
use crate::exec::OnTheFly;

const EXIT_CODES: &str = "Exit codes: 0 success, 2 invalid options, 3 missing file, 4 Stan \
compiler error, 5 Rust build failure, 6 invalid data, 7 invalid initialization, 8 inference \
failure, 9 output failure, 10 some chains failed, 11 diagnostic check failed (diagnose, \
summary --strict), 70 internal error, 130 interrupted.";

#[derive(Parser)]
#[command(
    name = "cmdstan-rs",
    version,
    about = "Rust-native Stan compilation and Bayesian inference (unofficial; not a Stan \
             Development Team product)",
    after_help = EXIT_CODES
)]
struct Cli {
    /// Build cache directory (default: $CMDSTAN_RS_CACHE, else the platform cache directory).
    #[arg(long, global = true, value_name = "DIR")]
    cache_dir: Option<PathBuf>,
    /// Report errors as one JSON object on stderr.
    #[arg(long, global = true)]
    json_errors: bool,
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compile a Stan program into a model executable (cached).
    Compile(CompileArgs),
    /// Parse and type check a Stan program without building it.
    Check(CheckArgs),
    /// Show a compiled model's variables, dimensions, and build provenance.
    Inspect(InspectArgs),
    /// Draw from the posterior with NUTS (or static HMC, or fixed parameters).
    Sample(SampleArgs),
    /// Find a posterior mode (or maximum likelihood estimate) with L-BFGS, BFGS, or Newton.
    Optimize(OptimizeArgs),
    /// Approximate the posterior with ADVI.
    Variational(VariationalArgs),
    /// Approximate the posterior with Pathfinder.
    Pathfinder(PathfinderArgs),
    /// Gaussian approximation at a mode (Laplace).
    Laplace(LaplaceArgs),
    /// Evaluate transformed parameters and generated quantities for existing draws.
    #[command(name = "generate-quantities")]
    GenerateQuantities(GqArgs),
    /// Evaluate the log density and its gradient.
    #[command(name = "log-prob")]
    LogProb(LogProbArgs),
    /// Compare gradients with finite differences.
    Diagnose(DiagnoseArgs),
    /// Summarize Stan CSV files from sample.
    Summary(SummaryArgs),
    /// Show the provenance and configuration recorded for a run.
    #[command(name = "inspect-run")]
    InspectRun(InspectRunArgs),
    /// Run a model executable with CmdStan's argument syntax.
    Run(RunArgs),
    /// Manage the build cache.
    #[command(subcommand)]
    Cache(CacheCmd),
    /// Print the versions of cmdstan-rs, its components, and the Rust toolchain.
    Version,
}

#[derive(Args)]
struct BuildFlags {
    /// `#include` search directory (repeatable).
    #[arg(long = "include-path", value_name = "DIR")]
    include_paths: Vec<PathBuf>,
    /// Pass --offline to Cargo: no network access, dependencies must already be fetched.
    #[arg(long)]
    offline: bool,
    /// Build the unoptimized dev profile instead of release.
    #[arg(long)]
    dev: bool,
}

#[derive(Args)]
struct CompileArgs {
    /// The Stan program.
    source: PathBuf,
    /// Artifact name (default: the file stem).
    #[arg(long)]
    name: Option<String>,
    /// Model name recorded in outputs (default: `<stem>_model`, as stanc3).
    #[arg(long)]
    model_name: Option<String>,
    /// Accept functions declared without a definition (the build then fails: linking user
    /// code is not supported yet).
    #[arg(long)]
    allow_undefined: bool,
    #[command(flatten)]
    build: BuildFlags,
    /// Rebuild even if the cache has this model.
    #[arg(long)]
    force: bool,
    /// Also copy the executable to --output (a standalone model executable).
    #[arg(long)]
    standalone: bool,
    /// Where to copy the standalone executable (file or directory).
    #[arg(long, short)]
    output: Option<PathBuf>,
    /// Replace an existing --output file.
    #[arg(long)]
    overwrite: bool,
    /// Show Cargo's output.
    #[arg(long, short)]
    verbose: bool,
    /// Print the build manifest as JSON on stdout.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct CheckArgs {
    /// The Stan program.
    source: PathBuf,
    /// `#include` search directory (repeatable).
    #[arg(long = "include-path", value_name = "DIR")]
    include_paths: Vec<PathBuf>,
    /// Accept functions declared without a definition.
    #[arg(long)]
    allow_undefined: bool,
    /// Print the program's variables as JSON (stanc --info).
    #[arg(long)]
    info: bool,
}

#[derive(Args)]
struct InspectArgs {
    /// Model name, executable, or .stan file.
    model: String,
    /// Data file, to resolve data-dependent dimensions.
    #[arg(long)]
    data: Option<PathBuf>,
    /// Print JSON.
    #[arg(long)]
    json: bool,
    #[command(flatten)]
    build: BuildFlags,
}

/// Options of every command that runs a model.
#[derive(Args)]
struct Common {
    /// Model name (from the cache), model executable, or .stan file (compiled if needed).
    model: String,
    /// JSON data file.
    #[arg(long)]
    data: Option<PathBuf>,
    /// Initialization: a radius (0 for zeros) or a JSON file of initial values.
    #[arg(long)]
    init: Option<String>,
    /// Random seed, 0 to 4294967295 (default: from the clock; the seed used is recorded).
    #[arg(long)]
    seed: Option<u32>,
    /// Identifier of the first chain (selects the random streams; default 1, or for
    /// generate-quantities the ids recorded in the draw files).
    #[arg(long)]
    id: Option<u32>,
    /// Significant digits of real values in output files (1 to 18).
    #[arg(long, default_value_t = 8)]
    sig_figs: u32,
    /// Progress every this many iterations (0: none).
    #[arg(long, default_value_t = 100)]
    refresh: usize,
    /// Replace existing output files.
    #[arg(long)]
    overwrite: bool,
    /// No progress output.
    #[arg(long, short)]
    quiet: bool,
    #[command(flatten)]
    build: BuildFlags,
}

#[derive(Clone, Copy, ValueEnum)]
enum MetricArg {
    Diag,
    Dense,
    Unit,
}

#[derive(Clone, Copy, ValueEnum)]
enum EngineArg {
    Nuts,
    Static,
}

#[derive(Args)]
struct SampleArgs {
    #[command(flatten)]
    common: Common,
    /// Number of chains.
    #[arg(long, default_value_t = 4)]
    chains: usize,
    /// Chains run at once (default: all, at most the number of CPUs).
    #[arg(long)]
    parallel_chains: Option<usize>,
    /// Warmup iterations per chain.
    #[arg(long, default_value_t = 1000)]
    warmup: usize,
    /// Post-warmup iterations per chain (retained draws: ceil(draws / thin)).
    #[arg(long, default_value_t = 1000)]
    draws: usize,
    /// Keep every thin-th iteration.
    #[arg(long, default_value_t = 1)]
    thin: usize,
    /// Also write the warmup draws.
    #[arg(long)]
    save_warmup: bool,
    /// Target acceptance statistic of step size adaptation.
    #[arg(long, default_value_t = 0.8)]
    adapt_delta: f64,
    /// Turn adaptation off.
    #[arg(long)]
    no_adapt: bool,
    /// Maximum NUTS tree depth (default 10).
    #[arg(long)]
    max_treedepth: Option<usize>,
    /// HMC engine (default nuts).
    #[arg(long, value_enum)]
    engine: Option<EngineArg>,
    /// Integration time of static HMC.
    #[arg(long)]
    int_time: Option<f64>,
    /// Euclidean metric (default diag).
    #[arg(long, value_enum)]
    metric: Option<MetricArg>,
    /// Initial inverse metric (JSON with inv_metric).
    #[arg(long)]
    metric_file: Option<PathBuf>,
    /// Initial step size.
    #[arg(long)]
    step_size: Option<f64>,
    /// Step size jitter in [0, 1).
    #[arg(long)]
    step_size_jitter: Option<f64>,
    /// Write each chain's adapted step size and inverse metric as JSON.
    #[arg(long)]
    save_metric: bool,
    /// Do not move the parameters (for models with only generated quantities).
    #[arg(long)]
    fixed_param: bool,
    /// Output directory.
    #[arg(long, short, default_value = "output")]
    output: PathBuf,
    /// File name prefix: chain files are `<prefix>_<id>.csv` (default: `<model>_chain`).
    #[arg(long)]
    output_prefix: Option<String>,
}

#[derive(Clone, Copy, ValueEnum)]
enum OptimizerArg {
    Lbfgs,
    Bfgs,
    Newton,
}

#[derive(Args)]
struct OptimizerTolerances {
    /// First line search step.
    #[arg(long)]
    init_alpha: Option<f64>,
    /// Absolute objective tolerance.
    #[arg(long)]
    tol_obj: Option<f64>,
    /// Relative objective tolerance (units of machine epsilon).
    #[arg(long)]
    tol_rel_obj: Option<f64>,
    /// Gradient norm tolerance.
    #[arg(long)]
    tol_grad: Option<f64>,
    /// Relative gradient tolerance (units of machine epsilon).
    #[arg(long)]
    tol_rel_grad: Option<f64>,
    /// Parameter change tolerance.
    #[arg(long)]
    tol_param: Option<f64>,
    /// L-BFGS history size.
    #[arg(long)]
    history_size: Option<usize>,
}

#[derive(Args)]
struct OptimizeArgs {
    #[command(flatten)]
    common: Common,
    /// Algorithm.
    #[arg(long, value_enum, default_value_t = OptimizerArg::Lbfgs)]
    algorithm: OptimizerArg,
    /// Include the Jacobian (mode of the unconstrained density, as a Laplace approximation
    /// needs). Without it: the posterior mode of the constrained parameters.
    #[arg(long)]
    jacobian: bool,
    /// Maximum iterations.
    #[arg(long)]
    iter: Option<usize>,
    #[command(flatten)]
    tol: OptimizerTolerances,
    /// Output file (default: `<model>_optimize.csv`).
    #[arg(long, short)]
    output: Option<PathBuf>,
    /// Print the result as JSON on stdout.
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum FamilyArg {
    Meanfield,
    Fullrank,
}

#[derive(Args)]
struct VariationalArgs {
    #[command(flatten)]
    common: Common,
    /// Variational family.
    #[arg(long, value_enum, default_value_t = FamilyArg::Meanfield)]
    algorithm: FamilyArg,
    /// Maximum iterations.
    #[arg(long)]
    iter: Option<usize>,
    /// Draws per gradient.
    #[arg(long)]
    grad_samples: Option<usize>,
    /// Draws per ELBO estimate.
    #[arg(long)]
    elbo_samples: Option<usize>,
    /// Step size scale (chosen by grid search unless --no-adapt).
    #[arg(long)]
    eta: Option<f64>,
    /// Use --eta as given instead of the grid search.
    #[arg(long)]
    no_adapt: bool,
    /// Iterations per eta candidate in the grid search.
    #[arg(long)]
    adapt_iter: Option<usize>,
    /// Relative ELBO tolerance.
    #[arg(long)]
    tol_rel_obj: Option<f64>,
    /// Evaluate the ELBO every this many iterations.
    #[arg(long)]
    eval_elbo: Option<usize>,
    /// Approximate draws.
    #[arg(long)]
    draws: Option<usize>,
    /// Output file (default: `<model>_variational.csv`).
    #[arg(long, short)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct PathfinderArgs {
    #[command(flatten)]
    common: Common,
    /// Independent paths.
    #[arg(long)]
    num_paths: Option<usize>,
    /// Draws per path.
    #[arg(long)]
    draws: Option<usize>,
    /// Draws after importance resampling.
    #[arg(long)]
    psis_draws: Option<usize>,
    /// Return every path's draws without importance resampling.
    #[arg(long)]
    no_psis_resample: bool,
    /// L-BFGS iterations per path.
    #[arg(long)]
    max_lbfgs_iters: Option<usize>,
    /// Draws per ELBO estimate.
    #[arg(long)]
    num_elbo_draws: Option<usize>,
    #[command(flatten)]
    tol: OptimizerTolerances,
    /// Output file (default: `<model>_pathfinder.csv`).
    #[arg(long, short)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct LaplaceArgs {
    #[command(flatten)]
    common: Common,
    /// The mode: an optimize CSV file (computed with --jacobian unless --no-jacobian) or a JSON
    /// file of constrained values. Without it the mode is found by L-BFGS first.
    #[arg(long)]
    mode: Option<PathBuf>,
    /// Number of draws.
    #[arg(long)]
    draws: Option<usize>,
    /// Approximate the density of the constrained parameters at their mode (no Jacobian).
    #[arg(long)]
    no_jacobian: bool,
    /// Output file (default: `<model>_laplace.csv`).
    #[arg(long, short)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct GqArgs {
    #[command(flatten)]
    common: Common,
    /// Stan CSV files of draws from sample (one per chain).
    #[arg(long, required = true, num_args = 1..)]
    draws: Vec<PathBuf>,
    /// Output file stem: chain files are `<stem>_<id>.csv` (default: `<model>_gq.csv`).
    #[arg(long, short)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct LogProbArgs {
    #[command(flatten)]
    common: Common,
    /// JSON file of unconstrained vectors ({"params_r": [...]}).
    #[arg(long)]
    parameters: Option<PathBuf>,
    /// JSON file of constrained values, or a Stan CSV file of draws.
    #[arg(long)]
    constrained: Option<PathBuf>,
    /// Leave out the Jacobian of the constraining transforms.
    #[arg(long)]
    no_jacobian: bool,
    /// Output file (default: `<model>_log_prob.csv`).
    #[arg(long, short)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct DiagnoseArgs {
    #[command(flatten)]
    common: Common,
    /// JSON file of unconstrained vectors ({"params_r": [...]}); default: the initial point.
    #[arg(long)]
    parameters: Option<PathBuf>,
    /// Finite difference step.
    #[arg(long)]
    epsilon: Option<f64>,
    /// Error threshold.
    #[arg(long)]
    error: Option<f64>,
    /// Output file (default: `<model>_diagnose.csv`).
    #[arg(long, short)]
    output: Option<PathBuf>,
    /// Print the result as JSON on stdout.
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum FormatArg {
    Table,
    Json,
    Csv,
}

#[derive(Args)]
struct SummaryArgs {
    /// Directories or Stan CSV files.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
    /// Summarize the completed chains even if some chains failed.
    #[arg(long)]
    allow_partial: bool,
    /// Exit 11 when a convergence or sampler check fails (R-hat, ESS, divergences, tree depth,
    /// E-BFMI), so scripts can stop on unreliable runs.
    #[arg(long)]
    strict: bool,
    /// Output format.
    #[arg(long, value_enum, default_value_t = FormatArg::Table)]
    format: FormatArg,
    /// Significant digits.
    #[arg(long, default_value_t = 4)]
    sig_figs: usize,
}

#[derive(Args)]
struct InspectRunArgs {
    /// Run directory or manifest file.
    path: PathBuf,
    /// Print JSON.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
struct RunArgs {
    /// Model name, executable, or .stan file.
    model: String,
    #[command(flatten)]
    build: BuildFlags,
    /// CmdStan arguments for the model executable, after `--`.
    #[arg(last = true)]
    args: Vec<String>,
}

#[derive(Subcommand)]
enum CacheCmd {
    /// List compiled models.
    List,
    /// Print a cache entry's manifest.
    Inspect {
        /// Model name or cache key prefix.
        model: String,
    },
    /// Remove cache entries (only inside the cache directory).
    Clear {
        /// Only entries of this model.
        #[arg(long)]
        name: Option<String>,
        /// Also remove the shared build directory.
        #[arg(long)]
        all: bool,
    },
    /// Print the cache directory.
    Dir,
}

fn main() {
    let cli = match Cli::try_parse() {
        Ok(c) => c,
        Err(e) => {
            // Help and version are not errors.
            if !e.use_stderr() {
                let _ = e.print();
                std::process::exit(0);
            }
            if std::env::args().any(|a| a == "--json-errors") {
                Failure::usage(e.to_string().trim().to_string()).report(true);
                std::process::exit(ExitCode::Usage.code());
            }
            let _ = e.print();
            std::process::exit(ExitCode::Usage.code());
        }
    };
    let json_errors = cli.json_errors;
    let code = match dispatch(cli) {
        Ok(c) => c,
        Err(f) => {
            f.report(json_errors);
            f.code.code()
        }
    };
    std::process::exit(code);
}

fn fly(b: &BuildFlags) -> OnTheFly {
    OnTheFly {
        include_paths: b.include_paths.clone(),
        offline: b.offline,
        dev: b.dev,
    }
}

fn path_arg(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// The CmdStan arguments shared by every method, after the method's own arguments.
fn common_args(c: &Common, output_file: &Path, json_errors: bool) -> Result<Vec<String>, Failure> {
    if !(1..=18).contains(&c.sig_figs) {
        return Err(Failure::usage(format!(
            "--sig-figs {} is not in 1..=18",
            c.sig_figs
        )));
    }
    let mut a = Vec::new();
    if let Some(id) = c.id {
        a.push(format!("id={id}"));
    }
    if let Some(d) = &c.data {
        a.push("data".into());
        a.push(format!("file={}", path_arg(d)));
    }
    if let Some(i) = &c.init {
        a.push(format!("init={i}"));
    }
    if let Some(s) = c.seed {
        a.push("random".into());
        a.push(format!("seed={s}"));
    }
    a.push("output".into());
    a.push(format!("file={}", path_arg(output_file)));
    a.push(format!("refresh={}", c.refresh));
    a.push(format!("sig_figs={}", c.sig_figs));
    if c.overwrite {
        a.push("--overwrite".into());
    }
    if c.quiet {
        a.push("--quiet".into());
    }
    if json_errors {
        a.push("--json-errors".into());
    }
    Ok(a)
}

fn tol_args(t: &OptimizerTolerances, history: bool) -> Vec<String> {
    let mut a = Vec::new();
    let mut real = |k: &str, v: Option<f64>| {
        if let Some(v) = v {
            a.push(format!("{k}={v}"));
        }
    };
    real("init_alpha", t.init_alpha);
    real("tol_obj", t.tol_obj);
    real("tol_rel_obj", t.tol_rel_obj);
    real("tol_grad", t.tol_grad);
    real("tol_rel_grad", t.tol_rel_grad);
    real("tol_param", t.tol_param);
    if history && let Some(h) = t.history_size {
        a.push(format!("history_size={h}"));
    }
    a
}

fn any_tol(t: &OptimizerTolerances) -> bool {
    t.init_alpha.is_some()
        || t.tol_obj.is_some()
        || t.tol_rel_obj.is_some()
        || t.tol_grad.is_some()
        || t.tol_rel_grad.is_some()
        || t.tol_param.is_some()
        || t.history_size.is_some()
}

fn default_output(name: &str, method: &str) -> PathBuf {
    PathBuf::from(format!("{name}_{method}.csv"))
}

/// Resolves the model and runs it with `method_args` followed by the common arguments.
fn run_method(
    c: &Common,
    cache: &Cache,
    json_errors: bool,
    build_args: impl Fn(&str) -> Result<(Vec<String>, PathBuf), Failure>,
) -> Result<i32, Failure> {
    // Check the options before resolving the model, which can mean minutes of compilation.
    let (_, out) = build_args("model")?;
    common_args(c, &out, json_errors)?;
    let m = exec::resolve(&c.model, cache, &fly(&c.build))?;
    let (mut args, output) = build_args(&m.name)?;
    args.extend(common_args(c, &output, json_errors)?);
    exec::run(&m.executable, &args)
}

fn dispatch(cli: Cli) -> Result<i32, Failure> {
    // Located only by the commands that use it, so `summary` and `check` work without a
    // cache directory.
    let cache_dir = cli.cache_dir.clone();
    let cache = || Cache::locate(cache_dir.as_deref());
    let je = cli.json_errors;
    match cli.command {
        Cmd::Compile(a) => {
            let o = CompileOptions {
                source: a.source,
                name: a.name,
                model_name: a.model_name,
                include_paths: a.build.include_paths,
                allow_undefined: a.allow_undefined,
                profile: if a.build.dev {
                    Profile::Dev
                } else {
                    Profile::Release
                },
                offline: a.build.offline,
                force: a.force,
                verbose: a.verbose,
            };
            if a.standalone && a.output.is_none() {
                return Err(Failure::usage("--standalone needs --output <path>"));
            }
            let art = compile::compile(&o, &cache()?)?;
            eprintln!(
                "{} {} ({})",
                if art.cached { "cached" } else { "built" },
                art.name,
                art.entry_dir.display()
            );
            let path = match &a.output {
                Some(dest) => compile::export(&art, dest, a.overwrite)?,
                None => art.executable.clone(),
            };
            if a.json {
                let mut m = art.manifest.clone();
                m["exported"] = Value::String(path_arg(&path));
                println!("{}", serde_json::to_string_pretty(&m).expect("serializes"));
            } else {
                println!("{}", path.display());
            }
            Ok(0)
        }
        Cmd::Check(a) => {
            let o = CompileOptions {
                source: a.source,
                name: None,
                model_name: None,
                include_paths: a.include_paths,
                allow_undefined: a.allow_undefined,
                profile: Profile::Release,
                offline: true,
                force: false,
                verbose: false,
            };
            let c = compile::check(&o)?;
            for w in &c.warnings {
                eprintln!("{w}");
            }
            if a.info {
                println!("{}", c.info);
            } else {
                eprintln!("{}: ok", o.source.display());
            }
            Ok(0)
        }
        Cmd::Inspect(a) => inspect(&a, &cache()?, je),
        Cmd::Sample(a) => run_method(&a.common, &cache()?, je, |name| {
            if a.chains == 0 {
                return Err(Failure::usage("--chains must be at least 1"));
            }
            let cpus = std::thread::available_parallelism().map_or(1, usize::from);
            let parallel = a.parallel_chains.unwrap_or(a.chains.min(cpus));
            if parallel == 0 {
                return Err(Failure::usage("--parallel-chains must be at least 1"));
            }
            let prefix = a
                .output_prefix
                .clone()
                .unwrap_or_else(|| format!("{name}_chain"));
            // One plain file name component (no separators, drive prefixes, or `..`).
            let plain = {
                let mut c = Path::new(&prefix).components();
                matches!(
                    (c.next(), c.next()),
                    (Some(std::path::Component::Normal(_)), None)
                )
            };
            if !plain || prefix.contains(['/', '\\', ':']) {
                return Err(Failure::usage("--output-prefix must be a plain file name"));
            }
            let mut v = vec![
                "sample".to_string(),
                format!("num_samples={}", a.draws),
                format!("num_warmup={}", a.warmup),
                format!("thin={}", a.thin),
                format!("save_warmup={}", u8::from(a.save_warmup)),
                format!("num_chains={}", a.chains),
                "adapt".into(),
                format!("engaged={}", u8::from(!a.no_adapt)),
                format!("delta={}", a.adapt_delta),
                format!("save_metric={}", u8::from(a.save_metric)),
            ];
            if a.fixed_param
                && (a.metric_file.is_some()
                    || a.metric.is_some()
                    || a.engine.is_some()
                    || a.max_treedepth.is_some()
                    || a.step_size.is_some()
                    || a.step_size_jitter.is_some()
                    || a.int_time.is_some()
                    || a.save_metric)
            {
                return Err(Failure::usage(
                    "--fixed-param takes no engine, metric, tree depth, step size, \
                     integration time, or --save-metric options",
                ));
            }
            if a.save_metric && a.no_adapt {
                return Err(Failure::usage(
                    "--save-metric needs adaptation; it cannot be combined with --no-adapt",
                ));
            }
            if a.fixed_param {
                v.push("algorithm=fixed_param".into());
            } else {
                v.push("algorithm=hmc".into());
                match a.engine.unwrap_or(EngineArg::Nuts) {
                    EngineArg::Nuts => {
                        if a.int_time.is_some() {
                            return Err(Failure::usage("--int-time applies to --engine static"));
                        }
                        v.push("engine=nuts".into());
                        v.push(format!("max_depth={}", a.max_treedepth.unwrap_or(10)));
                    }
                    EngineArg::Static => {
                        if a.max_treedepth.is_some() {
                            return Err(Failure::usage("--max-treedepth applies to --engine nuts"));
                        }
                        v.push("engine=static".into());
                        if let Some(t) = a.int_time {
                            v.push(format!("int_time={t}"));
                        }
                    }
                }
                v.push(format!(
                    "metric={}",
                    match a.metric.unwrap_or(MetricArg::Diag) {
                        MetricArg::Diag => "diag_e",
                        MetricArg::Dense => "dense_e",
                        MetricArg::Unit => "unit_e",
                    }
                ));
                if let Some(f) = &a.metric_file {
                    v.push(format!("metric_file={}", path_arg(f)));
                }
                if let Some(s) = a.step_size {
                    v.push(format!("stepsize={s}"));
                }
                if let Some(j) = a.step_size_jitter {
                    v.push(format!("stepsize_jitter={j}"));
                }
            }
            v.push(format!("num_threads={parallel}"));
            Ok((v, a.output.join(format!("{prefix}.csv"))))
        }),
        Cmd::Optimize(a) => run_method(&a.common, &cache()?, je, |name| {
            let alg = match a.algorithm {
                OptimizerArg::Lbfgs => "lbfgs",
                OptimizerArg::Bfgs => "bfgs",
                OptimizerArg::Newton => "newton",
            };
            let mut v = vec!["optimize".to_string(), format!("algorithm={alg}")];
            match a.algorithm {
                OptimizerArg::Lbfgs => v.extend(tol_args(&a.tol, true)),
                OptimizerArg::Bfgs => {
                    if a.tol.history_size.is_some() {
                        return Err(Failure::usage("--history-size applies to lbfgs"));
                    }
                    v.extend(tol_args(&a.tol, false));
                }
                OptimizerArg::Newton => {
                    if any_tol(&a.tol) {
                        return Err(Failure::usage(
                            "newton takes no line search or tolerance options",
                        ));
                    }
                }
            }
            v.push(format!("jacobian={}", u8::from(a.jacobian)));
            if let Some(i) = a.iter {
                v.push(format!("iter={i}"));
            }
            if a.json {
                v.push("--json".into());
            }
            Ok((
                v,
                a.output
                    .clone()
                    .unwrap_or_else(|| default_output(name, "optimize")),
            ))
        }),
        Cmd::Variational(a) => run_method(&a.common, &cache()?, je, |name| {
            let mut v = vec![
                "variational".to_string(),
                format!(
                    "algorithm={}",
                    match a.algorithm {
                        FamilyArg::Meanfield => "meanfield",
                        FamilyArg::Fullrank => "fullrank",
                    }
                ),
            ];
            let mut opt = |k: &str, x: Option<String>| {
                if let Some(x) = x {
                    v.push(format!("{k}={x}"));
                }
            };
            opt("iter", a.iter.map(|x| x.to_string()));
            opt("grad_samples", a.grad_samples.map(|x| x.to_string()));
            opt("elbo_samples", a.elbo_samples.map(|x| x.to_string()));
            opt("eta", a.eta.map(|x| x.to_string()));
            opt("tol_rel_obj", a.tol_rel_obj.map(|x| x.to_string()));
            opt("eval_elbo", a.eval_elbo.map(|x| x.to_string()));
            opt("output_samples", a.draws.map(|x| x.to_string()));
            v.push("adapt".into());
            v.push(format!("engaged={}", u8::from(!a.no_adapt)));
            if let Some(i) = a.adapt_iter {
                v.push(format!("iter={i}"));
            }
            Ok((
                v,
                a.output
                    .clone()
                    .unwrap_or_else(|| default_output(name, "variational")),
            ))
        }),
        Cmd::Pathfinder(a) => run_method(&a.common, &cache()?, je, |name| {
            let mut v = vec!["pathfinder".to_string()];
            let mut opt = |k: &str, x: Option<usize>| {
                if let Some(x) = x {
                    v.push(format!("{k}={x}"));
                }
            };
            opt("num_paths", a.num_paths);
            opt("num_draws", a.draws);
            opt("num_psis_draws", a.psis_draws);
            opt("max_lbfgs_iters", a.max_lbfgs_iters);
            opt("num_elbo_draws", a.num_elbo_draws);
            v.push(format!("psis_resample={}", u8::from(!a.no_psis_resample)));
            v.extend(tol_args(&a.tol, true));
            Ok((
                v,
                a.output
                    .clone()
                    .unwrap_or_else(|| default_output(name, "pathfinder")),
            ))
        }),
        Cmd::Laplace(a) => run_method(&a.common, &cache()?, je, |name| {
            let mut v = vec![
                "laplace".to_string(),
                format!("jacobian={}", u8::from(!a.no_jacobian)),
            ];
            if let Some(m) = &a.mode {
                v.push(format!("mode={}", path_arg(m)));
            }
            if let Some(d) = a.draws {
                v.push(format!("draws={d}"));
            }
            Ok((
                v,
                a.output
                    .clone()
                    .unwrap_or_else(|| default_output(name, "laplace")),
            ))
        }),
        Cmd::GenerateQuantities(a) => run_method(&a.common, &cache()?, je, |name| {
            let files: Vec<String> = a.draws.iter().map(|p| path_arg(p)).collect();
            if files.iter().any(|f| f.contains(',')) {
                return Err(Failure::usage(
                    "draw file names must not contain commas (CmdStan's fitted_params list \
                     separator)",
                ));
            }
            let v = vec![
                "generate_quantities".to_string(),
                format!("fitted_params={}", files.join(",")),
            ];
            Ok((
                v,
                a.output
                    .clone()
                    .unwrap_or_else(|| default_output(name, "gq")),
            ))
        }),
        Cmd::LogProb(a) => run_method(&a.common, &cache()?, je, |name| {
            if a.parameters.is_none() && a.constrained.is_none() {
                return Err(Failure::usage("give --parameters or --constrained"));
            }
            if a.parameters.is_some() && a.constrained.is_some() {
                return Err(Failure::usage(
                    "give either --parameters or --constrained, not both",
                ));
            }
            let mut v = vec![
                "log_prob".to_string(),
                format!("jacobian={}", u8::from(!a.no_jacobian)),
            ];
            if let Some(p) = &a.parameters {
                v.push(format!("unconstrained_params={}", path_arg(p)));
            }
            if let Some(p) = &a.constrained {
                v.push(format!("constrained_params={}", path_arg(p)));
            }
            v.push("--json".into());
            Ok((
                v,
                a.output
                    .clone()
                    .unwrap_or_else(|| default_output(name, "log_prob")),
            ))
        }),
        Cmd::Diagnose(a) => run_method(&a.common, &cache()?, je, |name| {
            let mut v = vec!["diagnose".to_string(), "test=gradient".to_string()];
            if let Some(e) = a.epsilon {
                v.push(format!("epsilon={e}"));
            }
            if let Some(e) = a.error {
                v.push(format!("error={e}"));
            }
            if let Some(p) = &a.parameters {
                v.push(format!("--unconstrained={}", path_arg(p)));
            }
            if a.json {
                v.push("--json".into());
            }
            Ok((
                v,
                a.output
                    .clone()
                    .unwrap_or_else(|| default_output(name, "diagnose")),
            ))
        }),
        Cmd::Summary(a) => {
            let (out, flagged) = summary::summary(&summary::SummaryOptions {
                inputs: a.inputs,
                allow_partial: a.allow_partial,
                format: match a.format {
                    FormatArg::Table => summary::Format::Table,
                    FormatArg::Json => summary::Format::Json,
                    FormatArg::Csv => summary::Format::Csv,
                },
                sig_figs: a.sig_figs.clamp(1, 18),
            })?;
            print!("{out}");
            if a.strict && flagged {
                eprintln!("error [diagnostics]: a convergence or sampler check failed (--strict)");
                return Ok(ExitCode::Diagnostics.code());
            }
            Ok(0)
        }
        Cmd::InspectRun(a) => {
            print!("{}", summary::inspect_run(&a.path, a.json)?);
            Ok(0)
        }
        Cmd::Run(a) => {
            let m = exec::resolve(&a.model, &cache()?, &fly(&a.build))?;
            let mut args = a.args;
            if je && !args.iter().any(|x| x == "--json-errors") {
                args.push("--json-errors".into());
            }
            exec::run(&m.executable, &args)
        }
        Cmd::Cache(c) => cache_cmd(c, &cache()?),
        Cmd::Version => {
            println!("cmdstan-rs {}", env!("CARGO_PKG_VERSION"));
            println!("CmdStan reference: {}", compile::CMDSTAN_REFERENCE);
            println!("stanc-rs: {}", compile::STANC_RS_REV);
            println!("stan-model-rs: {}", compile::STAN_MODEL_RS_REV);
            println!("stanmath-rs: {}", compile::STANMATH_RS_REV);
            println!("stan-services, stan-inference: sibling path dependencies");
            match compile::rustc_version(Path::new(".")) {
                Ok(v) => println!("{}", v.lines().next().unwrap_or("rustc: unknown")),
                Err(_) => println!("rustc: not found (needed to compile models)"),
            }
            match compile::runtime_root() {
                Ok(r) => println!("runtime source: {}", r.display()),
                Err(e) => println!("runtime source: {}", e.message),
            }
            match cache() {
                Ok(c) => println!("cache: {}", c.root.display()),
                Err(e) => println!("cache: {}", e.message),
            }
            Ok(0)
        }
    }
}

fn inspect(a: &InspectArgs, cache: &Cache, json_errors: bool) -> Result<i32, Failure> {
    let m = exec::resolve(&a.model, cache, &fly(&a.build))?;
    let mut args = vec!["info".to_string()];
    if let Some(d) = &a.data {
        args.push("data".into());
        args.push(format!("file={}", path_arg(d)));
    }
    if json_errors {
        args.push("--json-errors".into());
    }
    let (code, stdout) = exec::run_capture(&m.executable, &args)?;
    if code != 0 {
        return Ok(code);
    }
    // The model's info JSON is its last line (a Stan print() may write before it).
    let last = stdout
        .lines()
        .rev()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("");
    if a.json {
        println!("{last}");
        return Ok(0);
    }
    let v: Value = serde_json::from_str(last).map_err(|e| {
        Failure::new(
            ExitCode::Internal,
            format!("the model's info output is not JSON: {e}"),
        )
    })?;
    let b = &v["build"];
    let s = |x: &Value| {
        x.as_str()
            .map(str::to_string)
            .unwrap_or_else(|| x.to_string())
    };
    println!("model        {}", s(&b["name"]));
    println!("stanc name   {}", s(&b["stanc_model_name"]));
    println!("source       {}", s(&b["source"]["path"]));
    println!("source hash  {}", s(&b["source"]["sha256"]));
    println!("compiler     {}", s(&b["compiler"]));
    println!("runtime      {}", s(&b["runtime"]));
    println!("rustc        {}", s(&b["rustc"]));
    println!("target       {} ({})", s(&b["target"]), s(&b["profile"]));
    println!("executable   {}", m.executable.display());
    for (block, title) in [
        ("inputs", "data"),
        ("parameters", "parameters"),
        ("transformed parameters", "transformed parameters"),
        ("generated quantities", "generated quantities"),
    ] {
        if let Some(vars) = v["stanc_info"][block].as_object() {
            if vars.is_empty() {
                continue;
            }
            println!("{title}:");
            for (n, d) in vars {
                println!(
                    "  {n}: {} (dimensions {})",
                    s(&d["type"]),
                    s(&d["dimensions"])
                );
            }
        }
    }
    let model = &v["model"];
    if model["resolved"] == true {
        println!(
            "unconstrained dimension {}; output columns: {}",
            model["num_unconstrained"],
            model["outputs"]
                .as_array()
                .map(|a| a
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join(", "))
                .unwrap_or_default()
        );
    } else {
        println!(
            "dimensions unresolved: {} (pass --data)",
            s(&model["reason"])
        );
    }
    println!(
        "methods: {}",
        v["methods"]
            .as_array()
            .map(|a| a
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", "))
            .unwrap_or_default()
    );
    Ok(0)
}

fn cache_cmd(c: CacheCmd, cache: &Cache) -> Result<i32, Failure> {
    match c {
        CacheCmd::Dir => {
            println!("{}", cache.root.display());
        }
        CacheCmd::List => {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            let entries = cache.entries();
            if entries.is_empty() {
                eprintln!("no compiled models in {}", cache.root.display());
            }
            for e in entries {
                let age = (now - e.built()).max(0.0);
                let age = if age < 3600.0 {
                    format!("{:.0} min ago", age / 60.0)
                } else if age < 86400.0 {
                    format!("{:.1} h ago", age / 3600.0)
                } else {
                    format!("{:.1} days ago", age / 86400.0)
                };
                println!(
                    "{:<24} {}  {:<14} {}",
                    e.name(),
                    &e.key()[..e.key().len().min(16)],
                    age,
                    e.source()
                );
            }
        }
        CacheCmd::Inspect { model } => {
            let e = cache.find(&model)?;
            eprintln!("entry {}", e.dir.display());
            println!(
                "{}",
                serde_json::to_string_pretty(&e.manifest).expect("serializes")
            );
        }
        CacheCmd::Clear { name, all } => {
            if all && name.is_some() {
                return Err(Failure::usage("--all and --name cannot be combined"));
            }
            let removed = cache.clear(name.as_deref(), all)?;
            for r in &removed {
                eprintln!("removed {}", r.display());
            }
            if removed.is_empty() {
                eprintln!("nothing to remove");
            }
        }
    }
    Ok(0)
}
