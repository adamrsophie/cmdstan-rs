//! Translation of parsed arguments into stan-services configurations, method dispatch, and
//! reporting.

use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use stan_inference::optimize::Termination;
use stan_inference::{
    AdviConfig, Family, HmcSteps, Matrix, Metric, OptimizeConfig, Optimizer, PathfinderConfig,
};
use stan_model_rs::{Data, LogDensityOptions, Model, Outputs};
use stan_model_rs_stanc::{Stanc, convert_data};
use stan_services::{
    Adaptation, AdviServiceConfig, CancellationToken, ChainStatus, ChainValues, CsvOutput, GqInput,
    GradientCheckConfig, GradientPoints, InitStrategy, LaplaceMode, LaplaceServiceConfig,
    MetricType, OptimizationConfig, OutputBlocks, OutputConfig, OverwritePolicy,
    PathfinderServiceConfig, Point, PointCheck, RunOptions, RunStatus, SamplerAlgorithm,
    SamplingConfig, SamplingService, read_stan_csv,
};

use crate::args::{self, Extensions, Parsed};
use crate::exit::{ExitCode, Failure};
use crate::output::{self, Table, num, sha256_file, sibling};
use crate::progress::StderrProgress;
use crate::{BuildInfo, VERSION};

/// Maximum random initialization attempts (Stan's).
const MAX_INIT_ATTEMPTS: usize = 100;

/// Everything shared by the methods of one invocation.
struct Run<'a, M: Model + Sync> {
    model: &'a M,
    parsed: &'a Parsed,
    ext: &'a Extensions,
    build: BuildInfo,
    seed: u64,
    id: u32,
    init: InitStrategy,
    output_file: PathBuf,
    sig_figs: usize,
    start_unix: f64,
    cancel: CancellationToken,
    /// Input files with the SHA-256 of their contents when they were read.
    inputs: std::cell::RefCell<Vec<Value>>,
}

/// Parses the arguments, builds the model, and runs the method.
pub(crate) fn main<M, F>(
    program: &str,
    argv: &[String],
    build: BuildInfo,
    make: &F,
) -> Result<ExitCode, Failure>
where
    M: stanc_rs_runtime::Model + Sync,
    F: Fn(&stanc_rs_runtime::data::Data, u64) -> stanc_rs_runtime::Result<M>,
{
    let (rest, ext) = args::split_extensions(argv)?;
    match rest.first().map(String::as_str) {
        None | Some("help") | Some("-h") => {
            print!("{}", crate::usage(program));
            return Ok(ExitCode::Success);
        }
        Some("help-all") => {
            print!("{}", args::help_all());
            return Ok(ExitCode::Success);
        }
        Some("info") => {
            let parsed = args::parse(&rest[1..])?;
            if parsed.method().is_some() {
                return Err(Failure::usage("info takes only data and random arguments"));
            }
            return info(&parsed, build, make);
        }
        _ => {}
    }
    let parsed = args::parse(&rest)?;
    let Some(method) = parsed.method().map(str::to_string) else {
        return Err(Failure::usage(format!(
            "no method given; run `{program} help` for usage"
        )));
    };
    if ext.unconstrained.is_some() && method != "diagnose" {
        return Err(Failure::usage(
            "--unconstrained applies to diagnose only; log_prob takes unconstrained_params",
        ));
    }
    validate(&parsed, &method)?;
    let start_unix = unix_now();
    let seed = resolve_seed(&parsed)?;
    let data_file = parsed.file("data.file");
    let mut inputs = Vec::new();
    if let Some(d) = &data_file {
        inputs.push(input_record("data", d));
    }
    if let Some(raw) = parsed.get("init").filter(|r| r.parse::<f64>().is_err()) {
        inputs.push(input_record("init", Path::new(raw)));
    }
    let data = load_data(data_file.as_deref())?;
    let model = construct(make, &data, seed)?;
    let init = init_strategy(&parsed)?;
    let sig_figs = match parsed.int("output.sig_figs") {
        None | Some(-1) => 8,
        Some(s) if (1..=18).contains(&s) => s as usize,
        Some(s) => {
            return Err(Failure::usage(format!(
                "output sig_figs={s}: expected 1 to 18, or -1 for the default (8)"
            )));
        }
    };
    let id = parsed.usize("id").unwrap_or(1);
    let id = u32::try_from(id).map_err(|_| Failure::usage(format!("id={id} is too large")))?;
    let cancel = CancellationToken::new();
    // Only sampling can stop cleanly between transitions. The other methods keep the default
    // behavior: Ctrl+C ends the process at once (exit 130), never a silent success.
    if method == "sample" {
        install_interrupt_handler(&cancel);
    }
    let output_file = parsed
        .file("output.file")
        .unwrap_or_else(|| PathBuf::from("output.csv"));
    // The manifest is an output like the others: never replaced without --overwrite. With
    // --overwrite, the previous run's manifest is removed now, so an interrupted run never
    // leaves it beside new files.
    let manifest = sibling(&output_file, "_manifest", "json");
    output::preflight(&manifest, ext.overwrite)?;
    if ext.overwrite {
        let _ = std::fs::remove_file(&manifest);
    }
    let run = Run {
        model: &model,
        parsed: &parsed,
        ext: &ext,
        build,
        seed,
        id,
        init,
        output_file,
        sig_figs,
        start_unix,
        cancel,
        inputs: std::cell::RefCell::new(inputs),
    };
    match method.as_str() {
        "sample" => sample(&run),
        "optimize" => optimize(&run),
        "variational" => variational(&run),
        "pathfinder" => pathfinder(&run),
        "laplace" => laplace(&run),
        "generate_quantities" => generate_quantities(&run),
        "log_prob" => log_prob(&run),
        "diagnose" => diagnose(&run),
        other => Err(Failure::new(
            ExitCode::Internal,
            format!("method {other} is in the grammar but has no implementation"),
        )),
    }
}

/// An input file and the SHA-256 of its contents now (before the method runs).
fn input_record(role: &str, path: &Path) -> Value {
    json!({
        "role": role,
        "path": path.display().to_string(),
        "sha256": sha256_file(path).ok(),
    })
}

/// Range and consistency checks of CmdStan 2.40 that the grammar's types do not express.
fn validate(p: &Parsed, method: &str) -> Result<(), Failure> {
    let bad = |m: String| Err(Failure::usage(m));
    if let Some(s) = p.int("random.seed")
        && s != -1
        && !(0..=i64::from(u32::MAX)).contains(&s)
    {
        return bad(format!(
            "random seed={s}: expected 0 to {}, or -1 for a seed from the clock",
            u32::MAX
        ));
    }
    if let Some(t) = p.int("num_threads")
        && t != -1
        && t < 1
    {
        return bad(format!(
            "num_threads={t}: expected a positive number, or -1"
        ));
    }
    if method != "sample" {
        return Ok(());
    }
    let s = "method.sample";
    if let Some(d) = p.real(&format!("{s}.adapt.delta"))
        && !(0.1..1.0).contains(&d)
    {
        return bad(format!(
            "adapt delta={d}: expected at least 0.1 and less than 1"
        ));
    }
    let h = format!("{s}.algorithm.hmc");
    if let Some(j) = p.real(&format!("{h}.stepsize_jitter"))
        && !(0.0..=1.0).contains(&j)
    {
        return bad(format!("stepsize_jitter={j}: expected 0 to 1"));
    }
    if let Some(e) = p.real(&format!("{h}.stepsize"))
        && !(e > 0.0)
    {
        return bad(format!("stepsize={e}: expected a positive number"));
    }
    let chains = p.usize(&format!("{s}.num_chains")).unwrap_or(1);
    // stan-services' limit, checked before any per-chain allocation.
    if chains > 1_000_000 {
        return bad(format!("num_chains={chains}: at most 1000000"));
    }
    let fixed = p.get(&format!("{s}.algorithm")) == Some("fixed_param");
    let engaged = p.flag(&format!("{s}.adapt.engaged")).unwrap_or(true);
    let save_metric = p.flag(&format!("{s}.adapt.save_metric")).unwrap_or(false);
    if save_metric && (!engaged || fixed) {
        return bad(
            "Saving the adapted metric (save_metric=1) can only be enabled when adaptation is \
             engaged (adapt engaged=1) with algorithm=hmc"
                .into(),
        );
    }
    if fixed && p.get(&format!("{h}.metric_file")).is_some() {
        return bad("metric_file has no effect with algorithm=fixed_param".into());
    }
    if !fixed && engaged && p.usize(&format!("{s}.num_warmup")) == Some(0) {
        return bad(
            "The number of warmup samples (num_warmup) must be greater than zero if \
             adaptation is enabled (adapt engaged=1)"
                .into(),
        );
    }
    Ok(())
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// The seed: the given one, or (CmdStan's default `seed=-1`) one from the clock. The seed used
/// is always recorded in the outputs.
fn resolve_seed(parsed: &Parsed) -> Result<u64, Failure> {
    match parsed.int("random.seed") {
        Some(s) if s >= 0 => Ok(s as u64),
        None | Some(-1) => {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.subsec_nanos() ^ (d.as_secs() as u32))
                .unwrap_or(0);
            Ok(u64::from(nanos))
        }
        Some(s) => Err(Failure::usage(format!(
            "random seed={s}: expected a nonnegative integer, or -1 for a seed from the clock"
        ))),
    }
}

/// The first variable defined twice at the top level of a JSON object. A JSON reader keeps
/// only one of the values, silently; Stan rejects a redefinition.
fn duplicate_variable(text: &str) -> Option<String> {
    use serde::de::{Deserializer, IgnoredAny, MapAccess, Visitor};
    struct Keys;
    impl<'de> Visitor<'de> for Keys {
        type Value = Option<String>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a JSON object")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut seen = std::collections::HashSet::new();
            let mut dup = None;
            while let Some(k) = map.next_key::<String>()? {
                map.next_value::<IgnoredAny>()?;
                if dup.is_none() && !seen.insert(k.clone()) {
                    dup = Some(k);
                }
            }
            Ok(dup)
        }
    }
    serde_json::Deserializer::from_str(text)
        .deserialize_map(Keys)
        .ok()
        .flatten()
}

/// Reads a JSON file of variables, refusing duplicate definitions.
fn read_json_vars(path: &Path, what: &str, code: ExitCode) -> Result<Data, Failure> {
    let text = std::fs::read_to_string(path).map_err(|e| Failure::input(what, path, &e))?;
    if let Some(k) = duplicate_variable(&text) {
        return Err(Failure::new(
            code,
            format!("{what} {}: variable {k} is defined twice", path.display()),
        ));
    }
    Data::from_json_file(path)
        .map_err(|e| Failure::new(code, format!("{what} {}: {e}", path.display())))
}

fn load_data(path: Option<&Path>) -> Result<Data, Failure> {
    let Some(path) = path else {
        return Ok(Data::empty());
    };
    if !path.exists() {
        return Err(Failure::new(
            ExitCode::NotFound,
            format!("data file {} does not exist", path.display()),
        ));
    }
    read_json_vars(path, "data file", ExitCode::Data)
}

fn construct<M, F>(make: &F, data: &Data, seed: u64) -> Result<Stanc<M>, Failure>
where
    M: stanc_rs_runtime::Model,
    F: Fn(&stanc_rs_runtime::data::Data, u64) -> stanc_rs_runtime::Result<M>,
{
    let rt_data = convert_data(data).map_err(|e| Failure::new(ExitCode::Data, e.to_string()))?;
    // A constructor error is a data error (missing, malformed, or constraint-violating data),
    // or a failure of the transformed data block, which only depends on the data and the seed.
    let model = make(&rt_data, seed)
        .map_err(|e| Failure::new(ExitCode::Data, format!("constructing the model: {e}")))?;
    Ok(Stanc::new(model))
}

/// CmdStan's `init`: a radius (`0` for zeros) or a JSON file of constrained values.
fn init_strategy(parsed: &Parsed) -> Result<InitStrategy, Failure> {
    let raw = parsed.get("init").unwrap_or("2");
    if let Ok(r) = raw.parse::<f64>() {
        if !(r.is_finite() && r >= 0.0) {
            return Err(Failure::usage(format!(
                "init={raw}: a radius must be finite and nonnegative"
            )));
        }
        return Ok(if r == 0.0 {
            InitStrategy::Zero
        } else {
            InitStrategy::Random { radius: r }
        });
    }
    let path = Path::new(raw);
    if !path.exists() {
        return Err(Failure::new(
            ExitCode::NotFound,
            format!("init file {raw} does not exist"),
        ));
    }
    let values = read_json_vars(path, "init file", ExitCode::Init)?;
    // Parameters missing from the file are drawn on (-2, 2), as in CmdStan.
    Ok(InitStrategy::Constrained {
        values,
        radius: 2.0,
    })
}

fn install_interrupt_handler(token: &CancellationToken) {
    let t = token.clone();
    // A second interrupt ends the process at once.
    let result = ctrlc::set_handler(move || {
        if t.is_cancelled() {
            std::process::exit(ExitCode::Interrupted.code());
        }
        eprintln!(
            "interrupt or termination signal received: stopping after the current transitions"
        );
        t.cancel();
    });
    if let Err(e) = result {
        eprintln!("warning: no interrupt handler ({e}); Ctrl+C ends the process at once");
    }
}

/// The pinned stan-model-rs-stanc adapter cannot map flat constrained values of complex or
/// tuple parameters back to the model correctly (it drops imaginary parts). Inputs given as
/// flat values (Stan CSV draws or modes) are refused for such models; JSON inputs work.
fn reject_flat_complex<M: Model + Sync>(run: &Run<'_, M>, what: &str) -> Result<(), Failure> {
    let info: Value = serde_json::from_str(run.build.stanc_info_json).unwrap_or(Value::Null);
    let bad = info["parameters"].as_object().is_some_and(|ps| {
        ps.values().any(|d| {
            let t = d["type"].as_str().unwrap_or("");
            t.contains("complex") || t.contains("tuple")
        })
    });
    if bad {
        return Err(Failure::usage(format!(
            "this model has complex or tuple parameters, which cannot yet be read back from \
             {what} without losing values; give constrained values as JSON instead"
        )));
    }
    Ok(())
}

/// Output blocks evaluated for each draw.
fn blocks() -> OutputBlocks {
    OutputBlocks::default()
}

impl<M: Model + Sync> Run<'_, M> {
    fn overwrite(&self) -> bool {
        self.ext.overwrite
    }

    fn progress(&self) -> StderrProgress {
        StderrProgress::new(
            self.parsed.usize("output.refresh").unwrap_or(100),
            self.ext.quiet,
        )
    }

    fn manifest_path(&self) -> PathBuf {
        sibling(&self.output_file, "_manifest", "json")
    }

    /// Configuration comments common to every table.
    fn config_lines(&self, method: &str) -> Vec<(String, String)> {
        let mut v = vec![
            ("model".to_string(), self.model.info().name.clone()),
            ("method".to_string(), method.to_string()),
            ("producer".to_string(), format!("cmdstan-rs {VERSION}")),
            ("seed".to_string(), self.seed.to_string()),
            ("id".to_string(), self.id.to_string()),
        ];
        for (k, val) in self.parsed.values() {
            v.push((k.clone(), val.clone()));
        }
        v.push(("sig_figs".to_string(), self.sig_figs.to_string()));
        v
    }

    /// Writes the run manifest: provenance and everything needed to repeat the run.
    fn write_manifest(
        &self,
        method: &str,
        exit: ExitCode,
        files: &[PathBuf],
        details: Value,
        warnings: &[String],
    ) -> Result<(), Failure> {
        let inputs = self.inputs.borrow().clone();
        let role = |r: &str| {
            inputs
                .iter()
                .find(|i| i["role"] == r)
                .cloned()
                .unwrap_or(Value::Null)
        };
        let data = role("data");
        let init = match self.parsed.get("init") {
            Some(raw) if raw.parse::<f64>().is_err() => role("init"),
            other => json!(other.unwrap_or("2")),
        };
        let build: Value = serde_json::from_str(self.build.build_json).unwrap_or(Value::Null);
        let manifest = json!({
            "manifest_version": 1,
            "producer": format!("cmdstan-runner {VERSION}"),
            "build": build,
            "model": self.model.info().name,
            "method": method,
            "arguments": self.parsed.values(),
            "seed": self.seed,
            "id": self.id,
            "data": data,
            "init": init,
            "inputs": inputs,
            "sig_figs": self.sig_figs,
            "start_time_unix": num(self.start_unix),
            "end_time_unix": num(unix_now()),
            "exit": {"code": exit.code(), "status": exit.name()},
            "files": files.iter().map(|f| f.display().to_string()).collect::<Vec<_>>(),
            "warnings": warnings,
            "details": details,
        });
        // The manifest is written last; it was checked to be writable before the run.
        output::write_json(&self.manifest_path(), self.overwrite(), &manifest)
    }

    /// Records an input file (hashing its contents now, as it is read).
    fn record_input(&self, role: &str, path: &Path) {
        self.inputs.borrow_mut().push(input_record(role, path));
    }

    fn report_warnings(&self, warnings: &[String]) {
        for w in warnings {
            eprintln!("warning: {w}");
        }
    }
}

// ------------------------------------------------------------------------------------------
// info

fn info<M, F>(parsed: &Parsed, build: BuildInfo, make: &F) -> Result<ExitCode, Failure>
where
    M: stanc_rs_runtime::Model + Sync,
    F: Fn(&stanc_rs_runtime::data::Data, u64) -> stanc_rs_runtime::Result<M>,
{
    let build_v: Value = serde_json::from_str(build.build_json).unwrap_or(Value::Null);
    let stanc_v: Value = serde_json::from_str(build.stanc_info_json).unwrap_or(Value::Null);
    let data_file = parsed.file("data.file");
    let seed = parsed.int("random.seed").map_or(0, |s| s.max(0) as u64);
    // Dimensions that depend on the data are known only once the model is constructed.
    let resolved = load_data(data_file.as_deref()).and_then(|d| construct(make, &d, seed));
    // Data given explicitly must be valid; without data, unresolved dimensions are reported.
    if data_file.is_some()
        && let Err(e) = resolved
    {
        return Err(e);
    }
    let model_v = match &resolved {
        Ok(m) => json!({
            "resolved": true,
            "name": m.info().name,
            "num_unconstrained": m.num_unconstrained(),
            "parameters": m.param_names(Outputs::PARAMETERS),
            "outputs": m.param_names(Outputs::ALL),
            "unconstrained_parameters": m.unconstrained_param_names(),
            "metadata": serde_json::from_str::<Value>(&m.info().to_json()).unwrap_or(Value::Null),
        }),
        Err(e) => json!({
            "resolved": false,
            "reason": e.message,
        }),
    };
    let v = json!({
        "build": build_v,
        "stanc_info": stanc_v,
        "model": model_v,
        "methods": ["sample", "optimize", "variational", "pathfinder", "laplace",
                    "generate_quantities", "log_prob", "diagnose"],
        "notes": [
            "stanc-rs models report no Hessian: Newton optimization and Laplace use finite-difference Hessians of the gradient",
            "the stan-model-rs-stanc adapter reports every variable as unconstrained real values with its declared shape",
        ],
    });
    // One line, printed last: a Stan print() in transformed data writes to stdout before it,
    // and readers take the last line.
    println!(
        "{}",
        serde_json::to_string(&v).expect("JSON values serialize")
    );
    Ok(ExitCode::Success)
}

// ------------------------------------------------------------------------------------------
// sample

fn read_metric_file(path: &Path, metric: MetricType) -> Result<Metric, Failure> {
    let text =
        std::fs::read_to_string(path).map_err(|e| Failure::input("metric_file", path, &e))?;
    let v: Value = serde_json::from_str(&text)
        .map_err(|e| Failure::usage(format!("metric_file {}: {e}", path.display())))?;
    let bad = |m: &str| Failure::usage(format!("metric_file {}: {m}", path.display()));
    let inv = v
        .get("inv_metric")
        .ok_or_else(|| bad("no inv_metric entry"))?;
    let reals = |x: &Value| -> Result<Vec<f64>, Failure> {
        x.as_array()
            .ok_or_else(|| bad("inv_metric must be an array"))?
            .iter()
            .map(|e| {
                e.as_f64()
                    .ok_or_else(|| bad("inv_metric entries must be numbers"))
            })
            .collect()
    };
    match metric {
        MetricType::Unit => Err(Failure::usage(
            "metric_file is given but the metric is unit_e, which has no free values",
        )),
        MetricType::Diagonal => {
            let d = reals(inv)?;
            Metric::diag(d).map_err(|e| bad(&e.to_string()))
        }
        MetricType::Dense => {
            let rows = inv
                .as_array()
                .ok_or_else(|| bad("a dense inv_metric must be an array of rows"))?;
            let mut flat = Vec::new();
            for r in rows {
                let r = reals(r)?;
                if r.len() != rows.len() {
                    return Err(bad("a dense inv_metric must be square"));
                }
                flat.extend(r);
            }
            let m = Matrix::from_row_major(flat).ok_or_else(|| bad("not a square matrix"))?;
            Metric::dense(m).map_err(|e| bad(&e.to_string()))
        }
    }
}

fn metric_json(m: &Metric) -> Value {
    match m {
        Metric::Unit(n) => json!(vec![1.0; *n]),
        Metric::Diag(d) => json!(d.iter().map(|&x| num(x)).collect::<Vec<_>>()),
        Metric::Dense { inv, .. } => {
            let n = m.dim();
            json!(
                (0..n)
                    .map(|i| (0..n).map(|j| num(inv.get(i, j))).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            )
        }
    }
}

fn sampling_config<M: Model + Sync>(run: &Run<'_, M>) -> Result<(SamplingConfig, bool), Failure> {
    let p = run.parsed;
    let s = "method.sample";
    let chains = p.usize(&format!("{s}.num_chains")).unwrap_or(1);
    let threads = match p.int("num_threads") {
        Some(-1) => std::thread::available_parallelism().map_or(1, usize::from),
        Some(t) => t.max(1) as usize,
        None => 1,
    };
    let algorithm = match p.get(&format!("{s}.algorithm")).unwrap_or("hmc") {
        "fixed_param" => SamplerAlgorithm::FixedParam,
        _ => {
            let h = format!("{s}.algorithm.hmc");
            match p.get(&format!("{h}.engine")).unwrap_or("nuts") {
                "static" => SamplerAlgorithm::StaticHmc(HmcSteps::IntegrationTime(
                    p.real(&format!("{h}.engine.static.int_time"))
                        .unwrap_or(2.0 * std::f64::consts::PI),
                )),
                _ => SamplerAlgorithm::Nuts {
                    max_depth: p
                        .usize(&format!("{h}.engine.nuts.max_depth"))
                        .unwrap_or(10)
                        .try_into()
                        .map_err(|_| Failure::usage("max_depth is too large"))?,
                    max_delta_h: 1000.0,
                },
            }
        }
    };
    let h = format!("{s}.algorithm.hmc");
    let metric = match p.get(&format!("{h}.metric")).unwrap_or("diag_e") {
        "unit_e" => MetricType::Unit,
        "dense_e" => MetricType::Dense,
        _ => MetricType::Diagonal,
    };
    let initial_inv_metric = match p.file(&format!("{h}.metric_file")) {
        Some(f) => {
            run.record_input("metric_file", &f);
            let m = read_metric_file(&f, metric)?;
            if m.dim() != run.model.num_unconstrained() {
                return Err(Failure::usage(format!(
                    "metric_file {} has dimension {}, the model has {} unconstrained parameters",
                    f.display(),
                    m.dim(),
                    run.model.num_unconstrained()
                )));
            }
            Some(ChainValues::Shared(m))
        }
        None => None,
    };
    let a = format!("{s}.adapt");
    let d = Adaptation::default();
    let adaptation = Adaptation {
        engaged: p.flag(&format!("{a}.engaged")).unwrap_or(d.engaged),
        gamma: p.real(&format!("{a}.gamma")).unwrap_or(d.gamma),
        kappa: p.real(&format!("{a}.kappa")).unwrap_or(d.kappa),
        t0: p.real(&format!("{a}.t0")).unwrap_or(d.t0),
        init_buffer: p
            .usize(&format!("{a}.init_buffer"))
            .unwrap_or(d.init_buffer),
        term_buffer: p
            .usize(&format!("{a}.term_buffer"))
            .unwrap_or(d.term_buffer),
        window: p.usize(&format!("{a}.window")).unwrap_or(d.window),
    };
    let save_metric = p.flag(&format!("{a}.save_metric")).unwrap_or(false);
    let out = &run.output_file;
    let dir = out
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let prefix = out
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .ok_or_else(|| Failure::usage(format!("output file {} has no name", out.display())))?;
    let ids: Vec<u32> = (0..chains as u32)
        .map(|k| run.id.checked_add(k))
        .collect::<Option<_>>()
        .ok_or_else(|| Failure::usage("chain identifiers overflow"))?;
    let config = SamplingConfig {
        chains,
        parallel_chains: threads.clamp(1, chains.max(1)),
        chain_ids: Some(ids),
        seed: run.seed,
        num_warmup: p.usize(&format!("{s}.num_warmup")).unwrap_or(1000),
        num_samples: p.usize(&format!("{s}.num_samples")).unwrap_or(1000),
        thin: p.usize(&format!("{s}.thin")).unwrap_or(1),
        save_warmup: p.flag(&format!("{s}.save_warmup")).unwrap_or(false),
        algorithm,
        metric,
        initial_inv_metric,
        target_accept: p.real(&format!("{a}.delta")).unwrap_or(0.8),
        adaptation,
        step_size: p.real(&format!("{h}.stepsize")).unwrap_or(1.0),
        step_size_jitter: p.real(&format!("{h}.stepsize_jitter")).unwrap_or(0.0),
        init: ChainValues::Shared(run.init.clone()),
        max_init_attempts: MAX_INIT_ATTEMPTS,
        progress_interval: p.usize("output.refresh").unwrap_or(100),
        output: OutputConfig {
            retain_in_memory: false,
            csv: Some(CsvOutput {
                dir,
                prefix,
                overwrite: if run.overwrite() {
                    OverwritePolicy::Overwrite
                } else {
                    OverwritePolicy::Error
                },
                sig_figs: run.sig_figs,
            }),
            blocks: blocks(),
        },
    };
    Ok((config, save_metric))
}

fn sample<M: Model + Sync>(run: &Run<'_, M>) -> Result<ExitCode, Failure> {
    let (config, save_metric) = sampling_config(run)?;
    if save_metric {
        for id in config.chain_ids() {
            output::preflight(
                &sibling(&run.output_file, &format!("_metric_{id}"), "json"),
                run.overwrite(),
            )?;
        }
    }
    let progress = run.progress();
    let options = RunOptions {
        observer: &progress,
        cancel: Some(&run.cancel),
        sinks: None,
    };
    let result = SamplingService::new(config).run_with(run.model, options)?;
    let mut files: Vec<PathBuf> = result.chains.iter().flat_map(|c| c.files.clone()).collect();
    let mut warnings = result.warnings.clone();
    // A requested output that could not be written fails the run (exit 9), even when every
    // chain completed.
    let mut output_failed = result
        .warnings
        .iter()
        .any(|w| w.contains("metadata file could not be written"));
    if save_metric {
        for c in &result.chains {
            if let Some(a) = &c.adaptation {
                let path = sibling(&run.output_file, &format!("_metric_{}", c.chain_id), "json");
                let v = json!({
                    "stepsize": num(a.step_size),
                    "metric_type": a.metric_type.cmdstan_name(),
                    "inv_metric": metric_json(&a.inv_metric),
                });
                match output::write_json(&path, run.overwrite(), &v) {
                    Ok(()) => files.push(path),
                    Err(e) => {
                        warnings.push(format!("chain {}: {}", c.chain_id, e.message));
                        output_failed = true;
                    }
                }
            }
        }
    }
    let exit = match result.status {
        RunStatus::Completed if output_failed => ExitCode::Output,
        RunStatus::Completed => ExitCode::Success,
        RunStatus::Partial => ExitCode::Partial,
        RunStatus::Cancelled => ExitCode::Interrupted,
        RunStatus::Failed => result
            .first_error()
            .map(|e| ExitCode::from_category(e.category))
            .unwrap_or(ExitCode::Inference),
    };
    let chains: Vec<Value> = result
        .chains
        .iter()
        .map(|c| {
            let d = &c.diagnostics;
            json!({
                "chain_id": c.chain_id,
                "status": c.status.as_str(),
                "error": match &c.status {
                    ChainStatus::Failed(e) => json!({
                        "phase": e.phase.as_str(),
                        "iteration": e.iteration,
                        "category": format!("{:?}", e.category),
                        "message": e.message,
                    }),
                    _ => Value::Null,
                },
                "divergent": d.divergent,
                "max_treedepth_hits": d.max_treedepth_hits,
                "ebfmi": num(d.ebfmi),
                "mean_accept_stat": num(d.mean_accept_stat),
                "output_failures": d.output_failures,
                "step_size": c.adaptation.as_ref().map(|a| num(a.step_size)),
                "warmup_seconds": num(c.timing.warmup),
                "sampling_seconds": num(c.timing.sampling),
                "grad_evals": c.grad_evals_warmup + c.grad_evals_sampling,
                "warnings": c.warnings,
            })
        })
        .collect();
    for c in &result.chains {
        warnings.extend(
            c.warnings
                .iter()
                .map(|w| format!("chain {}: {w}", c.chain_id)),
        );
    }
    let details = json!({
        "status": format!("{:?}", result.status),
        "services_metadata": stan_services::output::metadata::run_metadata_json(&result),
        "chains": chains,
    });
    let manifest = run.write_manifest("sample", exit, &files, details, &warnings);
    if !run.ext.quiet {
        let done = result.completed_chains().count();
        eprintln!(
            "{done} of {} chains completed in {:.1} s",
            result.chains.len(),
            result.metadata.wall_seconds
        );
        for c in &result.chains {
            let d = &c.diagnostics;
            if d.divergent > 0 {
                eprintln!(
                    "warning: chain {}: {} divergent transitions after warmup",
                    c.chain_id, d.divergent
                );
            }
            if d.max_treedepth_hits > 0 {
                eprintln!(
                    "warning: chain {}: {} transitions hit the maximum tree depth",
                    c.chain_id, d.max_treedepth_hits
                );
            }
        }
        for f in &files {
            eprintln!("wrote {}", f.display());
        }
    }
    run.report_warnings(&result.warnings);
    if let Some(e) = result.first_error() {
        eprintln!("first chain failure: chain {}: {}", e.chain_id, e.message);
    }
    manifest?;
    Ok(exit)
}

// ------------------------------------------------------------------------------------------
// optimize

fn optimizer_settings(p: &Parsed, prefix: &str, base: OptimizeConfig) -> OptimizeConfig {
    let r = |k: &str| p.real(&format!("{prefix}.{k}"));
    OptimizeConfig {
        init_alpha: r("init_alpha").unwrap_or(base.init_alpha),
        tol_obj: r("tol_obj").unwrap_or(base.tol_obj),
        tol_rel_obj: r("tol_rel_obj").unwrap_or(base.tol_rel_obj),
        tol_grad: r("tol_grad").unwrap_or(base.tol_grad),
        tol_rel_grad: r("tol_rel_grad").unwrap_or(base.tol_rel_grad),
        tol_param: r("tol_param").unwrap_or(base.tol_param),
        ..base
    }
}

fn optimization_config<M: Model + Sync>(
    run: &Run<'_, M>,
    jacobian_default: bool,
) -> OptimizationConfig {
    let p = run.parsed;
    let o = "method.optimize";
    let base = OptimizeConfig::default();
    let algorithm = p.get(&format!("{o}.algorithm")).unwrap_or("lbfgs");
    let mut optimizer = match algorithm {
        "bfgs" => optimizer_settings(
            p,
            &format!("{o}.algorithm.bfgs"),
            OptimizeConfig {
                algorithm: Optimizer::Bfgs,
                ..base
            },
        ),
        "newton" => OptimizeConfig {
            algorithm: Optimizer::Newton,
            ..base
        },
        _ => {
            let l = format!("{o}.algorithm.lbfgs");
            let history = p.usize(&format!("{l}.history_size")).unwrap_or(5);
            optimizer_settings(
                p,
                &l,
                OptimizeConfig {
                    algorithm: Optimizer::Lbfgs { history },
                    ..base
                },
            )
        }
    };
    if let Some(iter) = p.usize(&format!("{o}.iter")) {
        optimizer.max_iterations = iter;
    }
    OptimizationConfig {
        optimizer,
        jacobian: p.flag(&format!("{o}.jacobian")).unwrap_or(jacobian_default),
        init: run.init.clone(),
        max_init_attempts: MAX_INIT_ATTEMPTS,
        seed: run.seed,
        chain_id: run.id,
        blocks: blocks(),
    }
}

/// Stan's optimizer return code (`stan::optimization::TerminationCondition`) for the
/// `converged__` column.
fn termination_code(t: Termination) -> i32 {
    match t {
        Termination::AbsoluteParameter => 10,
        Termination::AbsoluteObjective | Termination::NewtonConverged => 20,
        Termination::RelativeObjective => 21,
        Termination::AbsoluteGradient => 30,
        Termination::RelativeGradient => 31,
        Termination::MaxIterations => 40,
        Termination::LineSearchFailed => -1,
    }
}

/// Natural log of the determinant of a symmetric positive definite matrix (row major), by
/// Cholesky; `None` if it is not positive definite.
fn log_det_spd(a: &[f64], n: usize) -> Option<f64> {
    let mut l = vec![0.0; n * n];
    let mut log_det = 0.0;
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            if i == j {
                let d = a[i * n + i] - s;
                if !(d > 0.0) {
                    return None;
                }
                l[i * n + i] = d.sqrt();
                log_det += d.ln();
            } else {
                l[i * n + j] = (a[i * n + j] - s) / l[j * n + j];
            }
        }
    }
    Some(log_det)
}

fn optimize<M: Model + Sync>(run: &Run<'_, M>) -> Result<ExitCode, Failure> {
    output::preflight(&run.output_file, run.overwrite())?;
    let config = optimization_config(run, false);
    let newton = config.optimizer.algorithm == Optimizer::Newton;
    let r = stan_services::run_optimization(run.model, &config)?;
    let mut warnings = r.warnings.clone();
    if newton {
        warnings.push(
            "Newton's method used a finite-difference Hessian: stanc-rs models provide no \
             analytic Hessian"
                .into(),
        );
    }
    if r.output_failed {
        warnings.push(
            "transformed parameters or generated quantities failed at the optimum and were \
             written as NaN"
                .into(),
        );
    }
    let mode_kind = if r.jacobian {
        "mode of the density of the unconstrained parameters (jacobian=1)"
    } else {
        "posterior mode of the constrained parameters (jacobian=0); a maximum likelihood \
         estimate only if the model has flat priors"
    };
    // CmdStan 2.40's layout: lp__, converged__ (the optimizer's return code), then the outputs.
    let mut columns = vec!["lp__".to_string(), "converged__".to_string()];
    columns.extend(r.column_names.iter().cloned());
    let mut row = vec![r.log_density, f64::from(termination_code(r.termination))];
    row.extend(&r.values);
    let mut config_lines = run.config_lines("optimize");
    config_lines.push(("jacobian".into(), u8::from(r.jacobian).to_string()));
    let trailer = vec![
        format!("termination = {:?}", r.termination),
        format!("converged = {}", r.converged),
        format!("iterations = {}", r.iterations),
        format!("grad_evals = {}", r.grad_evals),
        format!("estimate = {mode_kind}"),
        format!("Elapsed Time: {:.6} seconds (Total)", r.seconds),
    ];
    output::write_table(
        &run.output_file,
        run.overwrite(),
        &Table {
            config: &config_lines,
            columns: &columns,
            rows: &[row],
            trailer: &trailer,
            sig_figs: run.sig_figs,
        },
    )?;
    let failed = matches!(r.termination, Termination::LineSearchFailed);
    let exit = if failed {
        ExitCode::Inference
    } else {
        ExitCode::Success
    };
    if !r.converged {
        warnings.push(format!(
            "the optimizer stopped without meeting a convergence criterion ({:?})",
            r.termination
        ));
    }
    let result = json!({
        "algorithm": format!("{:?}", r.algorithm),
        "jacobian": r.jacobian,
        "estimate": mode_kind,
        "objective": num(r.log_density),
        "objective_with_constants": num(r.log_density_full),
        "iterations": r.iterations,
        "termination": format!("{:?}", r.termination),
        "converged": r.converged,
        "grad_evals": r.grad_evals,
        "unconstrained": r.theta.iter().map(|&x| num(x)).collect::<Vec<_>>(),
        "columns": r.column_names,
        "values": r.values.iter().map(|&x| num(x)).collect::<Vec<_>>(),
        "seconds": num(r.seconds),
        "note": "a local optimum: nothing shows that it is global for a nonconvex objective",
    });
    if run.ext.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&result).expect("serializes")
        );
    }
    if !run.ext.quiet {
        eprintln!(
            "optimization {:?} after {} iterations: objective {}, {mode_kind}",
            r.termination, r.iterations, r.log_density
        );
        eprintln!("wrote {}", run.output_file.display());
    }
    run.report_warnings(&warnings);
    run.write_manifest(
        "optimize",
        exit,
        std::slice::from_ref(&run.output_file),
        result,
        &warnings,
    )?;
    Ok(exit)
}

// ------------------------------------------------------------------------------------------
// approximations: shared table of draws

fn approximate_table<M: Model + Sync>(
    run: &Run<'_, M>,
    method: &str,
    lead: &[&str],
    lead_values: &dyn Fn(usize) -> Vec<f64>,
    first_rows: Vec<Vec<f64>>,
    draws: &stan_services::ApproximateDraws,
    extra_config: Vec<(String, String)>,
    trailer: Vec<String>,
) -> Result<(), Failure> {
    let mut columns: Vec<String> = lead.iter().map(|s| s.to_string()).collect();
    columns.extend(draws.column_names.iter().cloned());
    let k = draws.column_names.len();
    let mut rows = first_rows;
    for i in 0..draws.len() {
        let mut row = lead_values(i);
        row.extend_from_slice(&draws.values[i * k..(i + 1) * k]);
        rows.push(row);
    }
    let mut config = run.config_lines(method);
    config.extend(extra_config);
    output::write_table(
        &run.output_file,
        run.overwrite(),
        &Table {
            config: &config,
            columns: &columns,
            rows: &rows,
            trailer: &trailer,
            sig_figs: run.sig_figs,
        },
    )
}

const APPROXIMATE_NOTE: &str = "these are draws from an approximation, not MCMC draws: R-hat \
                                and ESS do not apply to them";

// ------------------------------------------------------------------------------------------
// variational

fn variational<M: Model + Sync>(run: &Run<'_, M>) -> Result<ExitCode, Failure> {
    output::preflight(&run.output_file, run.overwrite())?;
    let p = run.parsed;
    let v = "method.variational";
    let d = AdviConfig::default();
    let advi = AdviConfig {
        family: match p.get(&format!("{v}.algorithm")).unwrap_or("meanfield") {
            "fullrank" => Family::FullRank,
            _ => Family::MeanField,
        },
        max_iterations: p.usize(&format!("{v}.iter")).unwrap_or(d.max_iterations),
        grad_samples: p
            .usize(&format!("{v}.grad_samples"))
            .unwrap_or(d.grad_samples),
        elbo_samples: p
            .usize(&format!("{v}.elbo_samples"))
            .unwrap_or(d.elbo_samples),
        eta: p.real(&format!("{v}.eta")).unwrap_or(d.eta),
        adapt_engaged: p
            .flag(&format!("{v}.adapt.engaged"))
            .unwrap_or(d.adapt_engaged),
        adapt_iterations: p
            .usize(&format!("{v}.adapt.iter"))
            .unwrap_or(d.adapt_iterations),
        tol_rel_obj: p.real(&format!("{v}.tol_rel_obj")).unwrap_or(d.tol_rel_obj),
        eval_elbo: p.usize(&format!("{v}.eval_elbo")).unwrap_or(d.eval_elbo),
        output_draws: p
            .usize(&format!("{v}.output_samples"))
            .unwrap_or(d.output_draws),
        ..d
    };
    let config = AdviServiceConfig {
        advi,
        init: run.init.clone(),
        max_init_attempts: MAX_INIT_ATTEMPTS,
        seed: run.seed,
        chain_id: run.id,
        blocks: blocks(),
    };
    let r = stan_services::run_advi(run.model, &config)?;
    let mut warnings = r.warnings.clone();
    warnings.push(APPROXIMATE_NOTE.into());
    // CmdStan's layout: the first row is the mean of the approximation (lp__, log_p__, and
    // log_g__ are 0 there), then one row per draw with lp__ = 0.
    let mut mean_row = vec![0.0, 0.0, 0.0];
    mean_row.extend(&r.mean_values);
    let draws = &r.draws;
    // CmdStan's log_g__ is the standard normal log density of the draw before it is
    // transformed, without its constant: -|eta|^2 / 2 = log q + log|det L| + (d/2) log(2 pi),
    // with log|det L| = log det(covariance) / 2.
    let d = r.mean_unconstrained.len();
    let shift = log_det_spd(&r.covariance, d)
        .map(|ld| 0.5 * ld + 0.5 * d as f64 * (2.0 * std::f64::consts::PI).ln())
        .unwrap_or(f64::NAN);
    approximate_table(
        run,
        "variational",
        &["lp__", "log_p__", "log_g__"],
        &|i| vec![0.0, draws.log_p[i], draws.log_q[i] + shift],
        vec![mean_row],
        draws,
        vec![
            ("family".into(), format!("{:?}", r.family)),
            ("eta".into(), r.eta.to_string()),
        ],
        vec![
            format!("converged = {}", r.converged),
            format!("iterations = {}", r.iterations),
            format!("pareto_k = {}", r.draws.pareto_k),
            format!("output_failures = {}", r.draws.output_failures),
            format!("Elapsed Time: {:.6} seconds (Total)", r.seconds),
        ],
    )?;
    let details = json!({
        "family": format!("{:?}", r.family),
        "converged": r.converged,
        "iterations": r.iterations,
        "eta": num(r.eta),
        "elbo_trace": r.elbo_trace.iter().map(|(i, e)| json!([i, num(*e)])).collect::<Vec<_>>(),
        "pareto_k": num(r.draws.pareto_k),
        "draws": r.draws.len(),
        "output_failures": r.draws.output_failures,
        "grad_evals": r.grad_evals,
        "seconds": num(r.seconds),
    });
    if !run.ext.quiet {
        for (i, e) in &r.elbo_trace {
            eprintln!("iteration {i:>6}  ELBO {e:.4}");
        }
        eprintln!(
            "ADVI ({:?}) {} after {} iterations; {} draws; Pareto k {:.2}",
            r.family,
            if r.converged {
                "converged"
            } else {
                "did not converge"
            },
            r.iterations,
            r.draws.len(),
            r.draws.pareto_k
        );
        eprintln!("wrote {}", run.output_file.display());
    }
    run.report_warnings(&warnings);
    run.write_manifest(
        "variational",
        ExitCode::Success,
        std::slice::from_ref(&run.output_file),
        details,
        &warnings,
    )?;
    Ok(ExitCode::Success)
}

// ------------------------------------------------------------------------------------------
// pathfinder

fn pathfinder<M: Model + Sync>(run: &Run<'_, M>) -> Result<ExitCode, Failure> {
    output::preflight(&run.output_file, run.overwrite())?;
    let p = run.parsed;
    let f = "method.pathfinder";
    let d = PathfinderConfig::default();
    let r = |k: &str| p.real(&format!("{f}.{k}"));
    let u = |k: &str| p.usize(&format!("{f}.{k}"));
    let pf = PathfinderConfig {
        num_paths: u("num_paths").unwrap_or(d.num_paths),
        history: u("history_size").unwrap_or(d.history),
        max_iterations: u("max_lbfgs_iters").unwrap_or(d.max_iterations),
        init_alpha: r("init_alpha").unwrap_or(d.init_alpha),
        tol_obj: r("tol_obj").unwrap_or(d.tol_obj),
        tol_rel_obj: r("tol_rel_obj").unwrap_or(d.tol_rel_obj),
        tol_grad: r("tol_grad").unwrap_or(d.tol_grad),
        tol_rel_grad: r("tol_rel_grad").unwrap_or(d.tol_rel_grad),
        tol_param: r("tol_param").unwrap_or(d.tol_param),
        num_elbo_draws: u("num_elbo_draws").unwrap_or(d.num_elbo_draws),
        num_draws: u("num_draws").unwrap_or(d.num_draws),
        num_psis_draws: u("num_psis_draws").unwrap_or(d.num_psis_draws),
        psis_resample: p
            .flag(&format!("{f}.psis_resample"))
            .unwrap_or(d.psis_resample),
        ..d
    };
    // CmdStan runs single-path Pathfinder for num_paths=1: the path's num_draws draws, without
    // importance resampling.
    let pf = PathfinderConfig {
        psis_resample: pf.psis_resample && pf.num_paths > 1,
        ..pf
    };
    let config = PathfinderServiceConfig {
        pathfinder: pf,
        init: run.init.clone(),
        max_init_attempts: MAX_INIT_ATTEMPTS,
        seed: run.seed,
        chain_id: run.id,
        blocks: blocks(),
    };
    let res = stan_services::run_pathfinder(run.model, &config)?;
    let mut warnings = res.warnings.clone();
    warnings.push(APPROXIMATE_NOTE.into());
    let draws = &res.draws;
    // path__ identifies the path that produced the draw; as in CmdStan, paths are numbered
    // from the run's id.
    let first = f64::from(run.id);
    let path_of = |i: usize| -> f64 {
        res.draw_path
            .get(i)
            .and_then(|&k| res.paths.get(k))
            .map_or(f64::NAN, |s| first + s.id as f64)
    };
    approximate_table(
        run,
        "pathfinder",
        &["lp_approx__", "lp__", "path__"],
        &|i| vec![draws.log_q[i], draws.log_p[i], path_of(i)],
        Vec::new(),
        draws,
        Vec::new(),
        vec![
            format!(
                "paths = {} of {} succeeded",
                res.paths.len(),
                res.num_paths_requested
            ),
            format!("pareto_k = {}", draws.pareto_k),
            format!("output_failures = {}", draws.output_failures),
            format!("Elapsed Time: {:.6} seconds (Total)", res.seconds),
        ],
    )?;
    let details = json!({
        "paths": res.paths.iter().map(|s| json!({
            "path": u64::from(run.id) + s.id,
            "elbo": num(s.elbo),
            "best_iteration": s.best_iteration,
            "iterations": s.iterations,
            "termination": format!("{:?}", s.termination),
            "grad_evals": s.grad_evals,
            "lp_evals": s.lp_evals,
        })).collect::<Vec<_>>(),
        "num_paths_requested": res.num_paths_requested,
        "pareto_k": num(draws.pareto_k),
        "draws": draws.len(),
        "output_failures": draws.output_failures,
        "seconds": num(res.seconds),
    });
    if !run.ext.quiet {
        eprintln!(
            "Pathfinder: {} of {} paths succeeded; {} draws; Pareto k {:.2}",
            res.paths.len(),
            res.num_paths_requested,
            draws.len(),
            draws.pareto_k
        );
        eprintln!("wrote {}", run.output_file.display());
    }
    run.report_warnings(&warnings);
    run.write_manifest(
        "pathfinder",
        ExitCode::Success,
        std::slice::from_ref(&run.output_file),
        details,
        &warnings,
    )?;
    Ok(ExitCode::Success)
}

// ------------------------------------------------------------------------------------------
// laplace

/// The mode on the unconstrained scale, from an optimize CSV file or a JSON file of
/// constrained values.
fn read_mode<M: Model + Sync>(
    run: &Run<'_, M>,
    path: &Path,
    jacobian: bool,
) -> Result<Vec<f64>, Failure> {
    if !path.exists() {
        return Err(Failure::new(
            ExitCode::NotFound,
            format!("mode file {} does not exist", path.display()),
        ));
    }
    let is_json = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("json"));
    if is_json {
        let values = Data::from_json_file(path)
            .map_err(|e| Failure::usage(format!("mode file {}: {e}", path.display())))?;
        return run
            .model
            .param_unconstrain_json(&values)
            .map_err(|e| Failure::usage(format!("mode file {}: {e}", path.display())));
    }
    reject_flat_complex(run, "a Stan CSV mode file")?;
    let csv = read_stan_csv(path).map_err(|e| Failure::input("mode file", path, &e))?;
    match csv.config_value("method") {
        Some("optimize") => {}
        Some(m) => {
            return Err(Failure::usage(format!(
                "mode file {} comes from method {m}, not optimize",
                path.display()
            )));
        }
        None => {
            return Err(Failure::usage(format!(
                "mode file {} does not say which method wrote it; give the CSV file of an \
                 optimize run, or a JSON file of constrained values",
                path.display()
            )));
        }
    }
    // An unfinished optimization (or one that failed) is not a mode.
    if let Some(c) = csv
        .column_index("converged__")
        .and_then(|j| csv.rows.last().map(|r| r[j]))
        && c < 0.0
    {
        return Err(Failure::new(
            ExitCode::Inference,
            format!(
                "mode file {}: the optimization failed (converged__ = {c}), so its last point \
                 is not a mode",
                path.display()
            ),
        ));
    }
    // CmdStan's optimize CSV records jacobian as `jacobian = 1` or `true`.
    if let Some(raw) = csv.config_value("jacobian") {
        let j = match raw.trim() {
            "1" | "true" => true,
            "0" | "false" => false,
            other => {
                return Err(Failure::usage(format!(
                    "mode file {}: jacobian = {other:?} is neither 0 nor 1",
                    path.display()
                )));
            }
        };
        if j != jacobian {
            return Err(Failure::usage(format!(
                "mode file {} was computed with jacobian={}, but laplace uses jacobian={}; the \
                 mode would not be the mode of the approximated density",
                path.display(),
                u8::from(j),
                u8::from(jacobian)
            )));
        }
    }
    let row = csv
        .rows
        .last()
        .ok_or_else(|| Failure::usage(format!("mode file {} has no rows", path.display())))?;
    let names = run.model.param_names(Outputs::PARAMETERS);
    let mut constrained = Vec::with_capacity(names.len());
    for n in &names {
        let j = csv.column_index(n).ok_or_else(|| {
            Failure::usage(format!(
                "mode file {} has no column {n}; it comes from another model",
                path.display()
            ))
        })?;
        constrained.push(row[j]);
    }
    run.model
        .param_unconstrain(&constrained)
        .map_err(|e| Failure::usage(format!("mode file {}: {e}", path.display())))
}

fn laplace<M: Model + Sync>(run: &Run<'_, M>) -> Result<ExitCode, Failure> {
    output::preflight(&run.output_file, run.overwrite())?;
    let p = run.parsed;
    let jacobian = p.flag("method.laplace.jacobian").unwrap_or(true);
    let mode = match p.file("method.laplace.mode") {
        Some(f) => {
            run.record_input("mode", &f);
            if f.extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("json"))
            {
                eprintln!(
                    "note: a JSON mode file does not record whether it was computed with the \
                     Jacobian; it is used as the mode of the jacobian={} density",
                    u8::from(jacobian)
                );
            }
            LaplaceMode::Unconstrained(read_mode(run, &f, jacobian)?)
        }
        None => LaplaceMode::Optimize(optimization_config(run, jacobian)),
    };
    let config = LaplaceServiceConfig {
        mode,
        jacobian,
        draws: p.usize("method.laplace.draws").unwrap_or(1000),
        seed: run.seed,
        chain_id: run.id,
        blocks: blocks(),
    };
    let r = stan_services::run_laplace(run.model, &config)?;
    if let Some(o) = &r.optimization
        && o.termination == Termination::LineSearchFailed
    {
        return Err(Failure::new(
            ExitCode::Inference,
            "the optimization that finds the mode failed (line search failure); no Laplace \
             approximation was computed. Pass a mode with mode=, or try other inits",
        ));
    }
    let mut warnings = r.warnings.clone();
    warnings.push(
        "the Hessian is a finite-difference Hessian of the gradient: stanc-rs models provide no \
         analytic Hessian"
            .into(),
    );
    warnings.push(APPROXIMATE_NOTE.into());
    let draws = &r.draws;
    approximate_table(
        run,
        "laplace",
        &["log_p__", "log_q__"],
        &|i| vec![draws.log_p[i], draws.log_q[i]],
        Vec::new(),
        draws,
        vec![("jacobian".into(), u8::from(jacobian).to_string())],
        vec![
            format!("log_density_mode = {}", r.log_density_mode),
            format!("grad_norm_at_mode = {}", r.grad_norm),
            format!("output_failures = {}", draws.output_failures),
            format!("Elapsed Time: {:.6} seconds (Total)", r.seconds),
        ],
    )?;
    let details = json!({
        "jacobian": jacobian,
        "coordinates": "Gaussian on the unconstrained scale, mapped through the constraining transforms",
        "mode_unconstrained": r.mode.iter().map(|&x| num(x)).collect::<Vec<_>>(),
        "log_density_mode": num(r.log_density_mode),
        "grad_norm": num(r.grad_norm),
        "optimization": r.optimization.as_ref().map(|o| json!({
            "termination": format!("{:?}", o.termination),
            "converged": o.converged,
            "iterations": o.iterations,
        })),
        "draws": draws.len(),
        "output_failures": draws.output_failures,
        "seconds": num(r.seconds),
    });
    if !run.ext.quiet {
        eprintln!(
            "Laplace approximation: {} draws; gradient norm at the mode {:.3e}",
            draws.len(),
            r.grad_norm
        );
        eprintln!("wrote {}", run.output_file.display());
    }
    run.report_warnings(&warnings);
    run.write_manifest(
        "laplace",
        ExitCode::Success,
        std::slice::from_ref(&run.output_file),
        details,
        &warnings,
    )?;
    Ok(ExitCode::Success)
}

// ------------------------------------------------------------------------------------------
// generate_quantities

fn generate_quantities<M: Model + Sync>(run: &Run<'_, M>) -> Result<ExitCode, Failure> {
    let raw = run
        .parsed
        .get("method.generate_quantities.fitted_params")
        .filter(|s| !s.is_empty())
        .ok_or_else(|| Failure::usage("generate_quantities needs fitted_params=<csv file>"))?;
    reject_flat_complex(run, "Stan CSV draws")?;
    // Only the generated quantities block is written (as CmdStan): without one there is
    // nothing to generate.
    let all = run.model.param_names(Outputs::ALL);
    let skip = run
        .model
        .param_names(Outputs {
            transformed_parameters: true,
            generated_quantities: false,
        })
        .len();
    if all.len() == skip {
        return Err(Failure::usage(
            "the model has no generated quantities; generate_quantities has nothing to compute",
        ));
    }
    let paths: Vec<PathBuf> = raw.split(',').map(|s| PathBuf::from(s.trim())).collect();
    let mut files = Vec::with_capacity(paths.len());
    for path in &paths {
        run.record_input("fitted_params", path);
        files.push(read_stan_csv(path).map_err(|e| Failure::input("fitted_params", path, &e))?);
    }
    // Chain identifiers select the random streams. Use the files' ids when they are distinct
    // and no id was given; otherwise (CmdStan writes the first id into every file of a
    // multi-chain run) number the chains from `id`.
    let mut file_ids: Vec<Option<&str>> = files.iter().map(|f| f.config_value("id")).collect();
    let n_files = file_ids.len();
    file_ids.sort_unstable();
    file_ids.dedup();
    let distinct = file_ids.len() == n_files && file_ids.iter().all(Option::is_some);
    let ids: Option<Vec<u32>> = if distinct && run.parsed.get("id").is_none() {
        None
    } else {
        Some(
            (0..n_files as u32)
                .map(|k| run.id.checked_add(k))
                .collect::<Option<_>>()
                .ok_or_else(|| Failure::usage("chain identifiers overflow"))?,
        )
    };
    let input = GqInput::from_stan_csv(&files, ids.as_deref())?;
    let targets: Vec<(u32, PathBuf)> = input
        .chains
        .iter()
        .map(|c| {
            (
                c.chain_id,
                sibling(&run.output_file, &format!("_{}", c.chain_id), "csv"),
            )
        })
        .collect();
    for (_, t) in &targets {
        output::preflight(t, run.overwrite())?;
        // Never replace an input with an output (even with --overwrite).
        let same = |a: &Path, b: &Path| match (a.canonicalize(), b.canonicalize()) {
            (Ok(x), Ok(y)) => x == y,
            _ => a == b,
        };
        if paths.iter().any(|p| same(p, t)) {
            return Err(Failure::usage(format!(
                "the output file {} is also an input; choose another output file",
                t.display()
            )));
        }
    }
    let start = Instant::now();
    let r = stan_services::generate_quantities(run.model, &input, run.seed)?;
    let k = r.columns.len();
    let gq_columns = &r.columns[skip.min(k)..];
    let mut written = Vec::new();
    let mut warnings = Vec::new();
    for (pos, (chain, (id, path))) in r.chains.iter().zip(&targets).enumerate() {
        debug_assert_eq!(chain.chain_id, *id);
        let rows: Vec<Vec<f64>> = chain
            .values
            .chunks(k.max(1))
            .map(|r| r[skip.min(r.len())..].to_vec())
            .collect();
        let mut config = run.config_lines("generate_quantities");
        config.push(("chain_id".into(), id.to_string()));
        // The draw file this output belongs to: join outputs and draws by it, not by position.
        config.push(("fitted_params".into(), paths[pos].display().to_string()));
        output::write_table(
            path,
            run.overwrite(),
            &Table {
                config: &config,
                columns: gq_columns,
                rows: &rows,
                // Timing, as the other methods' outputs (stan-dev/stan#3376).
                trailer: &[
                    format!("output_failures = {}", chain.failures),
                    format!(
                        "Elapsed Time: {:.6} seconds (Total)",
                        start.elapsed().as_secs_f64()
                    ),
                ],
                sig_figs: run.sig_figs,
            },
        )?;
        if chain.failures > 0 {
            warnings.push(format!(
                "chain {id}: {} draws failed and were written as NaN",
                chain.failures
            ));
        }
        written.push(path.clone());
    }
    let details = json!({
        "inputs": paths.iter().map(|p| json!({
            "path": p.display().to_string(),
            "sha256": sha256_file(p).ok(),
        })).collect::<Vec<_>>(),
        "chains": r.chains.iter().map(|c| json!({
            "chain_id": c.chain_id,
            "draws": c.values.len() / k.max(1),
            "failures": c.failures,
        })).collect::<Vec<_>>(),
        "seconds": num(start.elapsed().as_secs_f64()),
    });
    if !run.ext.quiet {
        for w in &written {
            eprintln!("wrote {}", w.display());
        }
    }
    run.report_warnings(&warnings);
    // Every draw failing is a failed run, not output full of NaN with exit 0.
    let total: usize = r.chains.iter().map(|c| c.values.len() / k.max(1)).sum();
    let failed: usize = r.chains.iter().map(|c| c.failures).sum();
    let exit = if total > 0 && failed == total {
        eprintln!("error: the generated quantities failed for every draw");
        ExitCode::Inference
    } else {
        ExitCode::Success
    };
    run.write_manifest("generate_quantities", exit, &written, details, &warnings)?;
    Ok(exit)
}

// ------------------------------------------------------------------------------------------
// log_prob and diagnose

/// Unconstrained vectors from CmdStan's JSON (`{"params_r": [...]}` with one vector or a list of
/// vectors).
fn read_unconstrained(path: &Path) -> Result<Vec<Vec<f64>>, Failure> {
    let text =
        std::fs::read_to_string(path).map_err(|e| Failure::input("parameter file", path, &e))?;
    let v: Value = serde_json::from_str(&text)
        .map_err(|e| Failure::usage(format!("{}: {e}", path.display())))?;
    let bad = || {
        Failure::usage(format!(
            "{}: expected {{\"params_r\": [numbers]}} or {{\"params_r\": [[numbers], ...]}}",
            path.display()
        ))
    };
    let pr = v
        .get("params_r")
        .and_then(Value::as_array)
        .ok_or_else(bad)?;
    let vector = |x: &Value| -> Result<Vec<f64>, Failure> {
        x.as_array()
            .ok_or_else(bad)?
            .iter()
            .map(|e| e.as_f64().ok_or_else(bad))
            .collect()
    };
    if pr.iter().all(Value::is_array) && !pr.is_empty() {
        pr.iter().map(vector).collect()
    } else {
        Ok(vec![vector(&Value::Array(pr.clone()))?])
    }
}

fn log_prob<M: Model + Sync>(run: &Run<'_, M>) -> Result<ExitCode, Failure> {
    output::preflight(&run.output_file, run.overwrite())?;
    let p = run.parsed;
    let jacobian = p.flag("method.log_prob.jacobian").unwrap_or(true);
    let unc = p.file("method.log_prob.unconstrained_params");
    let con = p.file("method.log_prob.constrained_params");
    // As CmdStan: one input, never both concatenated.
    if unc.is_some() && con.is_some() {
        return Err(Failure::usage(
            "give either unconstrained_params or constrained_params, not both",
        ));
    }
    let mut points = Vec::new();
    if let Some(f) = unc {
        run.record_input("unconstrained_params", &f);
        points.extend(
            read_unconstrained(&f)?
                .into_iter()
                .map(Point::Unconstrained),
        );
    }
    if let Some(f) = con {
        run.record_input("constrained_params", &f);
        if !f.exists() {
            return Err(Failure::new(
                ExitCode::NotFound,
                format!("constrained_params file {} does not exist", f.display()),
            ));
        }
        let is_csv = f.extension().is_some_and(|e| e.eq_ignore_ascii_case("csv"));
        if is_csv {
            // Every post-warmup draw of a complete Stan CSV file.
            reject_flat_complex(run, "a Stan CSV file")?;
            let csv =
                read_stan_csv(&f).map_err(|e| Failure::input("constrained_params", &f, &e))?;
            if !csv.is_complete() {
                return Err(Failure::usage(format!(
                    "constrained_params {} is incomplete (a failed or interrupted run)",
                    f.display()
                )));
            }
            let names = run.model.param_names(Outputs::PARAMETERS);
            let idx: Vec<usize> = names
                .iter()
                .map(|n| {
                    csv.column_index(n)
                        .ok_or_else(|| Failure::usage(format!("{} has no column {n}", f.display())))
                })
                .collect::<Result<_, _>>()?;
            for row in csv.draws() {
                let c: Vec<f64> = idx.iter().map(|&j| row[j]).collect();
                let theta = run.model.param_unconstrain(&c).map_err(|e| {
                    Failure::usage(format!("constrained_params {}: {e}", f.display()))
                })?;
                points.push(Point::Unconstrained(theta));
            }
        } else {
            let d = Data::from_json_file(&f)
                .map_err(|e| Failure::usage(format!("constrained_params {}: {e}", f.display())))?;
            points.push(Point::Constrained(d));
        }
    }
    if points.is_empty() {
        return Err(Failure::usage(
            "log_prob needs unconstrained_params=<json> or constrained_params=<json|csv>",
        ));
    }
    // CmdStan's log_prob drops constant terms.
    let options = LogDensityOptions {
        jacobian,
        propto: true,
    };
    let values = stan_services::log_density(run.model, &points, options)?;
    let names = run.model.unconstrained_param_names();
    let mut columns = vec!["lp__".to_string()];
    // CmdStan's header: lp__, then g_<unconstrained parameter name>.
    columns.extend(names.iter().map(|n| format!("g_{n}")));
    let rows: Vec<Vec<f64>> = values
        .iter()
        .map(|v| {
            let mut r = vec![v.log_density];
            r.extend(&v.gradient);
            r
        })
        .collect();
    let mut config = run.config_lines("log_prob");
    config.push(("jacobian".into(), u8::from(jacobian).to_string()));
    config.push(("propto".into(), "1".into()));
    output::write_table(
        &run.output_file,
        run.overwrite(),
        &Table {
            config: &config,
            columns: &columns,
            rows: &rows,
            trailer: &[],
            sig_figs: run.sig_figs,
        },
    )?;
    let result = json!({
        "jacobian": jacobian,
        "propto": true,
        "coordinates": "gradients are with respect to the unconstrained parameters",
        "unconstrained_names": names,
        "points": values.iter().map(|v| json!({
            "unconstrained": v.theta.iter().map(|&x| num(x)).collect::<Vec<_>>(),
            "log_density": num(v.log_density),
            "gradient": v.gradient.iter().map(|&x| num(x)).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    });
    if run.ext.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&result).expect("serializes")
        );
    }
    if !run.ext.quiet {
        eprintln!("wrote {}", run.output_file.display());
    }
    run.write_manifest(
        "log_prob",
        ExitCode::Success,
        std::slice::from_ref(&run.output_file),
        result,
        &[],
    )?;
    Ok(ExitCode::Success)
}

fn diagnose<M: Model + Sync>(run: &Run<'_, M>) -> Result<ExitCode, Failure> {
    output::preflight(&run.output_file, run.overwrite())?;
    let p = run.parsed;
    let g = "method.diagnose.test.gradient";
    let points = if let Some(f) = &run.ext.unconstrained {
        GradientPoints::Given(
            read_unconstrained(f)?
                .into_iter()
                .map(Point::Unconstrained)
                .collect(),
        )
    } else {
        // CmdStan checks the gradient at the initial point.
        match &run.init {
            InitStrategy::Constrained { values, .. } => {
                GradientPoints::Given(vec![Point::Constrained(values.clone())])
            }
            InitStrategy::Zero => GradientPoints::Given(vec![Point::Unconstrained(vec![
                0.0;
                run.model
                    .num_unconstrained(
                    )
            ])]),
            InitStrategy::Random { radius } => GradientPoints::Random {
                count: 1,
                radius: *radius,
            },
            InitStrategy::Unconstrained(v) => {
                GradientPoints::Given(vec![Point::Unconstrained(v.clone())])
            }
        }
    };
    let config = GradientCheckConfig {
        epsilon: p.real(&format!("{g}.epsilon")).unwrap_or(1e-6),
        error: p.real(&format!("{g}.error")).unwrap_or(1e-6),
        points,
        // The full density (with constants): its gradient equals the gradient of the density
        // with constants dropped, and the full density is computed without automatic
        // differentiation, so a term dropped by mistake from the AD path shows up here.
        options: LogDensityOptions {
            jacobian: true,
            propto: false,
        },
        seed: run.seed,
    };
    if let Some(f) = &run.ext.unconstrained {
        run.record_input("unconstrained_params", f);
    }
    let report = stan_services::check_gradients(run.model, &config)?;
    let columns: Vec<String> = [
        "point",
        "param_idx",
        "value",
        "model",
        "finite_diff",
        "error",
        "abs_error",
        "rel_error",
        "flagged",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    let mut rows = Vec::new();
    let mut warnings = Vec::new();
    let mut points_json = Vec::new();
    for (k, pc) in report.points.iter().enumerate() {
        match pc {
            PointCheck::Checked {
                log_density,
                components,
            } => {
                let mut comps = Vec::new();
                for (i, c) in components.iter().enumerate() {
                    let rel = c.error.abs() / c.finite_diff.abs().max(1e-300);
                    rows.push(vec![
                        k as f64,
                        i as f64,
                        c.value,
                        c.model,
                        c.finite_diff,
                        c.error,
                        c.error.abs(),
                        rel,
                        f64::from(u8::from(c.flagged)),
                    ]);
                    comps.push(json!({
                        "index": i,
                        "name": c.name,
                        "value": num(c.value),
                        "model": num(c.model),
                        "finite_diff": num(c.finite_diff),
                        "abs_error": num(c.error.abs()),
                        "rel_error": num(rel),
                        "flagged": c.flagged,
                    }));
                }
                points_json.push(json!({"log_density": num(*log_density), "components": comps}));
            }
            PointCheck::Failed(msg) => {
                warnings.push(format!("point {k}: the check could not be computed: {msg}"));
                points_json.push(json!({"failed": msg}));
            }
        }
    }
    let flagged = report.num_flagged();
    let uncomputed = report
        .points
        .iter()
        .filter(|p| matches!(p, PointCheck::Failed(_)))
        .count();
    // A flagged component, or a point where the check could not be computed, is a failed
    // check: exit nonzero, never a pass without checking (stan-dev/cmdstan#1278 asks the same
    // of CmdStan's separate diagnose utility, which always exits 0).
    let exit = if flagged > 0 || uncomputed > 0 {
        ExitCode::Diagnostics
    } else {
        ExitCode::Success
    };
    if flagged > 0 {
        warnings.push(format!(
            "{flagged} gradient components differ from finite differences by more than {}",
            config.error
        ));
    }
    warnings.push(
        "agreement with finite differences at a few points is evidence, not proof: it can fail \
         near boundaries and at poorly conditioned points, and can hide errors elsewhere"
            .into(),
    );
    let mut cfg_lines = run.config_lines("diagnose");
    cfg_lines.push(("epsilon".into(), config.epsilon.to_string()));
    cfg_lines.push(("error".into(), config.error.to_string()));
    output::write_table(
        &run.output_file,
        run.overwrite(),
        &Table {
            config: &cfg_lines,
            columns: &columns,
            rows: &rows,
            trailer: &[],
            sig_figs: run.sig_figs,
        },
    )?;
    let result = json!({
        "epsilon": num(config.epsilon),
        "error_threshold": num(config.error),
        "step": "central differences of the log density (Jacobian included, constants included)",
        "flagged": flagged,
        "points": points_json,
    });
    if run.ext.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&result).expect("serializes")
        );
    } else if !run.ext.quiet {
        eprintln!(
            "{:>5} {:>9} {:>14} {:>14} {:>14} {:>11}",
            "point", "param_idx", "value", "model", "finite_diff", "error"
        );
        for r in &rows {
            eprintln!(
                "{:>5} {:>9} {:>14.6e} {:>14.6e} {:>14.6e} {:>11.3e}{}",
                r[0],
                r[1],
                r[2],
                r[3],
                r[4],
                r[5],
                if r[8] > 0.0 { "  *" } else { "" }
            );
        }
    }
    run.report_warnings(&warnings);
    run.write_manifest(
        "diagnose",
        exit,
        std::slice::from_ref(&run.output_file),
        result,
        &warnings,
    )?;
    Ok(exit)
}
