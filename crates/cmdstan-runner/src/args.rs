//! CmdStan's command line grammar, parsed into typed settings.
//!
//! CmdStan arguments form a tree. A *group* (`data`, `output`, `adapt`) is a bare word that
//! opens its sub-arguments; a *choice* (`method=sample`, `algorithm=hmc`, `engine=nuts`) selects
//! one option, which opens that option's sub-arguments; a *value* (`num_samples=1000`) sets a
//! number, flag, or file name. A word is looked up in the innermost open context first, then in
//! the enclosing ones, as CmdStan does; finding it in an outer context closes the inner ones.
//! Every method name may also be given bare (`sample` for `method=sample`).
//!
//! The grammar here is CmdStan 2.40's. Unknown words, values of the wrong type, a value given
//! twice, and a word in a context where it does not belong are errors; nothing is ignored.
//! Options that CmdStan has but cmdstan-rs cannot honor yet are recognized and rejected with an
//! explanation (see [`UNSUPPORTED`]).
//!
//! Arguments that start with `--` are cmdstan-rs extensions, not CmdStan syntax (see
//! [`Extensions`]).

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::exit::Failure;

/// The type of a value argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ty {
    /// Nonnegative integer.
    Uint,
    /// Positive integer.
    PosInt,
    /// Integer (may be negative).
    Int,
    /// Finite real.
    Real,
    /// `0`/`1` (also `false`/`true`).
    Bool,
    /// A file name or other text.
    Text,
}

#[derive(Debug)]
enum Kind {
    Value(Ty),
    Group(&'static [Node]),
    Choice(&'static [Node]),
}

#[derive(Debug)]
struct Node {
    name: &'static str,
    kind: Kind,
    help: &'static str,
}

const fn v(name: &'static str, ty: Ty, help: &'static str) -> Node {
    Node {
        name,
        kind: Kind::Value(ty),
        help,
    }
}

const fn g(name: &'static str, children: &'static [Node], help: &'static str) -> Node {
    Node {
        name,
        kind: Kind::Group(children),
        help,
    }
}

const fn c(name: &'static str, options: &'static [Node], help: &'static str) -> Node {
    Node {
        name,
        kind: Kind::Choice(options),
        help,
    }
}

/// An option of a choice without sub-arguments.
const fn o(name: &'static str, help: &'static str) -> Node {
    g(name, &[], help)
}

const BFGS_ARGS: &[Node] = &[
    v(
        "init_alpha",
        Ty::Real,
        "first line search step (default 0.001)",
    ),
    v(
        "tol_obj",
        Ty::Real,
        "absolute objective tolerance (default 1e-12)",
    ),
    v(
        "tol_rel_obj",
        Ty::Real,
        "relative objective tolerance (default 1e4)",
    ),
    v(
        "tol_grad",
        Ty::Real,
        "gradient norm tolerance (default 1e-8)",
    ),
    v(
        "tol_rel_grad",
        Ty::Real,
        "relative gradient tolerance (default 1e7)",
    ),
    v(
        "tol_param",
        Ty::Real,
        "parameter change tolerance (default 1e-8)",
    ),
];

const LBFGS_ARGS: &[Node] = &[
    v(
        "init_alpha",
        Ty::Real,
        "first line search step (default 0.001)",
    ),
    v(
        "tol_obj",
        Ty::Real,
        "absolute objective tolerance (default 1e-12)",
    ),
    v(
        "tol_rel_obj",
        Ty::Real,
        "relative objective tolerance (default 1e4)",
    ),
    v(
        "tol_grad",
        Ty::Real,
        "gradient norm tolerance (default 1e-8)",
    ),
    v(
        "tol_rel_grad",
        Ty::Real,
        "relative gradient tolerance (default 1e7)",
    ),
    v(
        "tol_param",
        Ty::Real,
        "parameter change tolerance (default 1e-8)",
    ),
    v(
        "history_size",
        Ty::PosInt,
        "number of update vectors kept (default 5)",
    ),
];

const SAMPLE: &[Node] = &[
    v(
        "num_samples",
        Ty::Uint,
        "post-warmup iterations (default 1000)",
    ),
    v("num_warmup", Ty::Uint, "warmup iterations (default 1000)"),
    v("save_warmup", Ty::Bool, "write warmup draws (default 0)"),
    v(
        "thin",
        Ty::PosInt,
        "keep every thin-th iteration (default 1)",
    ),
    g(
        "adapt",
        &[
            v("engaged", Ty::Bool, "adapt during warmup (default 1)"),
            v(
                "gamma",
                Ty::Real,
                "dual averaging regularization (default 0.05)",
            ),
            v(
                "delta",
                Ty::Real,
                "target acceptance statistic (default 0.8)",
            ),
            v(
                "kappa",
                Ty::Real,
                "dual averaging relaxation exponent (default 0.75)",
            ),
            v(
                "t0",
                Ty::Real,
                "dual averaging iteration offset (default 10)",
            ),
            v(
                "init_buffer",
                Ty::Uint,
                "first fast adaptation interval (default 75)",
            ),
            v(
                "term_buffer",
                Ty::Uint,
                "last fast adaptation interval (default 50)",
            ),
            v(
                "window",
                Ty::PosInt,
                "first slow adaptation window (default 25)",
            ),
            v(
                "save_metric",
                Ty::Bool,
                "write the adapted metric as JSON (default 0)",
            ),
        ],
        "warmup adaptation",
    ),
    c(
        "algorithm",
        &[
            g(
                "hmc",
                &[
                    c(
                        "engine",
                        &[
                            g(
                                "nuts",
                                &[v(
                                    "max_depth",
                                    Ty::PosInt,
                                    "maximum tree depth (default 10)",
                                )],
                                "the No-U-Turn sampler (default)",
                            ),
                            g(
                                "static",
                                &[v("int_time", Ty::Real, "integration time (default 2 pi)")],
                                "static HMC",
                            ),
                        ],
                        "HMC engine",
                    ),
                    c(
                        "metric",
                        &[
                            o("unit_e", "identity metric"),
                            o("diag_e", "diagonal metric (default)"),
                            o("dense_e", "dense metric"),
                        ],
                        "Euclidean metric",
                    ),
                    v(
                        "metric_file",
                        Ty::Text,
                        "initial inverse metric (JSON inv_metric)",
                    ),
                    v("stepsize", Ty::Real, "initial step size (default 1)"),
                    v(
                        "stepsize_jitter",
                        Ty::Real,
                        "step size jitter in [0, 1) (default 0)",
                    ),
                ],
                "Hamiltonian Monte Carlo (default)",
            ),
            o("fixed_param", "no parameter updates"),
        ],
        "sampling algorithm",
    ),
    v("num_chains", Ty::PosInt, "number of chains (default 1)"),
];

const OPTIMIZE: &[Node] = &[
    c(
        "algorithm",
        &[
            g("bfgs", BFGS_ARGS, "BFGS"),
            g("lbfgs", LBFGS_ARGS, "L-BFGS (default)"),
            o(
                "newton",
                "Newton's method (numerical Hessian for stanc-rs models)",
            ),
        ],
        "optimization algorithm",
    ),
    v(
        "jacobian",
        Ty::Bool,
        "include the Jacobian: mode of the unconstrained density (default 0)",
    ),
    v("iter", Ty::PosInt, "maximum iterations (default 2000)"),
    v(
        "save_iterations",
        Ty::Bool,
        "write every iterate (not supported)",
    ),
];

const VARIATIONAL: &[Node] = &[
    c(
        "algorithm",
        &[
            o("meanfield", "independent normals (default)"),
            o("fullrank", "multivariate normal"),
        ],
        "variational family",
    ),
    v("iter", Ty::PosInt, "maximum iterations (default 10000)"),
    v("grad_samples", Ty::PosInt, "draws per gradient (default 1)"),
    v(
        "elbo_samples",
        Ty::PosInt,
        "draws per ELBO estimate (default 100)",
    ),
    v("eta", Ty::Real, "step size scale (default 1)"),
    g(
        "adapt",
        &[
            v("engaged", Ty::Bool, "choose eta by grid search (default 1)"),
            v(
                "iter",
                Ty::PosInt,
                "iterations per eta candidate (default 50)",
            ),
        ],
        "step size adaptation",
    ),
    v(
        "tol_rel_obj",
        Ty::Real,
        "relative ELBO tolerance (default 0.01)",
    ),
    v(
        "eval_elbo",
        Ty::PosInt,
        "evaluate the ELBO every this many iterations (default 100)",
    ),
    v(
        "output_samples",
        Ty::PosInt,
        "approximate draws (default 1000)",
    ),
];

const DIAGNOSE: &[Node] = &[c(
    "test",
    &[g(
        "gradient",
        &[
            v("epsilon", Ty::Real, "finite difference step (default 1e-6)"),
            v("error", Ty::Real, "error threshold (default 1e-6)"),
        ],
        "compare gradients with finite differences (default)",
    )],
    "diagnostic test",
)];

const GENERATE_QUANTITIES: &[Node] = &[v(
    "fitted_params",
    Ty::Text,
    "Stan CSV file of draws (required); comma separated for several chains",
)];

const PATHFINDER: &[Node] = &[
    v(
        "init_alpha",
        Ty::Real,
        "first line search step (default 0.001)",
    ),
    v(
        "tol_obj",
        Ty::Real,
        "absolute objective tolerance (default 1e-12)",
    ),
    v(
        "tol_rel_obj",
        Ty::Real,
        "relative objective tolerance (default 1e4)",
    ),
    v(
        "tol_grad",
        Ty::Real,
        "gradient norm tolerance (default 1e-8)",
    ),
    v(
        "tol_rel_grad",
        Ty::Real,
        "relative gradient tolerance (default 1e7)",
    ),
    v(
        "tol_param",
        Ty::Real,
        "parameter change tolerance (default 1e-8)",
    ),
    v("history_size", Ty::PosInt, "L-BFGS history (default 5)"),
    v(
        "num_psis_draws",
        Ty::PosInt,
        "draws after importance resampling (default 1000)",
    ),
    v("num_paths", Ty::PosInt, "independent paths (default 4)"),
    v(
        "save_single_paths",
        Ty::Bool,
        "write each path (not supported)",
    ),
    v(
        "psis_resample",
        Ty::Bool,
        "importance resample the pooled draws (default 1)",
    ),
    v(
        "calculate_lp",
        Ty::Bool,
        "compute log densities of the draws (only 1 is supported)",
    ),
    v(
        "max_lbfgs_iters",
        Ty::PosInt,
        "L-BFGS iterations per path (default 1000)",
    ),
    v("num_draws", Ty::PosInt, "draws per path (default 1000)"),
    v(
        "num_elbo_draws",
        Ty::PosInt,
        "draws per ELBO estimate (default 25)",
    ),
];

const LAPLACE: &[Node] = &[
    v(
        "mode",
        Ty::Text,
        "mode as a Stan CSV (optimize output) or JSON file; omitted: optimize first",
    ),
    v("jacobian", Ty::Bool, "include the Jacobian (default 1)"),
    v("draws", Ty::PosInt, "number of draws (default 1000)"),
    v(
        "calculate_lp",
        Ty::Bool,
        "compute log densities of the draws (only 1 is supported)",
    ),
];

const LOG_PROB: &[Node] = &[
    v(
        "unconstrained_params",
        Ty::Text,
        "JSON file of unconstrained vectors",
    ),
    v(
        "constrained_params",
        Ty::Text,
        "JSON file of constrained values",
    ),
    v("jacobian", Ty::Bool, "include the Jacobian (default 1)"),
];

const METHODS: &[Node] = &[
    g("sample", SAMPLE, "MCMC"),
    g("optimize", OPTIMIZE, "posterior mode or maximum likelihood"),
    g("variational", VARIATIONAL, "ADVI"),
    g("diagnose", DIAGNOSE, "gradient check"),
    g(
        "generate_quantities",
        GENERATE_QUANTITIES,
        "generated quantities of existing draws",
    ),
    g("pathfinder", PATHFINDER, "Pathfinder"),
    g("laplace", LAPLACE, "Laplace approximation"),
    g("log_prob", LOG_PROB, "log density and gradient"),
];

const ROOT: &[Node] = &[
    c("method", METHODS, "inference method (required)"),
    v(
        "id",
        Ty::Uint,
        "chain identifier of the first chain (default 1)",
    ),
    g(
        "data",
        &[v("file", Ty::Text, "JSON data file")],
        "input data",
    ),
    v(
        "init",
        Ty::Text,
        "initialization radius, or a JSON file of initial values (default 2)",
    ),
    g(
        "random",
        &[v(
            "seed",
            Ty::Int,
            "random seed (default: from the clock, recorded in the output)",
        )],
        "random number generation",
    ),
    g(
        "output",
        &[
            v("file", Ty::Text, "output file (default output.csv)"),
            v(
                "diagnostic_file",
                Ty::Text,
                "unconstrained draws (not supported)",
            ),
            v(
                "refresh",
                Ty::Uint,
                "progress every this many iterations; 0 for none (default 100)",
            ),
            v(
                "sig_figs",
                Ty::Int,
                "significant digits of real values, 1 to 18 (default 8)",
            ),
            v("profile_file", Ty::Text, "profiling output (not supported)"),
            v(
                "save_cmdstan_config",
                Ty::Bool,
                "write the configuration as JSON (not supported)",
            ),
        ],
        "output files",
    ),
    v(
        "num_threads",
        Ty::Int,
        "chains run at once; -1 for the number of CPUs (default 1)",
    ),
];

/// Recognized CmdStan options that cmdstan-rs rejects, with the reason. Each one is an error
/// when set to anything other than its default, never a silent no-op.
pub const UNSUPPORTED: &[(&str, &str)] = &[
    (
        "output.diagnostic_file",
        "unconstrained draws are not written yet",
    ),
    (
        "output.profile_file",
        "stanc-rs models do not record profiles",
    ),
    (
        "output.save_cmdstan_config",
        "the configuration is recorded in the run manifest instead",
    ),
    (
        "method.optimize.save_iterations",
        "stan-services does not return the optimizer's iterates",
    ),
    (
        "method.pathfinder.save_single_paths",
        "stan-services returns only the pooled draws",
    ),
    (
        "method.pathfinder.calculate_lp",
        "log densities are always computed",
    ),
    (
        "method.laplace.calculate_lp",
        "log densities are always computed",
    ),
];

/// The arguments as given, by path (`method.sample.adapt.delta`), with the selected choice
/// option as the value of a choice path (`method` = `sample`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Parsed {
    values: BTreeMap<String, String>,
}

/// Parses CmdStan arguments (without the program name and without `--` extensions).
pub fn parse(args: &[String]) -> Result<Parsed, Failure> {
    let mut parsed = Parsed::default();
    // Open contexts: (path, children).
    let mut stack: Vec<(String, &'static [Node])> = vec![(String::new(), ROOT)];
    for arg in args {
        let (key, value) = match arg.split_once('=') {
            Some((k, v)) => (k, Some(v)),
            None => (arg.as_str(), None),
        };
        if key.is_empty() {
            return Err(Failure::usage(format!("malformed argument {arg:?}")));
        }
        // A bare method name selects the method.
        if value.is_none() && METHODS.iter().any(|m| m.name == key) {
            stack.truncate(1);
            parsed.set("method", key)?;
            let m = METHODS.iter().find(|m| m.name == key).expect("checked");
            stack.push((format!("method.{key}"), children(m)));
            continue;
        }
        let found = stack
            .iter()
            .enumerate()
            .rev()
            .find_map(|(level, (path, nodes))| {
                nodes
                    .iter()
                    .find(|n| n.name == key)
                    .map(|n| (level, join(path, key), n))
            });
        let Some((level, path, node)) = found else {
            let context = stack
                .last()
                .map(|(p, _)| p.as_str())
                .filter(|p| !p.is_empty())
                .unwrap_or("the top level");
            return Err(Failure::usage(format!(
                "unrecognized argument {key:?} (in {context}); run with `help-all` for the \
                 argument tree"
            )));
        };
        stack.truncate(level + 1);
        match &node.kind {
            Kind::Value(ty) => {
                let Some(v) = value else {
                    return Err(Failure::usage(format!(
                        "{path} needs a value: {key}=<value>"
                    )));
                };
                check_type(&path, v, *ty)?;
                parsed.set(&path, v)?;
            }
            Kind::Group(children) => {
                if value.is_some() {
                    return Err(Failure::usage(format!(
                        "{path} is a group of arguments and takes no value"
                    )));
                }
                stack.push((path, children));
            }
            Kind::Choice(options) => {
                let Some(v) = value else {
                    let names: Vec<&str> = options.iter().map(|o| o.name).collect();
                    return Err(Failure::usage(format!(
                        "{path} needs one of: {}",
                        names.join(", ")
                    )));
                };
                let Some(opt) = options.iter().find(|o| o.name == v) else {
                    let names: Vec<&str> = options.iter().map(|o| o.name).collect();
                    return Err(Failure::usage(format!(
                        "{path}={v} is not valid; expected one of: {}",
                        names.join(", ")
                    )));
                };
                parsed.set(&path, v)?;
                stack.push((format!("{path}.{v}"), children(opt)));
            }
        }
    }
    for (path, reason) in UNSUPPORTED {
        if let Some(v) = parsed.values.get(*path) {
            let default = matches!(
                (*path, v.as_str()),
                (
                    "method.pathfinder.calculate_lp" | "method.laplace.calculate_lp",
                    "1" | "true"
                ) | (
                    "output.save_cmdstan_config"
                        | "method.optimize.save_iterations"
                        | "method.pathfinder.save_single_paths",
                    "0" | "false"
                )
            ) || ((*path == "output.diagnostic_file"
                || *path == "output.profile_file")
                && v.is_empty());
            if !default {
                return Err(Failure::usage(format!(
                    "{path}={v} is not supported by cmdstan-rs: {reason}"
                )));
            }
        }
    }
    Ok(parsed)
}

fn children(n: &'static Node) -> &'static [Node] {
    match &n.kind {
        Kind::Group(c) => c,
        _ => &[],
    }
}

fn join(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.to_string()
    } else {
        format!("{path}.{key}")
    }
}

fn check_type(path: &str, v: &str, ty: Ty) -> Result<(), Failure> {
    let bad = |what: &str| Err(Failure::usage(format!("{path}={v}: expected {what}")));
    match ty {
        Ty::Uint => {
            if v.parse::<usize>().is_err() {
                return bad("a nonnegative integer");
            }
        }
        Ty::PosInt => {
            if !v.parse::<usize>().is_ok_and(|x| x > 0) {
                return bad("a positive integer");
            }
        }
        Ty::Int => {
            if v.parse::<i64>().is_err() {
                return bad("an integer");
            }
        }
        Ty::Real => {
            if !v.parse::<f64>().is_ok_and(f64::is_finite) {
                return bad("a finite real number");
            }
        }
        Ty::Bool => {
            if !matches!(v, "0" | "1" | "true" | "false") {
                return bad("0 or 1");
            }
        }
        Ty::Text => {}
    }
    Ok(())
}

impl Parsed {
    fn set(&mut self, path: &str, v: &str) -> Result<(), Failure> {
        if let Some(old) = self.values.insert(path.to_string(), v.to_string()) {
            return Err(Failure::usage(format!(
                "{path} is given twice ({old} and {v})"
            )));
        }
        Ok(())
    }

    /// The raw value of `path`, if given.
    pub fn get(&self, path: &str) -> Option<&str> {
        self.values.get(path).map(String::as_str)
    }

    /// Every given argument, by path.
    pub fn values(&self) -> &BTreeMap<String, String> {
        &self.values
    }

    /// The selected method.
    pub fn method(&self) -> Option<&str> {
        self.get("method")
    }

    /// An unsigned integer, if given (types were checked while parsing).
    pub fn usize(&self, path: &str) -> Option<usize> {
        self.get(path)
            .map(|v| v.parse().expect("checked while parsing"))
    }

    /// A real number, if given.
    pub fn real(&self, path: &str) -> Option<f64> {
        self.get(path)
            .map(|v| v.parse().expect("checked while parsing"))
    }

    /// A flag, if given.
    pub fn flag(&self, path: &str) -> Option<bool> {
        self.get(path).map(|v| v == "1" || v == "true")
    }

    /// A signed integer, if given.
    pub fn int(&self, path: &str) -> Option<i64> {
        self.get(path)
            .map(|v| v.parse().expect("checked while parsing"))
    }

    /// A file name, if given and nonempty.
    pub fn file(&self, path: &str) -> Option<PathBuf> {
        self.get(path).filter(|v| !v.is_empty()).map(PathBuf::from)
    }
}

/// Arguments outside CmdStan's grammar, written `--name` or `--name=value`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Extensions {
    /// Replace existing output files (CmdStan overwrites silently; cmdstan-rs refuses unless
    /// this is given).
    pub overwrite: bool,
    /// Report a failure as one JSON object on stderr.
    pub json_errors: bool,
    /// No progress output.
    pub quiet: bool,
    /// Also print the result as JSON on stdout (`log_prob`, `diagnose`, `optimize`).
    pub json: bool,
    /// `diagnose` at this JSON file of unconstrained vectors (`params_r`) instead of the
    /// initialization.
    pub unconstrained: Option<PathBuf>,
}

/// Separates the `--` extensions from the CmdStan arguments.
pub fn split_extensions(args: &[String]) -> Result<(Vec<String>, Extensions), Failure> {
    let mut rest = Vec::new();
    let mut ext = Extensions::default();
    for a in args {
        match a.as_str() {
            "--overwrite" => ext.overwrite = true,
            "--json-errors" => ext.json_errors = true,
            "--quiet" => ext.quiet = true,
            "--json" => ext.json = true,
            s if s.starts_with("--unconstrained=") => {
                ext.unconstrained = Some(PathBuf::from(&s["--unconstrained=".len()..]));
            }
            s if s.starts_with("--") => {
                return Err(Failure::usage(format!(
                    "unknown option {s:?}; the cmdstan-rs extensions are --overwrite, \
                     --json-errors, --quiet, --json, and --unconstrained=<file>"
                )));
            }
            _ => rest.push(a.clone()),
        }
    }
    Ok((rest, ext))
}

/// The argument tree as text, for `help-all`.
pub fn help_all() -> String {
    fn walk(out: &mut String, nodes: &[Node], depth: usize) {
        for n in nodes {
            let pad = "  ".repeat(depth);
            let shape = match &n.kind {
                Kind::Value(Ty::Bool) => "=<0|1>",
                Kind::Value(Ty::Text) => "=<text>",
                Kind::Value(Ty::Real) => "=<real>",
                Kind::Value(_) => "=<int>",
                Kind::Group(_) => "",
                Kind::Choice(_) => "=<option>",
            };
            out.push_str(&format!("{pad}{}{shape}  {}\n", n.name, n.help));
            match &n.kind {
                Kind::Group(c) | Kind::Choice(c) => walk(out, c, depth + 1),
                Kind::Value(_) => {}
            }
        }
    }
    let mut out = String::from(
        "CmdStan arguments (CmdStan 2.40 grammar). A method name may be given bare, as in \
         `sample num_samples=500 data file=d.json`.\n\n",
    );
    walk(&mut out, ROOT, 0);
    out.push_str(
        "\ncmdstan-rs extensions: --overwrite (replace existing outputs), --json-errors (errors \
         as JSON on stderr), --quiet, --json (results as JSON on stdout), \
         --unconstrained=<file> (diagnose at these unconstrained vectors).\nOther commands: \
         help, help-all, info [data file=<json>] [random seed=<int>].\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Result<Parsed, Failure> {
        let args: Vec<String> = s.split_whitespace().map(String::from).collect();
        parse(&args)
    }

    #[test]
    fn nested_contexts_resolve_outward() {
        let x = p(
            "sample num_samples=10 adapt delta=0.9 algorithm=hmc engine=nuts max_depth=5 \
                   data file=d.json output file=o.csv random seed=3",
        )
        .unwrap();
        assert_eq!(x.method(), Some("sample"));
        assert_eq!(x.usize("method.sample.num_samples"), Some(10));
        assert_eq!(x.real("method.sample.adapt.delta"), Some(0.9));
        assert_eq!(
            x.usize("method.sample.algorithm.hmc.engine.nuts.max_depth"),
            Some(5)
        );
        assert_eq!(x.get("data.file"), Some("d.json"));
        assert_eq!(x.get("output.file"), Some("o.csv"));
        assert_eq!(x.int("random.seed"), Some(3));
    }

    #[test]
    fn errors_are_not_ignored() {
        assert!(p("sample num_samples=-1").is_err());
        assert!(p("sample bogus=1").is_err());
        assert!(p("sample num_samples=1 num_samples=2").is_err());
        assert!(p("sample algorithm=nope").is_err());
        assert!(p("max_depth=3 sample").is_err());
        assert!(p("sample output diagnostic_file=x.csv").is_err());
        assert!(p("sample adapt=1").is_err());
    }
}
