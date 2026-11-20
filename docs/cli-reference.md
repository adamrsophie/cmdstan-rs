# Command reference

`cmdstan-rs [--cache-dir DIR] [--json-errors] <command> ...`. Messages and progress go to
stderr; stdout carries only results (paths, JSON, tables). `--json-errors` reports a failure as
one JSON object `{"error": {"exit_code", "category", "message"}}` on stderr.

## Model arguments

Commands that run a model take `MODEL`, which is one of:

* a `.stan` file: compiled (or taken from the cache when nothing changed);
* a path to a model executable, written as a path (`./m`, `build/m`, or absolute); a bare
  name never runs a file of the working directory;
* the name of a compiled model in the cache (the source file stem, or `--name`). When the
  recorded source still exists, the name resolves like the `.stan` file with the entry's
  settings: a changed source, toolchain, or runtime gives a fresh build, never a stale
  executable. A release build is preferred over a `--dev` build. A name shared by models
  compiled from different files is an error.

Common options: `--data FILE`, `--init RADIUS|FILE`, `--seed N` (default: from the clock, and
the seed used is recorded), `--id N` (first chain identifier; default 1), `--sig-figs N` (default 8),
`--refresh N`, `--overwrite`, `--quiet`, and for `.stan` arguments `--include-path DIR`,
`--offline`, `--dev`.

Existing output files are never replaced without `--overwrite`.

## compile

```sh
cmdstan-rs compile model.stan [--name N] [--model-name M] [--include-path DIR]...
    [--dev] [--offline] [--force] [--standalone --output PATH [--overwrite]] [--verbose] [--json]
```

Prints the executable path (or the build manifest with `--json`). A Stan error exits 4 with
stanc-rs' diagnostic; a Rust build failure exits 5 with the end of Cargo's log (the full log is
`models/.failed-<entry>/build.log` in the cache; with `--verbose` Cargo's output goes to the
terminal instead).

## check

`cmdstan-rs check model.stan [--info]`: parse and type check only; `--info` prints stanc-rs'
`--info` JSON.

## inspect

`cmdstan-rs inspect MODEL [--data FILE] [--json]`: name, source and hash, compiler and runtime
versions, declared data and parameter variables, and, when the model can be constructed (with
`--data` if it has data), the unconstrained dimension and output columns.

## sample

```sh
cmdstan-rs sample MODEL --data d.json --chains 4 --parallel-chains 4 --warmup 1000 \
    --draws 1000 --thin 1 --seed 1 --adapt-delta 0.8 --max-treedepth 10 --metric diag \
    --output results/ [--output-prefix P] [--save-warmup] [--save-metric] [--no-adapt] \
    [--engine static --int-time T] [--metric-file F] [--step-size S] [--step-size-jitter J] \
    [--fixed-param]
```

`--warmup` and `--draws` count iterations (transitions) per chain; with thinning, each phase of
`n` iterations keeps `ceil(n / thin)` draws. Chain `i` writes `results/<prefix>_<i>.csv`
(default prefix `<model>_chain`), stan-services writes `<prefix>_metadata.json`, and the runner
writes `<prefix>_manifest.json`. A chain that fails leaves `<prefix>_<i>.csv.partial` and the
command exits 10 (some chains failed) or with the cause of the failure when no chain completed.

## summary

`cmdstan-rs summary results/ [--format table|json|csv] [--allow-partial] [--strict]`: mean, sd,
5%, 50%, 95% quantiles, MCSE of mean and sd, bulk and tail ESS, rank-normalized R-hat, and per
chain the divergences, maximum tree depth hits, and E-BFMI. Divergences and tree depth hits are
counted over every post-warmup transition (from the run's `*_metadata.json`), so thinning does
not hide them; without the metadata file they are counted over the retained draws and the
output says so. For a directory, the chains are the files the run's metadata lists: files
left from earlier runs are ignored, missing files are errors, and a directory holding several
runs must be given as files. Chains must have equal lengths. Undefined R-hat or ESS of a
nonconstant variable and nonfinite draws count as failed checks. Warnings use R-hat above 1.01 and bulk or tail ESS below 100 per chain, the
values the CmdStan guide recommends. `--strict` exits 11 when any check fails. Incomplete files, files of other models or
methods, mismatched columns, and chains with identical draws (one file given twice, or two runs
with the same seed and id) are errors; repeated chain ids are accepted. Failed chains are an error
unless `--allow-partial`.

## optimize

`cmdstan-rs optimize MODEL --data d.json [--algorithm lbfgs|bfgs|newton] [--jacobian]
[--iter N] [--init-alpha A --tol-obj T ... --history-size H] [--output FILE] [--json]`

Without `--jacobian` the result is the posterior mode of the constrained parameters; it is a
maximum likelihood estimate only when the model has flat priors. With `--jacobian` it is the
mode of the unconstrained density (what a Laplace approximation is centered on). The objective
written as `lp__` drops constant terms. Results are local optima. Newton uses a
finite-difference Hessian because stanc-rs models provide no Hessian.

## variational

`cmdstan-rs variational MODEL --data d.json [--algorithm meanfield|fullrank] [--iter N]
[--draws N] [--eta E --no-adapt] [--tol-rel-obj T] [--eval-elbo N] [--output FILE]`

The first row is the mean of the approximation; the other rows are approximate draws with
`log_p__` and `log_g__`. These are not MCMC draws: R-hat and ESS do not apply.

## pathfinder

`cmdstan-rs pathfinder MODEL --data d.json [--num-paths N] [--draws N] [--psis-draws N]
[--no-psis-resample] [--max-lbfgs-iters N] [--output FILE]`

Columns `lp_approx__`, `lp__`, `path__` (`id` plus the path index), then the model outputs. The
trailer and manifest report failed paths and the Pareto k diagnostic.

## laplace

`cmdstan-rs laplace MODEL --data d.json [--mode optimize.csv|mode.json] [--draws N]
[--no-jacobian] [--output FILE]`

The approximation is Gaussian on the unconstrained scale, with covariance from a
finite-difference Hessian; draws are mapped through the constraining transforms. A mode file
from `optimize` must have been computed with the same Jacobian setting. Without `--mode`, the
mode is found by L-BFGS first.

## generate-quantities

`cmdstan-rs generate-quantities MODEL --data d.json --draws results/m_chain_1.csv ...
[--output FILE]`: one output file per input chain, `<stem>_<id>.csv`, where `id` is the chain id
recorded in the draw file when the files have distinct ids (else `--id`, default 1, plus the
file's position). Each output names its input file in a `fitted_params` comment; join outputs
and draws by that, not by file order.

## log-prob

`cmdstan-rs log-prob MODEL --data d.json (--parameters params.json | --constrained values.json)
[--no-jacobian]` (exactly one of the two inputs): prints JSON with the log density (constants dropped, as CmdStan) and its
gradient with respect to the unconstrained parameters, and writes a CSV file. `params.json` is
`{"params_r": [..]}` or `{"params_r": [[..], [..]]}`.

## diagnose

`cmdstan-rs diagnose MODEL --data d.json [--parameters params.json] [--epsilon E] [--error E]
[--json]`: compares the gradient with central finite differences at the initial point (or the
given unconstrained points) and reports each component's value, both gradients, and absolute
and relative differences. It exits 11 when a component exceeds the error threshold or a point
cannot be evaluated. Agreement is evidence, not proof, of a correct gradient.

## inspect-run

`cmdstan-rs inspect-run results/ [--json]`: the software versions, source and data hashes,
seed, arguments, files, and warnings recorded in the run manifests.

## run

`cmdstan-rs run MODEL -- <CmdStan arguments>`: resolves the model and passes the arguments to
its executable unchanged, e.g. `cmdstan-rs run model.stan -- sample num_samples=500 data
file=d.json`.

## cache

`cmdstan-rs cache list | inspect NAME | clear [--name NAME] [--all] | dir`. `clear` removes only
entries inside a directory marked as a cmdstan-rs cache; `--all` also removes the shared build
directory.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | success |
| 2 | invalid command line options or configuration |
| 3 | a required file is missing |
| 4 | the Stan compiler rejected the program |
| 5 | the generated Rust project failed to build |
| 6 | invalid data |
| 7 | invalid initialization |
| 8 | the inference algorithm failed |
| 9 | output could not be written (including existing files without `--overwrite`) |
| 10 | some chains failed |
| 11 | a diagnostic check failed (`diagnose`, `summary --strict`) |
| 70 | internal error |
| 130 | interrupted (Ctrl+C) |

On Unix, cmdstan-rs replaces itself with the model executable once the model is resolved, so
every signal reaches the model, and a model killed by a signal is reported by the shell like
any other program (for example 139 for a segmentation fault). `sample` treats SIGINT, SIGTERM, and SIGHUP alike: it stops
after the current transitions, leaves partial files, and exits 130; other methods stop at once.
On Windows the model runs as a child process and Ctrl+C reaches both.

A missing input file exits 3; an unreadable or malformed one (a truncated CSV, a file without
read permission) exits 2, never 9.
