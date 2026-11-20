# CmdStan compatibility

Reference: **CmdStan 2.40.0** (argument grammar and Stan CSV layout). Six separate claims, of
which none has been demonstrated yet because nothing has been run:

| Claim | Status |
| --- | --- |
| Language: programs stanc3 accepts compile | stanc-rs' claim (its corpus results); not retested here |
| Mathematical: log densities and gradients match Stan | stanc-rs and stan-model-rs' claim; not retested here |
| Statistical: posterior summaries agree with CmdStan | not validated |
| Output: Stan CSV files readable by CmdStanR, posterior, ArviZ | not validated |
| Command line: CmdStan argument syntax | implemented, untested (table below) |
| Bitwise reproducibility with CmdStan | not a goal: different RNG and algorithms |

## Arguments

| CmdStan argument | cmdstan-rs |
| --- | --- |
| `method=sample` (or bare `sample`) with `num_samples`, `num_warmup`, `save_warmup`, `thin`, `num_chains` | supported |
| `adapt engaged gamma delta kappa t0 init_buffer term_buffer window` | supported |
| `adapt save_metric=1` | writes `<prefix>_metric_<id>.json` (`stepsize`, `metric_type`, `inv_metric`) |
| `algorithm=hmc engine=nuts max_depth` / `engine=static int_time` | supported |
| `metric=unit_e,diag_e,dense_e`, `metric_file`, `stepsize`, `stepsize_jitter` | supported (one metric file for all chains) |
| `algorithm=fixed_param` | supported; also chosen automatically for models without parameters |
| `optimize algorithm=lbfgs,bfgs,newton`, tolerances, `history_size`, `jacobian`, `iter` | supported |
| `optimize save_iterations=1` | rejected (not available from stan-services) |
| `variational` (all arguments) | supported |
| `pathfinder` (all arguments except below) | supported |
| `pathfinder save_single_paths=1`, `calculate_lp=0` | rejected |
| `laplace mode jacobian draws` | supported; `mode` may also be omitted (optimize first) |
| `laplace calculate_lp=0` | rejected |
| `generate_quantities fitted_params` | supported; several files comma separated |
| `log_prob unconstrained_params constrained_params jacobian` | supported |
| `diagnose test=gradient epsilon error` | supported |
| `id`, `data file`, `init`, `random seed`, `output file refresh sig_figs`, `num_threads` | supported; `num_threads` sets the chains run at once |
| `output diagnostic_file`, `profile_file`, `save_cmdstan_config=1` | rejected |
| `help`, `help-all`, `info` | supported (`info` prints JSON) |

Unknown arguments, values of the wrong type, repeated arguments, and arguments in the wrong
context are errors. Recognized but unsupported options are errors, never no-ops.

## Differences from CmdStan

* Output files: chain `i` is always `<stem>_<i>.csv`, also with one chain (CmdStan writes
  `<stem>.csv` for one chain). Existing files are not overwritten without `--overwrite`.
* Progress and messages go to stderr, not stdout.
* Default `sig_figs` is 8 (stan-services'). Values that need more digits to read back (a
  probability within 1e-8 of 0 or 1, for example) need a larger `sig_figs`; `-1` also means 8
  here, not the stream default. `sig_figs=0` is rejected.
* Default seed: from the clock, as CmdStan's `seed=-1`, and always recorded.
* Initialization files: one file for every chain; CmdStan's per-chain `init_<id>.json`
  expansion is not implemented.
* Argument ranges follow CmdStan 2.40: `id` from 0, `seed` 0 to 4294967295 or -1,
  `num_threads` positive or -1 (all CPUs), `adapt delta` in [0.1, 1), `stepsize_jitter` in
  [0, 1]. `save_metric=1` needs adaptation; `num_warmup=0` needs `adapt engaged=0`;
  `metric_file` with `fixed_param` is an error.
* Not accepted: `generate_quantities num_chains` (give one draw file per chain) and
  `laplace draws=0`.
* Grammar differences: a repeated argument is an error (CmdStan sometimes takes the last),
  `sample help` is not a help request, and a group closed by a later argument cannot be
  reopened.
* Models with complex or tuple parameters cannot yet take Stan CSV inputs (Laplace mode,
  `log_prob` draws, generated quantities): the pinned stan-model-rs adapter loses imaginary
  parts when it reads flat values back. JSON inputs work.
* Stan `print()` statements write to stdout (the pinned stanc-rs runtime has no message
  sink); machine-readable output that the CLI parses is always the last line.
* Method tables follow CmdStan 2.40's columns: optimize `lp__, converged__` (Stan's
  optimizer return code), variational `lp__, log_p__, log_g__` (with `log_g__` the standard
  normal log density of the draw before transformation, without constants, as CmdStan),
  Pathfinder `lp_approx__, lp__, path__` (paths numbered from `id`; one path gives that path's
  draws without resampling), Laplace `log_p__, log_q__`, generated quantities only the
  generated quantities block, `log_prob` `lp__, g_<unconstrained name>`. Comment lines differ.
* `generate_quantities` with several files numbers the chains from `id` when the files share
  one id (as CmdStan's multi-chain files do) or when `id` is given; it refuses to write over an
  input file, and fails (exit 8) if every draw fails.
* `log_prob` takes `unconstrained_params` or `constrained_params`, not both; a CSV input must
  be complete.
* `diagnose` compares the gradient with finite differences of the full log density (CmdStan
  uses the density without constants for the model gradient; the gradients are equal), and
  exits 11 when a component is flagged or a point cannot be evaluated.
* Laplace `mode` CSV files must come from `optimize` (method recorded, last `converged__`
  not negative, same Jacobian setting). JSON mode files record no Jacobian setting and are
  taken as the mode of the requested density.
* Newton and Laplace use finite-difference Hessians (stanc-rs models have no Hessian).
* The stanc-rs adapter reports every variable as unconstrained real values in the metadata
  (constraints are still enforced by the generated code).

## Known CmdStan issues addressed

Open issues in stan-dev/cmdstan and stan-dev/stan (read 2026-11-19) that bear on a command line
and how cmdstan-rs handles them:

| Issue | cmdstan-rs |
| --- | --- |
| cmdstan#1185: thinning changes reported divergences, tree depth hits, and E-BFMI | `summary` takes them over every post-warmup transition from the run metadata |
| cmdstan#1257: one id in all files of a multi-chain run | `generate_quantities` renumbers such files from `id`; `summary` accepts them |
| cmdstan#1278: the `diagnose` utility always exits 0 | `summary --strict` exits 11 when a check fails; the `diagnose` method exits 11 when a gradient component is flagged or cannot be computed |
| cmdstan#1366: `diagnose` utility thresholds (R-hat 1.015, ESS N/1000) differ from the guide | `summary` warns at R-hat above 1.01 and ESS below 100 per chain, as documented |
| cmdstan#822: no running count of divergences | progress lines report divergences so far |
| cmdstan#1257: same `id` in every chain's CSV | cmdstan-rs chain files record their own id (stan-services) |
| cmdstan#1294: binaries considered up to date after build flags change | the cache key includes the profile, `RUSTFLAGS`, toolchain, and dependency revisions |
| cmdstan#1178: `info` method | `info` prints variables and build provenance as JSON; `cmdstan-rs inspect` |
| cmdstan#948: `diagnose` utility counts warmup draws | summaries use post-warmup draws only |
| cmdstan#991: interpret return codes | documented, specific exit codes |
| stan#3376: missing timing in some outputs | optimize, variational, pathfinder, laplace, and generated quantities outputs end with the elapsed time |

Not addressed yet: cmdstan#912 (summaries of standalone generated quantities output),
cmdstan#1035 (compressed data files), cmdstan#1025 (`help-all` as JSON), cmdstan#553
(selecting which variables to save).

## Data formats

JSON in CmdStan's format, read and validated by stan-model-rs (and checked again by the
generated model). Nonfinite values must be written as the strings `"NaN"`, `"Inf"`, and
`"-Inf"`; bare `NaN` and `Infinity` tokens are rejected. A variable defined twice is an
error (a JSON reader would silently keep one value). RDump is not supported.

## R and Python tools

The Stan CSV files of `sample` are written by stan-services in CmdStan 2.40's layout. Whether
CmdStanR's `as_cmdstan_fit()` and `cmdstanr::read_cmdstan_csv()` accept them has not been
tested. CmdStanR and CmdStanPy cannot compile or launch cmdstan-rs models; that would be a
separate integration.
