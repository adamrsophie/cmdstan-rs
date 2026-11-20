# Implementation status

As of 2026-11-19. **No test has been run.** The workspace type checks, and `cargo fmt --check`,
`cargo clippy --workspace --all-targets -- -D warnings`, and `cargo doc` pass. No model has
been compiled into an executable and no inference has been run with cmdstan-rs, so no
numerical or statistical result exists yet.

## Capability matrix

| Capability | Status |
| --- | --- |
| `compile` (stanc-rs library, generated Cargo project, cache, lock, manifest, `--standalone`) | experimental, untested |
| `check`, `inspect`, `cache`, `version`, `inspect-run` | experimental, untested |
| `sample`: NUTS, static HMC, fixed_param, multiple and parallel chains, Stan CSV via stan-services | experimental, untested |
| `summary` (stan-services estimators) | experimental, untested |
| `optimize`, `variational`, `pathfinder`, `laplace`, `generate-quantities`, `log-prob`, `diagnose` | experimental, untested |
| CmdStan argument grammar in model executables, `run` passthrough | experimental, untested |
| JSON data | experimental (stan-model-rs reader) |
| RDump data | unsupported |
| Diagnostic files, profiling, single-path Pathfinder output, optimizer iterates | unsupported (rejected) |
| CmdStanR / posterior reading of the CSV files | planned (needs R) |
| Differential validation against CmdStan, benchmarks | planned |
| Prebuilt binaries, release pipeline | planned |
| Tested operating systems | none yet |

## Stan language support

Whatever stanc-rs at revision 242398e generates code for (its own corpus results); not
retested through cmdstan-rs. Functions declared without definitions cannot be built.

## Tests written, not run

* `crates/cmdstan-runner`: argument grammar unit tests.
* `crates/cmdstan-rs/tests/end_to_end.rs`: exit codes of `check` and invalid options (fast);
  the full normal model workflow (compile, four chains, CSV layout, closed-form posterior mean,
  sd, and 5% quantile within Monte Carlo error, R-hat and ESS, reproducibility with the same
  seed, exit codes 9, 6, 3, and the standalone executable) under `--ignored`.

## Integration blockers

* stan-services-rs (c0cd1dc) and stan-inference-rs (6fb9c3e) are used by sibling path; CI
  checks out both at those revisions.
* The pinned stan-model-rs adapter drops imaginary parts when reading flat complex values and
  rejects flat tuples; cmdstan-rs refuses CSV inputs for such models (to fix upstream).
* The pinned stanc-rs runtime prints Stan `print()` output to stdout and interpolates the
  model name into a generated comment (cmdstan-rs validates the name); a message sink is an
  upstream task.
* The JSON reader accepts only quoted nonfinite values; duplicate variables are detected by
  cmdstan-rs before the reader runs.
* Generated projects depend on `cmdstan-runner` by path, so the source tree must be present
  where models are compiled (`CMDSTAN_RS_HOME`).
* stanc-rs models provide no Hessian; Newton and Laplace rely on finite differences.
* The stanc-rs adapter reports no constraints in the metadata.

## Next tasks

1. Run `cargo test --workspace` and the ignored end-to-end test; fix what fails.
2. Validate the normal model against the closed form and record the numbers here.
3. Add end-to-end tests for the Bernoulli, regression, and hierarchical examples, constrained
   parameters, transformed parameters, generated quantities, and every non-sampling method.
4. Read the CSV files with CmdStanR and `posterior`, and document the result.
5. Differential tests against CmdStan 2.40 (log densities, gradients, CSV structure,
   summaries).
6. Publish stan-services-rs and stan-inference-rs, pin them, and enable CI on three platforms.
