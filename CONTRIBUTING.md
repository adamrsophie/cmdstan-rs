# Contributing

* Run `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, and
  `cargo test --workspace` before sending a change. `cargo test -p cmdstan-rs --release --
  --ignored` runs the end-to-end tests that build model executables.
* Parsing and code generation belong in stanc-rs, model evaluation in stan-model-rs,
  algorithms and estimators in stan-inference-rs, chain orchestration and Stan CSV writing in
  stan-services-rs. cmdstan-rs orchestrates; it does not reimplement them.
* Never accept an option and ignore it: unsupported options are errors.
* Never report a partial run as complete, and never overwrite user files without
  `--overwrite`.
* Statistical tests compare with known values in Monte Carlo standard errors, with seeds fixed
  in advance.
* No `unsafe` code (the workspace forbids it).
