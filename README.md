# cmdstan-rs

A Rust-native platform for compiling and running [Stan](https://mc-stan.org) models: a
command-line counterpart of CmdStan built entirely on the Rust Stan ecosystem by Adam R Sophie.

This is an independent, unofficial project. It is not part of Stan, not a Stan Development Team
product, and not endorsed by the Stan Development Team or NumFOCUS.

> **Status: early development, unverified.** The code type checks and passes
> `cargo clippy -D warnings`, but no test has been run yet and no model has been sampled with
> it. See [docs/implementation-status.md](docs/implementation-status.md) before relying on
> anything here.

## Where it fits

| Crate | Role |
| --- | --- |
| [stanmath-rs](https://github.com/adamrsophie/stanmath-rs) | Automatic differentiation, distributions, special functions |
| [stanc-rs](https://github.com/adamrsophie/stanc-rs) | Compiles Stan programs to Rust |
| [stan-model-rs](https://github.com/adamrsophie/stan-model-rs) | Model runtime: metadata, data, transforms, log densities, generated quantities |
| [stan-inference-rs](https://github.com/adamrsophie/stan-inference-rs) | NUTS, static HMC, adaptation, optimization, ADVI, Pathfinder, Laplace, diagnostics |
| [stan-services-rs](https://github.com/adamrsophie/stan-services-rs) | Configuration, initialization, multi-chain execution, Stan CSV output, summaries |
| **cmdstan-rs** | Command line, model compilation and build cache, model executables |

cmdstan-rs contains no parser, no automatic differentiation, no sampler, and no estimator. It
calls stanc-rs to compile, Cargo to build, and stan-services to run inference.

## Quick start

```sh
cargo install --path crates/cmdstan-rs        # or: cargo build --release

cmdstan-rs compile tests/fixtures/normal/normal.stan
cmdstan-rs sample normal --data tests/fixtures/normal/data.json \
    --chains 4 --warmup 1000 --draws 1000 --seed 12345 --output results/
cmdstan-rs summary results/
```

`compile` translates the program with stanc-rs, generates a small Cargo project around the
Rust module, and builds a native model executable into the build cache. The first build
compiles the math library and takes minutes; later models reuse it.

Every compiled model is also a standalone executable with CmdStan's argument syntax:

```sh
cmdstan-rs compile model.stan --standalone --output build/model
./build/model sample num_warmup=1000 num_samples=1000 num_chains=4 \
    data file=data.json random seed=1 output file=out/model.csv
```

Both ways run the same code: the unified command runs the model executable.

## Commands

`compile`, `check`, `inspect`, `sample`, `optimize`, `variational`, `pathfinder`, `laplace`,
`generate-quantities`, `log-prob`, `diagnose`, `summary`, `inspect-run`, `run` (CmdStan syntax
passthrough), `cache list|inspect|clear|dir`, `version`. See
[docs/cli-reference.md](docs/cli-reference.md).

## Requirements

* A Rust toolchain (1.89 or newer) with Cargo, also at run time: compiling a model builds Rust
  code. A prebuilt `cmdstan-rs` binary does not remove this requirement.
* No C++, OCaml, Python, R, or Julia.
* Until stan-services-rs and stan-inference-rs are published, checkouts of both must sit next to
  this repository (`../stan-services-rs`, `../stan-inference-rs`), and generated model projects
  use the cmdstan-rs source tree (`CMDSTAN_RS_HOME`). See [docs/installation.md](docs/installation.md).

## Documentation

* [Installation](docs/installation.md)
* [Command reference](docs/cli-reference.md)
* [Architecture, build system, and security](docs/architecture.md)
* [CmdStan compatibility](docs/compatibility.md)
* [Reproducibility](docs/reproducibility.md)
* [Troubleshooting](docs/troubleshooting.md)
* [Implementation status](docs/implementation-status.md)

## License

GPL-3.0-only. See [LICENSE](LICENSE). Stan, CmdStan, and the Stan logo are trademarks of
NumFOCUS; they are used here only to describe compatibility.
