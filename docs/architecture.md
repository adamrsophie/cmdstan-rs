# Architecture

## Crates

* `cmdstan-runner` (library): the runtime of every model executable. It parses CmdStan's
  argument grammar into typed settings, builds the model from the data, calls stan-services for
  the requested method, writes output files and the run manifest, reports progress on stderr,
  and exits with a documented code. It has no inference code of its own.
* `cmdstan-rs` (binary): the unified command line. It compiles programs (stanc-rs as a
  library, then Cargo), manages the build cache, translates its options into the CmdStan
  arguments of a model executable and runs it as a child process, and summarizes Stan CSV files
  with stan-services' reader and estimators.

## One execution path

```text
model.stan ──stanc-rs──▶ model.rs ──Cargo──▶ model executable (cmdstan-runner + stan-services)
                                                  ▲
cmdstan-rs sample MODEL --chains 4 ... ──translates──┘   ./model sample num_chains=4 ...
```

Mode A (`cmdstan-rs sample`) and mode B (`./model sample ...`) run the same executable with
the same arguments, so there is one implementation of every method. The executable is a native
program, not a library loaded into cmdstan-rs: no unsafe dynamic loading and no ABI coupling
between the CLI and the math library.

A generated project contains:

```rust
mod model {
    use cmdstan_runner::stanc_rs_runtime;   // the generated code says `use stanc_rs_runtime as rt;`
    include!("model.rs");
}
fn main() { cmdstan_runner::run(BuildInfo { .. }, model::Model::new) }
```

The runner wraps the generated model with stan-model-rs-stanc's `Stanc` adapter, which
presents it through stan-model-rs' `Model` trait that stan-services uses. The compiler, the
runtime the generated code targets, and the adapter are pinned to one stanc-rs revision.

## Build system and cache

`compile` computes a SHA-256 key over: the source text and its path, every `#include`d file's
contents, the compiler options, the artifact name and the resolved stanc model name, the
stanc-rs, stan-model-rs, and stanmath-rs revisions, the sources of the path dependencies
(runner, stan-services, stan-inference: Cargo cannot pin these) and the sibling workspaces'
manifests, the workspace `Cargo.lock`, the generated `main.rs`, `rustc -vV` (compiler version
and host target), `rustc --print cfg` with the Rust flags from the environment (so with
`RUSTFLAGS="-C target-cpu=native"` the CPU features of this machine are part of the key;
`target-cpu=native` set in a Cargo configuration file is not, so do not share such a cache
between machines), the build profile, every
`CARGO_*` and `RUST*` environment variable that can change a build, and the Cargo
configuration files in `$CARGO_HOME`. Cargo runs with the generated project as its working
directory, so a `.cargo/config.toml` of the directory cmdstan-rs was started from never
applies. Unreadable inputs fail the build instead of being skipped, and included files are
hashed again after code generation. A changed input gives a new entry; the file name alone
never selects an entry. The cmdstan-rs binary contains the code generator and the cmdstan-rs
source tree supplies the runtime, so compilation refuses a tree whose `Cargo.lock` pins other
stanc-rs or stan-model-rs revisions than the binary.

An entry is built under an exclusive per-entry file lock (`File::lock`), so concurrent
compilations of the same model wait for one build; every build also holds a shared cache-wide
lock, which `cache clear` takes exclusively. A build runs in a staging directory that replaces
the entry only after it succeeds (the old entry is moved aside first and deleted last), so a
failed rebuild keeps the previous executable; a failed
build is kept as `models/.failed-<entry>` with its `build.log`. The executable path comes from
Cargo's JSON messages (so a configured `build.target` is honored). The manifest is written
last; an entry without one is incomplete and is rebuilt. The executable is always
`bin/<file>` inside its entry, and its hash is checked against the manifest before every
cached use. All entries share one Cargo target directory, so the math library compiles once;
Cargo locks it during builds. Each entry's binary has a unique package name, so builds of
different models never overwrite each other's outputs.

The generated project depends only on `cmdstan-runner` (by path) and copies the workspace
`Cargo.lock`, so it resolves to the same pinned versions; the lock Cargo produces for it is
recorded in the manifest. Nothing else is added to it.

## Security

Compiling a model runs Cargo and rustc on generated code and produces a native executable that
can do anything the user can. The Stan program cannot inject Rust code (stanc-rs generates all
of it), and `--model-name`, which reaches a generated comment, must be a Stan identifier. But
cmdstan-rs provides **no sandbox**: compile and run untrusted programs in an isolated
environment (a container or virtual machine). Subprocesses are started with argument vectors,
never through a shell, and model executables by absolute path. Output files are never replaced
without `--overwrite`, and an output never replaces an input. The cache only removes plain
directories inside a marked cache root: it refuses symbolic links for its `models`, `target`,
and `locks` directories and for entries, and never follows them. Absolute paths of the
source, included files, and include directories are recorded in every executable and copied
into run manifests, so a shared executable or manifest shows the directory layout. The
manifest records data file paths and hashes, never data values; but data validation errors can quote an offending
value, and Stan `print()` statements print whatever the program prints. Dependencies are
pinned git revisions and locked crates.io versions; the first build fetches them from the
network, and `--offline` forbids that.
