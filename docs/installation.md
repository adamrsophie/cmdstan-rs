# Installation

## Requirements

* Rust 1.89 or newer with Cargo (`rustup` from <https://rustup.rs>). The toolchain is needed
  whenever a model is compiled, because every model becomes a native Rust executable. A
  prebuilt `cmdstan-rs` binary does not remove this requirement.
* Network access for the first build, to fetch the pinned git and crates.io dependencies. After
  that, `--offline` builds without network access.
* No C++ compiler, OCaml, Python, R, or Julia.

## Source layout

stan-services-rs and stan-inference-rs are not published yet, so cmdstan-rs uses them from
sibling checkouts:

```text
<dir>/cmdstan-rs
<dir>/stan-services-rs
<dir>/stan-inference-rs
```

stanc-rs, stan-model-rs, and stanmath-rs are fetched from GitHub at pinned revisions (see
`Cargo.toml`).

## Building

```sh
cd cmdstan-rs
cargo build --release
./target/release/cmdstan-rs version
```

or `cargo install --path crates/cmdstan-rs`.

Generated model projects depend on the `cmdstan-runner` crate of this source tree by path. An
installed binary finds it through the path recorded at build time; set `CMDSTAN_RS_HOME` to the
cmdstan-rs checkout if the tree moved.

## Cache location

Compiled models live in the build cache: `--cache-dir`, else `$CMDSTAN_RS_CACHE`, else
`~/Library/Caches/cmdstan-rs` (macOS), `$XDG_CACHE_HOME/cmdstan-rs` or `~/.cache/cmdstan-rs`
(Linux), `%LOCALAPPDATA%\cmdstan-rs` (Windows). `cmdstan-rs cache dir` prints it.

## Platforms

The code uses only the Rust standard library for paths, processes, and file locking, and
handles executable suffixes. It has not been tested on any platform yet (see
[implementation-status.md](implementation-status.md)).
