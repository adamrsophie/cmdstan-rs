# Changelog

## Unreleased

* First implementation: `cmdstan-runner` (CmdStan argument grammar, method dispatch to
  stan-services, output files, run manifests, exit codes) and the `cmdstan-rs` command line
  (compilation with stanc-rs and Cargo, build cache, model resolution, summaries). Untested.
* Cache: symbolic link checks, cache-wide lock, staged builds, a key covering the build
  environment, Cargo home configuration, and target features; Cargo runs in the generated
  project directory.
* Model resolution: bare names always use the cache and recompile when the source, toolchain,
  or runtime changed; on Unix the CLI execs the model executable.
* Method outputs follow CmdStan 2.40 columns; generated quantities keep the input chain ids.
* Exit code 11 for failed diagnostic checks; input read errors exit 2 or 3.
