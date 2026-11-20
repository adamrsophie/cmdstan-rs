# Reproducibility

Every run writes `<output stem>_manifest.json` with: the build provenance embedded in the
executable (source hash and included files, stanc-rs, stan-model-rs, and stanmath-rs revisions,
a hash of the path-dependency sources, `rustc -vV`, target triple, profile, Rust flags), the
method, every argument as given, the seed actually used, the first chain id, SHA-256 hashes of
the data and init files (never their contents), start and end times, the exit status, the
files written, and warnings. `cmdstan-rs inspect-run DIR` prints it.

Random streams depend only on the seed, the chain identifier, and the purpose
(stan-services), so sequential and parallel runs with the same seed give the same draws on the
same build (the end-to-end test compares four parallel chains with the same chains run one at
a time). Do not expect identical draws across different hardware, operating systems,
compiler versions, optimization flags (`target-cpu`), or library revisions: floating point
results can differ in the last bits, and MCMC amplifies such differences. Compare posterior
summaries within Monte Carlo error instead.
