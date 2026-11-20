# Troubleshooting

* **The first compile takes minutes.** It builds stanmath-rs and the other dependencies into
  the shared cache target directory; later compiles reuse them.
* **Exit 5 (build failure).** The end of Cargo's log is printed; the full log is
  `models/.failed-<entry>/build.log` in the cache (`cmdstan-rs cache dir` prints the cache). Without network access, fetch once or
  pass `--offline` with a populated Cargo cache.
* **"the cmdstan-rs source tree was not found".** Set `CMDSTAN_RS_HOME` to the cmdstan-rs
  checkout.
* **Exit 9: output exists.** Choose another `--output` or pass `--overwrite`.
* **Exit 10: some chains failed.** The `.csv.partial` files end with the failure reason;
  `summary --allow-partial` summarizes the completed chains.
* **Name is ambiguous.** Two source files with the same stem were compiled; pass the `.stan`
  path instead of the name.
* **Stale model.** A run by name recompiles when the recorded source, toolchain, or runtime
  changed. If the source file was moved or deleted, a warning says the cached build is used
  unchecked; pass the `.stan` file to rebuild.
