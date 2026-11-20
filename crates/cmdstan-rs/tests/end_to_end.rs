//! End-to-end tests of the `cmdstan-rs` binary.
//!
//! The fast tests check exit codes that need no model build. The `#[ignore]` tests compile the
//! reference normal model with stanc-rs, build it with Cargo (the first build compiles the math
//! library and takes minutes, and needs the git dependencies fetched), sample, and check the
//! posterior against the closed form. Run them with
//! `cargo test -p cmdstan-rs --release -- --ignored`.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_cmdstan-rs");

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(p: &str) -> PathBuf {
    repo().join("tests/fixtures").join(p)
}

fn tmp(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// One cache for every test, so the model builds once.
fn cache() -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cache");
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .arg("--cache-dir")
        .arg(cache())
        .args(args)
        .output()
        .expect("the cmdstan-rs binary runs")
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

#[test]
fn missing_source_is_exit_3() {
    let o = run(&["check", "no/such/model.stan"]);
    assert_eq!(code(&o), 3, "{}", String::from_utf8_lossy(&o.stderr));
}

#[test]
fn compiler_error_is_exit_4() {
    let o = run(&["check", s(&fixture("bad/syntax_error.stan"))]);
    assert_eq!(code(&o), 4, "{}", String::from_utf8_lossy(&o.stderr));
}

#[test]
fn invalid_option_is_exit_2() {
    let o = run(&["sample", "normal", "--chains", "four"]);
    assert_eq!(code(&o), 2);
}

#[test]
fn check_accepts_the_reference_model() {
    let o = run(&["check", "--info", s(&fixture("normal/normal.stan"))]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    let info: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert!(info["parameters"]["mu"].is_object());
}

fn data_rows(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

#[test]
#[ignore = "builds a model executable (minutes on the first run)"]
fn normal_model_end_to_end() {
    let src = fixture("normal/normal.stan");
    let data = fixture("normal/data.json");

    // 1-3. The compiler accepts the model, Rust is generated, and it builds.
    let o = run(&["compile", s(&src)]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    let exe = PathBuf::from(String::from_utf8(o.stdout).unwrap().trim());
    assert!(exe.is_file());

    // 4-6. Valid data, four chains, draws written.
    let out = tmp("normal-results");
    // Sampled through the .stan path, so the executable is the one built from this tree.
    let sample_with = |dir: &Path, parallel: &str| {
        run(&[
            "sample",
            s(&src),
            "--data",
            s(&data),
            "--chains",
            "4",
            "--parallel-chains",
            parallel,
            "--warmup",
            "1000",
            "--draws",
            "2500",
            "--seed",
            "12345",
            "--output",
            s(dir),
            "--quiet",
        ])
    };
    let sample = |dir: &Path| sample_with(dir, "4");
    let o = sample(&out);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    for id in 1..=4 {
        let f = out.join(format!("normal_chain_{id}.csv"));
        assert!(f.is_file(), "{} missing", f.display());
        // 7. The header has the model parameter after the sampler columns.
        let rows = data_rows(&f);
        let header: Vec<&str> = rows[0].split(',').collect();
        assert_eq!(
            header,
            [
                "lp__",
                "accept_stat__",
                "stepsize__",
                "treedepth__",
                "n_leapfrog__",
                "divergent__",
                "energy__",
                "mu",
                "y_rep"
            ]
        );
        assert_eq!(rows.len(), 1 + 2500);
    }
    assert!(out.join("normal_chain_manifest.json").is_file());

    // 8-10. The posterior agrees with the closed form; diagnostics are available.
    let o = run(&["summary", "--format", "json", s(&out)]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    let mu = v["variables"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["name"] == "mu")
        .unwrap();
    // Posterior N(1.5, 1/2): the prior and the likelihood both matter (the likelihood alone
    // gives N(3, 1), the prior alone N(0, 1)). Tolerances do not trust the implementation's
    // MCSE: they assume only the effective sample size asserted below (2000 of the 10000
    // draws; NUTS on a one-dimensional normal typically gives 3500 or more) and allow 4.5
    // standard errors.
    // Dropping the prior moves the mean by 1.5, about 70 standard errors.
    let post_sd = f64::sqrt(0.5);
    let post_mean = 1.5;
    let ess = 2000.0_f64;
    // The draws, read directly from the CSV files (independent of the summary code).
    let column = |name: &str, rows: &[String]| -> Vec<f64> {
        let k = rows[0].split(',').position(|c| c == name).unwrap();
        rows[1..]
            .iter()
            .map(|r| r.split(',').nth(k).unwrap().parse::<f64>().unwrap())
            .collect()
    };
    let mut draws = Vec::new();
    let mut y_rep = Vec::new();
    for id in 1..=4 {
        let rows = data_rows(&out.join(format!("normal_chain_{id}.csv")));
        draws.extend(column("mu", &rows));
        y_rep.extend(column("y_rep", &rows));
    }
    assert_eq!(draws.len(), 10000);
    let n = draws.len() as f64;
    let mean_sd = |x: &[f64]| {
        let m = x.iter().sum::<f64>() / x.len() as f64;
        let v = x.iter().map(|d| (d - m).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0);
        (m, v.sqrt())
    };
    let (raw_mean, raw_sd) = mean_sd(&draws);
    let mut sorted = draws.clone();
    sorted.sort_by(f64::total_cmp);
    let quantile = |p: f64| {
        // Type 7 (linear interpolation), as the posterior package and summary use.
        let h = (n - 1.0) * p;
        let lo = h.floor() as usize;
        sorted[lo] + (h - lo as f64) * (sorted[(lo + 1).min(sorted.len() - 1)] - sorted[lo])
    };
    // The summary reports what the draws say.
    for (key, want) in [
        ("mean", raw_mean),
        ("sd", raw_sd),
        ("median", quantile(0.5)),
        ("q5", quantile(0.05)),
        ("q95", quantile(0.95)),
    ] {
        let got = mu[key].as_f64().unwrap();
        assert!(
            (got - want).abs() < 1e-5 * post_sd,
            "summary {key} {got} vs draws {want}"
        );
    }
    // The draws agree with the closed form.
    let se_mean = post_sd / ess.sqrt();
    assert!(
        (raw_mean - post_mean).abs() < 4.5 * se_mean,
        "mean {raw_mean} vs {post_mean} (se {se_mean})"
    );
    // sd of a normal sample: se(sd) ~ sd / sqrt(2 ess).
    let se_sd = post_sd / (2.0 * ess).sqrt();
    assert!(
        (raw_sd - post_sd).abs() < 4.5 * se_sd,
        "sd {raw_sd} vs {post_sd} (se {se_sd})"
    );
    // 5% and 95% quantiles: se ~ sqrt(p (1 - p) / ess) / density at the quantile.
    let z = 1.644_853_626_951_472_2_f64;
    let density = (-0.5 * z * z).exp() / (post_sd * (2.0 * std::f64::consts::PI).sqrt());
    let se_q = (0.05 * 0.95 / ess).sqrt() / density;
    for (p, exact) in [
        (0.05, post_mean - z * post_sd),
        (0.95, post_mean + z * post_sd),
    ] {
        let q = quantile(p);
        assert!(
            (q - exact).abs() < 4.5 * se_q,
            "q{p} {q} vs {exact} (se {se_q})"
        );
    }
    // The generated quantity is the posterior predictive N(1.5, 1.5): sd sqrt(1.5), and each
    // draw adds independent noise, so its ESS is at least that of mu.
    let (rep_mean, rep_sd) = mean_sd(&y_rep);
    let rep_sd_exact = 1.5_f64.sqrt();
    assert!(
        (rep_mean - 1.5).abs() < 4.5 * rep_sd_exact / ess.sqrt(),
        "y_rep mean {rep_mean}"
    );
    assert!(
        (rep_sd - rep_sd_exact).abs() < 4.5 * rep_sd_exact / (2.0 * ess).sqrt(),
        "y_rep sd {rep_sd}"
    );
    // The MCSE is consistent with the ESS the summary reports.
    let mcse = mu["mcse_mean"].as_f64().unwrap();
    let ess_bulk = mu["ess_bulk"].as_f64().unwrap();
    assert!(ess_bulk > ess, "ess_bulk {ess_bulk}");
    assert!(mu["ess_tail"].as_f64().unwrap() > ess / 2.0);
    assert!(
        mcse > 0.5 * raw_sd / ess_bulk.sqrt() && mcse < 2.0 * raw_sd / ess_bulk.sqrt(),
        "mcse_mean {mcse} vs sd / sqrt(ess_bulk)"
    );
    assert!(mu["rhat"].as_f64().unwrap() < 1.01);
    assert_eq!(v["chains"].as_array().unwrap().len(), 4);
    // Chains use different random streams: no two chains have the same draws.
    let chain_rows: Vec<Vec<String>> = (1..=4)
        .map(|id| data_rows(&out.join(format!("normal_chain_{id}.csv"))))
        .collect();
    for i in 0..4 {
        for j in i + 1..4 {
            assert_ne!(
                chain_rows[i][1..],
                chain_rows[j][1..],
                "chains {i} and {j} are identical"
            );
        }
    }

    // 11. The same seed reproduces the draws exactly, whether the chains run in parallel or
    // one after another.
    let again = tmp("normal-results-again");
    let o = sample_with(&again, "1");
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    for id in 1..=4 {
        let f = format!("normal_chain_{id}.csv");
        assert_eq!(data_rows(&out.join(&f)), data_rows(&again.join(&f)));
    }

    // summary --strict passes on these chains and refuses a chain given twice.
    let o = run(&["summary", "--strict", s(&out)]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    let c1 = out.join("normal_chain_1.csv");
    let o = run(&["summary", s(&c1), s(&again.join("normal_chain_1.csv"))]);
    assert_eq!(code(&o), 2, "{}", String::from_utf8_lossy(&o.stderr));

    // generate-quantities numbers outputs by the input chains' ids and records each input.
    let gq = tmp("normal-gq");
    let o = run(&[
        "generate-quantities",
        s(&src),
        "--data",
        s(&data),
        "--draws",
        s(&out.join("normal_chain_3.csv")),
        s(&out.join("normal_chain_4.csv")),
        "--output",
        s(&gq.join("normal_gq.csv")),
        "--quiet",
    ]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    for id in [3, 4] {
        let f = gq.join(format!("normal_gq_{id}.csv"));
        assert!(f.is_file(), "{} missing", f.display());
        let text = std::fs::read_to_string(&f).unwrap();
        assert!(
            text.contains(&format!("normal_chain_{id}.csv")),
            "{} does not name its input",
            f.display()
        );
        assert_eq!(data_rows(&f)[0], "y_rep");
    }

    // optimize finds the posterior mode 1.5 (no Jacobian for an unconstrained parameter).
    let opt = tmp("normal-optimize");
    let o = run(&[
        "optimize",
        s(&src),
        "--data",
        s(&data),
        "--output",
        s(&opt.join("normal_optimize.csv")),
        "--quiet",
    ]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    let rows = data_rows(&opt.join("normal_optimize.csv"));
    let mode = column("mu", &rows)[0];
    assert!((mode - 1.5).abs() < 1e-4, "optimize mode {mode}");

    // 12. Failures have their exit codes: existing outputs (9), invalid data (6), unknown
    // model (3).
    let o = sample(&out);
    assert_eq!(code(&o), 9, "{}", String::from_utf8_lossy(&o.stderr));
    let bad = tmp("normal-bad-data");
    let o = run(&[
        "sample",
        "normal",
        "--data",
        s(&fixture("bad/bad_data.json")),
        "--output",
        s(&bad),
        "--quiet",
    ]);
    assert_eq!(code(&o), 6, "{}", String::from_utf8_lossy(&o.stderr));
    let o = run(&["sample", "no_such_model", "--output", s(&bad)]);
    assert_eq!(code(&o), 3);
    // A bare name never runs a file of that name in the working directory.
    let o = Command::new(env!("CARGO_BIN_EXE_cmdstan-rs"))
        .args(["--cache-dir", s(&cache()), "sample", "normal_chain_1.csv"])
        .current_dir(&out)
        .output()
        .unwrap();
    assert_eq!(code(&o), 3);

    // The exported standalone executable accepts CmdStan's syntax directly.
    let direct = tmp("normal-direct");
    let o = run(&[
        "compile",
        s(&src),
        "--standalone",
        "--output",
        s(&direct.join("normal_model")),
    ]);
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    let exported = PathBuf::from(String::from_utf8(o.stdout).unwrap().trim());
    assert!(exported.starts_with(&direct), "{}", exported.display());
    assert_ne!(exported, exe);
    let o = Command::new(&exported)
        .args([
            "sample",
            "num_samples=200",
            "num_warmup=200",
            "data",
            &format!("file={}", s(&data)),
            "random",
            "seed=7",
            "output",
            &format!("file={}", s(&direct.join("out.csv"))),
            "refresh=0",
        ])
        .output()
        .unwrap();
    assert_eq!(code(&o), 0, "{}", String::from_utf8_lossy(&o.stderr));
    assert!(direct.join("out_1.csv").is_file());
}
