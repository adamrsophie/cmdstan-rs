//! Progress messages on stderr.
//!
//! The observer only formats events that stan-services reports between transitions; it never
//! touches the sampler. Messages go to stderr so that stdout stays free for machine readable
//! output (CmdStan prints progress to stdout; this is a documented difference). Output is plain
//! text without terminal control codes, so logs from CI and batch jobs stay readable.

use std::io::Write;
use std::sync::Mutex;

use stan_services::{
    AdaptationUpdate, ChainError, ChainPhase, ChainStatus, InitReport, RunObserver, RunProgress,
};

/// Prints CmdStan-style progress lines.
pub struct StderrProgress {
    /// Report every this many iterations (0: phase changes and errors only).
    pub refresh: usize,
    /// Print nothing but errors.
    pub quiet: bool,
    // Serializes lines from parallel chains so they never interleave.
    lock: Mutex<()>,
}

impl StderrProgress {
    /// An observer with the given refresh interval.
    pub fn new(refresh: usize, quiet: bool) -> Self {
        StderrProgress {
            refresh,
            quiet,
            lock: Mutex::new(()),
        }
    }

    fn line(&self, s: &str) {
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut err = std::io::stderr().lock();
        // A closed stderr must not fail the run.
        let _ = writeln!(err, "{s}");
    }
}

impl RunObserver for StderrProgress {
    fn on_initialized(&self, chain_id: u32, report: &InitReport) {
        if !self.quiet && report.attempts > 1 {
            self.line(&format!(
                "Chain [{chain_id}] initialized after {} attempts ({} rejected)",
                report.attempts,
                report.rejected.len()
            ));
        }
    }

    fn on_progress(&self, p: &RunProgress) {
        if self.quiet || self.refresh == 0 {
            return;
        }
        let label = match p.phase {
            ChainPhase::Warmup => "Warmup",
            ChainPhase::Sampling => "Sampling",
            _ => return,
        };
        // Report the first iteration, every `refresh`-th, and the last of each phase.
        let it = p.iteration;
        if !(it == 0 || it == p.total || it.is_multiple_of(self.refresh)) {
            return;
        }
        let pct = (100 * it).checked_div(p.total).unwrap_or(100);
        // Divergences so far in this phase, reported while the chain runs
        // (stan-dev/cmdstan#822).
        let div = if p.divergences > 0 {
            format!("  {} divergent", p.divergences)
        } else {
            String::new()
        };
        self.line(&format!(
            "Chain [{}] Iteration: {it:>w$} / {} [{pct:>3}%]  ({label})  {:.1} s{div}",
            p.chain_id,
            p.total,
            p.elapsed.as_secs_f64(),
            w = p.total.to_string().len()
        ));
    }

    fn on_adaptation(&self, u: &AdaptationUpdate) {
        if !self.quiet && self.refresh > 0 {
            self.line(&format!(
                "Chain [{}] warmup finished, step size {:.4}",
                u.chain_id, u.step_size
            ));
        }
    }

    fn on_warning(&self, chain_id: u32, message: &str) {
        if !self.quiet {
            self.line(&format!("Chain [{chain_id}] warning: {message}"));
        }
    }

    fn on_chain_error(&self, e: &ChainError) {
        let at = e
            .iteration
            .map(|i| format!(" at iteration {i}"))
            .unwrap_or_default();
        self.line(&format!(
            "Chain [{}] failed during {}{at}: {:?}: {}",
            e.chain_id,
            e.phase.as_str(),
            e.category,
            e.message
        ));
    }

    fn on_chain_finish(&self, chain_id: u32, status: &ChainStatus) {
        if !self.quiet || !status.is_completed() {
            self.line(&format!("Chain [{chain_id}] {}", status.as_str()));
        }
    }
}
