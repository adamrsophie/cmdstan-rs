//! Process exit codes and the error type that carries them.
//!
//! Both the unified `cmdstan-rs` command and every model executable use these codes, so a
//! script sees the same status whichever it calls. The unified command forwards the exit code
//! of the model executable it runs unchanged.

use std::fmt;

use stan_services::{ErrorCategory, ServiceError};

/// Exit codes. Zero is success; every failure is nonzero and names its cause.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExitCode {
    /// Everything requested completed.
    Success,
    /// Invalid command line options or an invalid configuration; nothing ran.
    Usage,
    /// A file that must exist (model source, data, init, draws) is missing or unreadable.
    NotFound,
    /// The Stan compiler rejected the program.
    Compile,
    /// The generated Rust project failed to build.
    Build,
    /// The data are missing, malformed, or violate their declarations.
    Data,
    /// No valid initial point was found, or a given initial value is invalid.
    Init,
    /// The inference algorithm failed (no chain completed, an optimizer error, a numerical
    /// failure).
    Inference,
    /// Output could not be written.
    Output,
    /// Some chains completed and some failed: the run is not the requested number of chains.
    Partial,
    /// The command completed, but a diagnostic check failed (flagged gradient components, or
    /// convergence warnings with `summary --strict`); see stan-dev/cmdstan#1278.
    Diagnostics,
    /// The run was interrupted (Ctrl+C) or cancelled.
    Interrupted,
    /// A bug or an unexpected condition.
    Internal,
}

impl ExitCode {
    /// The numeric process exit code.
    pub fn code(self) -> i32 {
        match self {
            ExitCode::Success => 0,
            ExitCode::Usage => 2,
            ExitCode::NotFound => 3,
            ExitCode::Compile => 4,
            ExitCode::Build => 5,
            ExitCode::Data => 6,
            ExitCode::Init => 7,
            ExitCode::Inference => 8,
            ExitCode::Output => 9,
            ExitCode::Partial => 10,
            ExitCode::Diagnostics => 11,
            ExitCode::Interrupted => 130,
            ExitCode::Internal => 70,
        }
    }

    /// A short machine readable name.
    pub fn name(self) -> &'static str {
        match self {
            ExitCode::Success => "success",
            ExitCode::Usage => "usage",
            ExitCode::NotFound => "not_found",
            ExitCode::Compile => "compile",
            ExitCode::Build => "build",
            ExitCode::Data => "data",
            ExitCode::Init => "init",
            ExitCode::Inference => "inference",
            ExitCode::Output => "output",
            ExitCode::Partial => "partial",
            ExitCode::Diagnostics => "diagnostics",
            ExitCode::Interrupted => "interrupted",
            ExitCode::Internal => "internal",
        }
    }

    /// The code for a services error category.
    pub fn from_category(c: ErrorCategory) -> Self {
        match c {
            ErrorCategory::InvalidConfig => ExitCode::Usage,
            ErrorCategory::InvalidData => ExitCode::Data,
            ErrorCategory::InvalidInitialization => ExitCode::Init,
            ErrorCategory::LogDensity
            | ErrorCategory::Gradient
            | ErrorCategory::Algorithm
            | ErrorCategory::Numerical => ExitCode::Inference,
            ErrorCategory::Output => ExitCode::Output,
            ErrorCategory::Cancelled => ExitCode::Interrupted,
            ErrorCategory::Resource | ErrorCategory::Internal => ExitCode::Internal,
        }
    }

    /// Every code with its meaning, for the documentation and `--help`.
    pub const TABLE: [(ExitCode, &'static str); 13] = [
        (ExitCode::Success, "success"),
        (
            ExitCode::Usage,
            "invalid command line options or configuration",
        ),
        (
            ExitCode::NotFound,
            "a required file is missing or unreadable",
        ),
        (ExitCode::Compile, "the Stan compiler rejected the program"),
        (
            ExitCode::Build,
            "the generated Rust project failed to build",
        ),
        (ExitCode::Data, "invalid data"),
        (ExitCode::Init, "invalid initialization"),
        (ExitCode::Inference, "the inference algorithm failed"),
        (ExitCode::Output, "output could not be written"),
        (ExitCode::Partial, "some chains failed"),
        (ExitCode::Diagnostics, "a diagnostic check failed"),
        (ExitCode::Interrupted, "interrupted"),
        (ExitCode::Internal, "internal error"),
    ];
}

/// An error that ends a command with a nonzero exit code.
#[derive(Debug)]
pub struct Failure {
    /// Exit code.
    pub code: ExitCode,
    /// Explanation, including the underlying cause.
    pub message: String,
}

impl Failure {
    /// A failure with `code`.
    pub fn new(code: ExitCode, message: impl Into<String>) -> Self {
        Failure {
            code,
            message: message.into(),
        }
    }

    /// Invalid options.
    pub fn usage(message: impl Into<String>) -> Self {
        Self::new(ExitCode::Usage, message)
    }

    /// A file that cannot be written or created (or a missing file); `what` says what it is for.
    pub fn io(what: &str, path: &std::path::Path, e: &std::io::Error) -> Self {
        let code = if e.kind() == std::io::ErrorKind::NotFound {
            ExitCode::NotFound
        } else {
            ExitCode::Output
        };
        Self::new(code, format!("{what} {}: {e}", path.display()))
    }

    /// An input file that cannot be read: missing (exit 3), or unreadable or malformed (exit 2).
    /// Never an output failure.
    pub fn input(what: &str, path: &std::path::Path, e: &std::io::Error) -> Self {
        let code = if e.kind() == std::io::ErrorKind::NotFound {
            ExitCode::NotFound
        } else {
            ExitCode::Usage
        };
        Self::new(code, format!("{what} {}: {e}", path.display()))
    }

    /// Writes the error to stderr, as text or as one JSON object.
    pub fn report(&self, json: bool) {
        if json {
            let v = serde_json::json!({
                "error": {
                    "exit_code": self.code.code(),
                    "category": self.code.name(),
                    "message": self.message,
                }
            });
            eprintln!("{v}");
        } else {
            eprintln!("error [{}]: {}", self.code.name(), self.message);
        }
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for Failure {}

impl From<ServiceError> for Failure {
    fn from(e: ServiceError) -> Self {
        // The services message already includes the model or inference error it wraps.
        Failure::new(ExitCode::from_category(e.category), e.message)
    }
}
