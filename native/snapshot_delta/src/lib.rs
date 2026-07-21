//! Native per-snapshot delta analysis for the StreamRAG pre-Send path.
//!
//! `analyze_delta` is a byte-for-byte reimplementation of
//! `stream/snapshot.py::SnapshotAnalyzer._analyze_python`. It runs on every
//! typed draft (~every 400 ms per user) and its output drives the StreamRAG
//! trigger's scheduling decision, so it is the one hot string-crunching path
//! where a native implementation earns its place. The Python fallback stays in
//! place; this module is optional and loaded through an import seam.

use blake2::digest::{Update, VariableOutput};
use blake2::Blake2sVar;
use pyo3::prelude::*;

/// Whitespace predicate matching CPython's `str.split()` / `str.isspace()`.
///
/// Rust's `char::is_whitespace()` follows the Unicode `White_Space` property,
/// which omits the four information separators U+001C..U+001F that CPython
/// treats as whitespace. Including them here keeps word counting identical to
/// the Python fallback across the entire input domain, not just ASCII text.
#[inline]
fn is_py_whitespace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Count whitespace-separated words the same way `len(text.split())` does:
/// runs of non-whitespace, ignoring leading/trailing/repeated whitespace.
fn word_count(text: &str) -> usize {
    let mut count = 0usize;
    let mut in_word = false;
    for c in text.chars() {
        if is_py_whitespace(c) {
            in_word = false;
        } else if !in_word {
            in_word = true;
            count += 1;
        }
    }
    count
}

/// Return the SnapshotDelta tuple:
/// `(fingerprint, common_prefix_chars, word_count, new_words, append_only)`.
///
/// Contract parity with the Python fallback:
/// - `common_prefix_chars`: shared leading Unicode scalar values (code points),
///   matching Python's `zip(previous, current)` iteration.
/// - `word_count` / `new_words`: `len(current.split())` and
///   `max(0, len(current.split()) - len(previous.split()))`.
/// - `append_only`: `common == len(previous)` counted in code points.
/// - `fingerprint`: `hashlib.blake2s(current.encode("utf-8"), digest_size=12).hexdigest()`.
#[pyfunction]
fn analyze_delta(previous: &str, current: &str) -> (String, usize, usize, usize, bool) {
    let mut common = 0usize;
    for (a, b) in previous.chars().zip(current.chars()) {
        if a != b {
            break;
        }
        common += 1;
    }

    let previous_words = word_count(previous);
    let current_words = word_count(current);
    let new_words = current_words.saturating_sub(previous_words);
    let append_only = common == previous.chars().count();

    let mut hasher = Blake2sVar::new(12).expect("12 is a valid blake2s digest length");
    hasher.update(current.as_bytes());
    let mut digest = [0u8; 12];
    hasher
        .finalize_variable(&mut digest)
        .expect("output buffer matches the configured digest length");
    let mut fingerprint = String::with_capacity(24);
    for byte in digest {
        use std::fmt::Write;
        write!(fingerprint, "{byte:02x}").expect("writing to a String cannot fail");
    }

    (fingerprint, common, current_words, new_words, append_only)
}

#[pymodule]
fn streamrag_snapshot(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_delta, m)?)?;
    Ok(())
}
