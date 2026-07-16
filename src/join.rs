//! Utilities for joining short lines into paragraphs

use crate::comments::find_comment_index;
use crate::format::{Pattern, State};
use crate::ignore::get_ignore;
use crate::logging::Log;
use crate::regexes::{RE_FORCED_BREAK, RE_SPLITTING, VERBS};
use crate::table::is_inside_table;
use crate::verbatim::get_verbatim;
use crate::LINE_END;
use std::path::Path;

/// File extensions for which lines are never joined
const NO_JOIN_EXTENSIONS: [&str; 3] = ["bib", "sty", "cls"];

/// Prefixes which prevent a line from taking part in joining.
/// Lines starting with a closing brace end a multi-line group,
/// so they are kept on their own line.
const BOUNDARY_STARTS: [&str; 4] = ["\\[", "\\]", "$$", "}"];

/// Commands with run-in headings which may begin a paragraph
const HEADING_STARTS: [&str; 2] = ["\\paragraph", "\\subparagraph"];

/// Display math environments in which lines are never joined
const MATH_ENVS: [&str; 8] = [
    "equation",
    "align",
    "gather",
    "multline",
    "eqnarray",
    "alignat",
    "flalign",
    "displaymath",
];

/// Check whether a line contains an equals sign outside inline math,
/// and track the inline math state across line breaks. Lines with a
/// bare equals sign are key-value assignments (e.g. \hypersetup
/// options) or displayed equations, which should not be reflowed.
fn scan_bare_equals(line: &str, in_math: &mut bool) -> bool {
    let mut bare = false;
    let mut prev_c = ' ';
    for c in line.chars() {
        match c {
            '$' if prev_c != '\\' => *in_math = !*in_math,
            '(' | ')' if prev_c == '\\' => *in_math = c == '(',
            '=' if !*in_math => bare = true,
            _ => {}
        }
        prev_c = c;
    }
    bare
}

/// Check whether a line may take part in joining at all
fn is_joinable(
    line: &str,
    pattern: &Pattern,
    protected: bool,
    bare_equals: bool,
) -> bool {
    let trimmed = line.trim();
    let contains_verb =
        pattern.contains_verb && VERBS.iter().any(|v| line.contains(v));
    let boundary = protected
        || trimmed.is_empty()
        || trimmed.starts_with('%')
        || RE_SPLITTING.is_match(line)
        || contains_verb
        || bare_equals
        || BOUNDARY_STARTS.iter().any(|p| trimmed.starts_with(p));
    !boundary
}

/// Check whether a line may absorb the following line
fn can_absorb(line: &str, pattern: &Pattern) -> bool {
    find_comment_index(line, pattern).is_none()
        && !RE_FORCED_BREAK.is_match(line)
        && !line.trim_end().ends_with('-')
}

/// Check whether a line may begin a chain of joined lines
fn can_begin_join(line: &str, pattern: &Pattern, after_boundary: bool) -> bool {
    // Only reflow paragraphs which start with text: lines starting with a
    // command or a brace (e.g. a preamble or macro code) are kept intact.
    // Run-in headings such as \paragraph{...} are the exception, as their
    // text continues on the same line.
    // A paragraph can only start after a boundary, so that continuations
    // of command lines (e.g. multi-line optional arguments) are not joined.
    let trimmed = line.trim_start();
    let starts_with_text = !trimmed.starts_with(['\\', '{'])
        || HEADING_STARTS.iter().any(|h| trimmed.starts_with(h));
    after_boundary && starts_with_text && can_absorb(line, pattern)
}

/// Track entering and leaving display math environments
fn update_math_depth(line: &str, pattern: &Pattern, depth: &mut i8) {
    if pattern.contains_env_begin
        && MATH_ENVS
            .iter()
            .any(|e| line.contains(&format!("\\begin{{{e}")))
    {
        *depth += 1;
    } else if pattern.contains_env_end
        && MATH_ENVS
            .iter()
            .any(|e| line.contains(&format!("\\end{{{e}")))
    {
        *depth = depth.saturating_sub(1);
    }
}

/// Join lines which belong to the same paragraph.
///
/// Short adjacent text lines are merged into a single long line, which
/// is then re-wrapped to the target line length by the wrapping pass.
pub fn join_lines(
    text: &str,
    file: &Path,
    logs: &mut Vec<Log>,
    verbatims_begin: &[String],
    verbatims_end: &[String],
) -> String {
    // Bibliographies, styles and classes contain no paragraph text.
    if file
        .extension()
        .is_some_and(|e| NO_JOIN_EXTENSIONS.iter().any(|n| e == *n))
    {
        return text.to_string();
    }

    let mut state = State::new();
    let mut new_text = String::with_capacity(text.len());
    let mut buffer: Option<String> = None;
    // The start of the file acts as a paragraph boundary.
    let mut after_boundary = true;
    let mut math_depth: i8 = 0;
    let mut in_display = false;
    let mut in_inline_math = false;

    for line in text.lines() {
        let pattern = Pattern::new(line);

        // Track ignored, verbatim and table regions.
        state.ignore = get_ignore(line, &state, logs, file, false);
        state.verbatim = get_verbatim(
            line,
            &state,
            logs,
            file,
            false,
            &pattern,
            verbatims_begin,
            verbatims_end,
        );
        state.table = is_inside_table(line, &state, &pattern);

        // Track display math regions, in which lines are never joined.
        update_math_depth(line, &pattern, &mut math_depth);
        let trimmed = line.trim();
        if trimmed.starts_with("\\[") && !trimmed.contains("\\]") {
            in_display = true;
        } else if trimmed.contains("\\]") {
            in_display = false;
        }

        let protected = state.ignore.visual
            || state.verbatim.visual
            || state.table.visual
            || math_depth > 0
            || in_display;

        // Scan for bare equals signs, ignoring comments and tracking
        // inline math spans across line breaks. Inline math cannot
        // cross paragraph boundaries, so the state is reset there.
        let bare_equals = if protected || trimmed.is_empty() {
            in_inline_math = false;
            false
        } else {
            let comment = find_comment_index(line, &pattern);
            let scanned = comment.map_or(line, |c| &line[..c]);
            scan_bare_equals(scanned, &mut in_inline_math)
        };

        let joinable = is_joinable(line, &pattern, protected, bare_equals);
        let begins = can_begin_join(line, &pattern, after_boundary);

        // Any non-joinable line is a paragraph boundary, and so is a
        // line ending with an opening brace such as \mycommand{.
        after_boundary = !joinable || line.trim_end().ends_with('{');

        // Absorb this line into the current paragraph if possible.
        if let Some(buf) = buffer.take() {
            if joinable {
                let joined = [buf.trim_end(), " ", line.trim_start()].concat();
                if can_absorb(line, &pattern) {
                    buffer = Some(joined);
                } else {
                    new_text.push_str(&joined);
                    new_text.push_str(LINE_END);
                }
                continue;
            }
            new_text.push_str(&buf);
            new_text.push_str(LINE_END);
        }

        // Otherwise check if this line starts a new paragraph.
        if joinable && begins {
            buffer = Some(line.to_string());
        } else {
            new_text.push_str(line);
            new_text.push_str(LINE_END);
        }
    }

    if let Some(buf) = buffer {
        new_text.push_str(&buf);
        new_text.push_str(LINE_END);
    }

    new_text
}
