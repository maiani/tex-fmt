//! Utilities for redistributing text across paragraph lines.
//!
//! This pass runs before indentation and line wrapping. It groups consecutive
//! source lines into prose paragraphs (see [`reflow_lines`]) and rewrites the
//! line breaks within each one according to the configured [`ReflowMode`]. The
//! three modes differ in how much they respect the authored breaks:
//!
//! - [`ReflowMode::Canonical`] ignores them, producing a consistent layout.
//! - [`ReflowMode::Minimal`] treats them as preferred break points and moves as
//!   few as possible, minimising the version-control diff.
//! - [`ReflowMode::Semantic`] keeps every authored break and only adds new ones.
//!
//! # Canonical
//!
//! `canonical_reflow` joins the paragraph onto a single line and lets the later
//! wrapping pass rebreak it from scratch. The input breaks carry no weight, so
//! the result depends only on `wraplen`/`wrapmin` and not on where the author
//! happened to break. There is no scoring: all the work is delegated to the
//! wrapping pass.
//!
//! # Minimal
//!
//! `minimally_reflow` is the non-trivial mode. It treats the authored breaks as
//! *anchors* and keeps an anchor unless moving it is forced by the width limits
//! — the aim is a small diff after a prose edit, not any interpretation of the
//! breaks. It is solved as a shortest-path problem:
//!
//! 1. The paragraph is joined into a single string, recording the byte offset of
//!    each authored break as an anchor (`join_paragraph`).
//! 2. Every position at which a line may legally break is enumerated
//!    (`legal_breaks`): after any wrap character, plus the anchors themselves.
//!    These are the nodes of a DAG whose edges are candidate lines.
//! 3. A dynamic program finds the path (choice of breaks) of minimum total
//!    `LayoutCost`, a tuple compared lexicographically. Its terms are ordered
//!    by priority so that each is only a tie-breaker for the ones above it:
//!    - `overflow` — the hard `wraplen` limit dominates everything.
//!    - `underflow` — then avoid lines shorter than the `wrapmin` target.
//!    - `changed_breaks` — then reuse as many authored breaks as possible; this
//!      is the term that actually keeps the diff small.
//!    - `displacement`, `raggedness`, `lines` — remaining ties are broken toward
//!      breaks near their original position, even line lengths, and compactness.
//!
//!    Placing `overflow`/`underflow` above `changed_breaks` means correctness of
//!    line width is never sacrificed to save a diff; placing `changed_breaks`
//!    above `raggedness` means a slightly uneven but stable layout is preferred
//!    to a prettier but noisier one. See `LayoutCost` and `transition_cost`
//!    for the exact per-line formulas.
//!
//! # Semantic
//!
//! `semantically_reflow` never joins or removes breaks. It keeps each authored
//! line and additionally splits it after every sentence boundary (a run of
//! `.`/`!`/`?` followed by space), so the result approximates one sentence per
//! line while leaving mid-sentence authored breaks in place. Sentence detection
//! is a heuristic: it skips terminators inside inline math and, for periods,
//! after digits, single-letter initials, and the abbreviations in
//! `NO_BREAK_ABBREVIATIONS`. A sentence that is still longer than `wraplen` is
//! then split at clause boundaries (`,`/`;`/`:`, outside math and any brace,
//! bracket, or parenthesis — see `clause_breaks`), packing whole clauses onto
//! each line up to the limit (`pack_clauses`); anything still too long is left
//! to the wrapping pass, which breaks it at word boundaries. No width scoring is
//! done, and length is measured on the trimmed text.
//!
//! # Parameters
//!
//! - `wraplen` is the hard maximum line length; `wrapmin` is a soft target, not
//!   a strict minimum. Both are measured including indentation width.
//! - `wrap_chars` defines where a line may break in minimal mode.
//! - `tabsize` converts leading tabs to a width for length accounting.
//! - Reflow requires wrapping to be enabled and never runs on `.bib`, `.sty`,
//!   or `.cls` files (`NO_REFLOW_EXTENSIONS`).
//!
//! All modes only ever touch prose. Comments, display and inline math, tables,
//! verbatim regions, and explicitly ignored regions are detected line by line
//! and act as paragraph boundaries, so their bytes pass through untouched.

use crate::args::{Args, ReflowMode};
use crate::comments::find_comment_index;
use crate::format::{Pattern, State};
use crate::ignore::get_ignore;
use crate::logging::Log;
use crate::regexes::{RE_FORCED_BREAK, RE_SPLITTING, VERBS};
use crate::table::is_inside_table;
use crate::verbatim::get_verbatim;
use crate::LINE_END;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::path::Path;

/// File extensions for which prose paragraphs are not reflowed.
const NO_REFLOW_EXTENSIONS: [&str; 3] = ["bib", "sty", "cls"];

/// Prefixes which prevent a line from taking part in reflowing.
const BOUNDARY_STARTS: [&str; 4] = ["\\[", "\\]", "$$", "}"];

/// Commands with run-in headings which may begin a paragraph.
const HEADING_STARTS: [&str; 2] = ["\\paragraph", "\\subparagraph"];

/// Multi-letter words after which a period does not end a sentence.
///
/// Single-letter tokens (initials such as `A.`, and the components of `e.g.`
/// and `i.e.`) are handled separately, so only abbreviations of two or more
/// letters need to be listed here. The comparison is case-insensitive.
const NO_BREAK_ABBREVIATIONS: [&str; 18] = [
    "etc", "cf", "vs", "al", "resp", "approx", "Fig", "Eq", "Sec", "Ref", "No",
    "vol", "pp", "Dr", "Mr", "Mrs", "Ms", "Prof",
];

/// Display math environments in which lines are never reflowed.
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

/// Lexicographic cost for a possible paragraph layout.
///
/// The fields are compared in declaration order (see the [`Ord`] impl), so each
/// term is a strict tie-breaker for the ones above it. This ordering encodes the
/// priorities of minimal reflow, from most to least important:
///
/// 1. `overflow` — total characters by which lines exceed `wraplen`. This is the
///    hard limit, so it dominates every other consideration.
/// 2. `underflow` — total characters by which non-final lines fall short of the
///    `wrapmin` target (with exceptions, see [`transition_cost`]). Avoids
///    leaving lines wastefully short.
/// 3. `changed_breaks` — number of authored breaks removed plus new breaks
///    introduced. This is what keeps the diff minimal: a layout that reuses the
///    input's breaks scores zero here.
/// 4. `displacement` — for each new break, how far it sits from the nearest
///    authored break. Among layouts that change the same number of breaks,
///    prefer the one whose new breaks land closest to where breaks already were.
/// 5. `raggedness` — total distance of non-final lines from `wrapmin`, breaking
///    ties toward more even line lengths.
/// 6. `lines` — number of lines, preferring the more compact layout last.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct LayoutCost {
    overflow: usize,
    underflow: usize,
    changed_breaks: usize,
    displacement: usize,
    raggedness: usize,
    lines: usize,
}

/// Shared paragraph data used to score candidate line breaks.
struct LayoutContext<'a> {
    text: &'a str,
    anchors: &'a [usize],
    anchor_set: &'a HashSet<usize>,
    args: &'a Args,
}

impl LayoutCost {
    fn add(self, other: Self) -> Self {
        Self {
            overflow: self.overflow.saturating_add(other.overflow),
            underflow: self.underflow.saturating_add(other.underflow),
            changed_breaks: self
                .changed_breaks
                .saturating_add(other.changed_breaks),
            displacement: self.displacement.saturating_add(other.displacement),
            raggedness: self.raggedness.saturating_add(other.raggedness),
            lines: self.lines.saturating_add(other.lines),
        }
    }
}

impl Ord for LayoutCost {
    fn cmp(&self, other: &Self) -> Ordering {
        (
            self.overflow,
            self.underflow,
            self.changed_breaks,
            self.displacement,
            self.raggedness,
            self.lines,
        )
            .cmp(&(
                other.overflow,
                other.underflow,
                other.changed_breaks,
                other.displacement,
                other.raggedness,
                other.lines,
            ))
    }
}

impl PartialOrd for LayoutCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Check whether a line contains an equals sign outside inline math.
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

/// Check whether a line may take part in paragraph reflowing.
fn is_reflowable(
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

/// Check whether a line may absorb the following line.
fn can_absorb(line: &str, pattern: &Pattern) -> bool {
    find_comment_index(line, pattern).is_none()
        && !RE_FORCED_BREAK.is_match(line)
        && !line.trim_end().ends_with('-')
}

/// Check whether a line may begin a prose paragraph.
fn can_begin_reflow(
    line: &str,
    pattern: &Pattern,
    after_boundary: bool,
) -> bool {
    let trimmed = line.trim_start();
    let starts_with_text = !trimmed.starts_with(['\\', '{'])
        || HEADING_STARTS.iter().any(|h| trimmed.starts_with(h));
    after_boundary && starts_with_text && can_absorb(line, pattern)
}

/// Track entering and leaving display math environments.
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

fn indent_width(line: &str, args: &Args) -> usize {
    line.chars()
        .take_while(char::is_ascii_whitespace)
        .map(|c| {
            if c == '\t' {
                usize::from(args.tabsize)
            } else {
                1
            }
        })
        .sum()
}

/// Join a paragraph and record the byte positions of its original breaks.
fn join_paragraph(lines: &[String]) -> (String, Vec<usize>) {
    let mut text = String::new();
    let mut anchors = Vec::with_capacity(lines.len().saturating_sub(1));
    for (index, line) in lines.iter().enumerate() {
        if index > 0 {
            text.push(' ');
            anchors.push(text.len());
        }
        text.push_str(line.trim());
    }
    (text, anchors)
}

/// Enumerate every byte position at which the joined paragraph may be broken.
///
/// Breaks are allowed after any wrap character (typically a space) that is not
/// escaped and not inside a trailing comment, plus the paragraph start and end
/// and all authored `anchors`. The returned positions are sorted and unique and
/// become the nodes of the shortest-path search in [`minimally_reflow`].
fn legal_breaks(text: &str, anchors: &[usize], args: &Args) -> Vec<usize> {
    let mut breaks = vec![0, text.len()];
    let mut previous = None;
    let pattern = Pattern::new(text);
    let comment = find_comment_index(text, &pattern);
    for (index, character) in text.char_indices() {
        if args.wrap_chars.contains(&character)
            && previous != Some('\\')
            && comment.is_none_or(|comment_index| index < comment_index)
        {
            let position = index + character.len_utf8();
            if position < text.len() {
                breaks.push(position);
            }
        }
        previous = Some(character);
    }
    breaks.extend(anchors.iter().copied());
    breaks.sort_unstable();
    breaks.dedup();
    breaks
}

fn nearest_anchor(position: usize, anchors: &[usize]) -> usize {
    anchors
        .iter()
        .map(|anchor| anchor.abs_diff(position))
        .min()
        .unwrap_or(0)
}

/// Cost of emitting a single line spanning `start..end` of the joined text.
///
/// This is the edge weight in the shortest-path formulation: a candidate line
/// runs from break `start` to break `end`, and `next_end` is the following
/// break (used to look one line ahead). The returned [`LayoutCost`] is summed
/// along a path and compared lexicographically, so the field values here are
/// tuned per line:
///
/// - `underflow`/`raggedness` are waived for the final line, which is allowed
///   to be short. `underflow` is also waived when the line already reaches
///   `wrapmin`, or when it *cannot* reach it without also pulling in the next
///   chunk and overshooting — this is what lets a legitimately short break at
///   the end of a paragraph survive.
/// - `changed_breaks` counts authored anchors that this line swallows plus one
///   if `end` is a newly introduced (non-final, non-anchor) break.
/// - `displacement` is only charged for a new break, measuring its distance to
///   the nearest authored anchor.
fn transition_cost(
    context: &LayoutContext,
    start: usize,
    end: usize,
    next_end: Option<usize>,
    indent: usize,
) -> LayoutCost {
    let length = context.text[start..end].trim().chars().count() + indent;
    let final_line = end == context.text.len();
    let reaches_target_with_next_chunk = next_end.is_some_and(|next| {
        context.text[start..next].trim().chars().count() + indent
            >= context.args.wrapmin
    });
    let removed_anchors = context
        .anchors
        .iter()
        .filter(|&&anchor| start < anchor && anchor < end)
        .count();
    let new_break =
        usize::from(!final_line && !context.anchor_set.contains(&end));
    let displacement = if new_break == 0 {
        0
    } else {
        nearest_anchor(end, context.anchors)
    };

    LayoutCost {
        overflow: length.saturating_sub(context.args.wraplen),
        underflow: if final_line
            || length >= context.args.wrapmin
            || reaches_target_with_next_chunk
        {
            0
        } else {
            context.args.wrapmin.saturating_sub(length)
        },
        changed_breaks: removed_anchors + new_break,
        displacement,
        raggedness: if final_line {
            0
        } else {
            context.args.wrapmin.abs_diff(length)
        },
        lines: 1,
    }
}

/// Reflow a paragraph while treating its existing breaks as preferred anchors.
///
/// The paragraph is joined into a single string whose legal break positions
/// form the nodes of a directed acyclic graph: an edge from break `i` to break
/// `j > i` represents laying out the text `breaks[i]..breaks[j]` as one line,
/// weighted by [`transition_cost`]. `costs[k]` holds the minimum total
/// [`LayoutCost`] of any layout ending at `breaks[k]`, and `previous[k]` records
/// the predecessor on that best path. Because the breaks are sorted, relaxing
/// them left to right computes the global optimum in one forward pass; the
/// chosen breaks are then recovered by walking `previous` back from the end.
fn minimally_reflow(lines: &[String], args: &Args) -> String {
    let (text, anchors) = join_paragraph(lines);
    if text.is_empty() {
        return String::new();
    }

    let breaks = legal_breaks(&text, &anchors, args);
    let anchor_set: HashSet<usize> = anchors.iter().copied().collect();
    let context = LayoutContext {
        text: &text,
        anchors: &anchors,
        anchor_set: &anchor_set,
        args,
    };
    let first_indent = lines.first().map_or(0, |line| indent_width(line, args));
    let continuation_indent = lines
        .get(1)
        .map_or(first_indent, |line| indent_width(line, args));
    let mut costs: Vec<Option<LayoutCost>> = vec![None; breaks.len()];
    let mut previous: Vec<Option<usize>> = vec![None; breaks.len()];
    costs[0] = Some(LayoutCost::default());

    for start_index in 0..breaks.len().saturating_sub(1) {
        let Some(base_cost) = costs[start_index] else {
            continue;
        };
        let indent = if start_index == 0 {
            first_indent
        } else {
            continuation_indent
        };
        // Candidate line lengths grow monotonically with `end_index`, so once a
        // within-limit break has been seen, the first overflowing one means all
        // later breaks overflow too and can be pruned.
        let mut saw_acceptable_break = false;
        for end_index in start_index + 1..breaks.len() {
            let segment_cost = transition_cost(
                &context,
                breaks[start_index],
                breaks[end_index],
                breaks.get(end_index + 1).copied(),
                indent,
            );
            if segment_cost.overflow == 0 {
                saw_acceptable_break = true;
            } else if saw_acceptable_break {
                break;
            }
            let candidate = base_cost.add(segment_cost);
            if costs[end_index].is_none_or(|cost| candidate < cost) {
                costs[end_index] = Some(candidate);
                previous[end_index] = Some(start_index);
            }
        }
    }

    let mut selected = vec![text.len()];
    let mut cursor = breaks.len() - 1;
    while let Some(prior) = previous[cursor] {
        if prior > 0 {
            selected.push(breaks[prior]);
        }
        cursor = prior;
    }
    selected.push(0);
    selected.sort_unstable();

    let mut output = String::with_capacity(text.len() + selected.len());
    for window in selected.windows(2) {
        output.push_str(text[window[0]..window[1]].trim());
        output.push_str(LINE_END);
    }
    output
}

/// Collapse a paragraph onto one line, discarding its authored breaks.
///
/// The later wrapping pass rebreaks the joined text, so no break scoring is
/// needed here.
fn canonical_reflow(lines: &[String]) -> String {
    let (text, _) = join_paragraph(lines);
    format!("{text}{LINE_END}")
}

/// Characters that terminate a sentence.
fn is_terminator(character: char) -> bool {
    matches!(character, '.' | '!' | '?')
}

/// Check whether a period at `chars[index]` is part of an abbreviation or
/// number rather than a genuine sentence end.
///
/// Only periods are ambiguous; `!` and `?` always terminate. A period is
/// treated as non-terminal when it directly follows a digit (decimals such as
/// `3.14` and enumerators such as `Section 3.`), a single letter (an initial or
/// a component of a dotted abbreviation like `e.g.`), or a listed abbreviation.
fn blocks_sentence_break(chars: &[(usize, char)], index: usize) -> bool {
    if chars[index].1 != '.' {
        return false;
    }
    let Some(previous) = index.checked_sub(1).map(|i| chars[i].1) else {
        return true;
    };
    if previous.is_ascii_digit() {
        return true;
    }
    if !previous.is_alphabetic() {
        return false;
    }
    let mut start = index;
    while start > 0 && chars[start - 1].1.is_alphabetic() {
        start -= 1;
    }
    let word: String = chars[start..index].iter().map(|&(_, c)| c).collect();
    word.chars().count() <= 1
        || NO_BREAK_ABBREVIATIONS
            .iter()
            .any(|abbreviation| abbreviation.eq_ignore_ascii_case(&word))
}

/// Find byte positions at which to break a line into sentences.
///
/// A break is placed immediately after a run of sentence terminators when it is
/// outside inline math, is followed by whitespace and further text, and is not
/// blocked by [`blocks_sentence_break`]. Breaks are never placed inside a
/// trailing comment.
fn sentence_breaks(line: &str, pattern: &Pattern) -> Vec<usize> {
    let limit = find_comment_index(line, pattern).unwrap_or(line.len());
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    let mut breaks = Vec::new();
    let mut in_math = false;
    let mut index = 0;
    while index < chars.len() {
        let (byte, character) = chars[index];
        if byte >= limit {
            break;
        }
        let escaped = index > 0 && chars[index - 1].1 == '\\';
        match character {
            '$' if !escaped => in_math = !in_math,
            '(' if escaped => in_math = true,
            ')' if escaped => in_math = false,
            _ if !in_math && is_terminator(character) => {
                let mut last = index;
                while last + 1 < chars.len() && is_terminator(chars[last + 1].1)
                {
                    last += 1;
                }
                let after = chars[last].0 + chars[last].1.len_utf8();
                let tail = line.get(after..limit).unwrap_or("");
                let ends_sentence = tail.starts_with(char::is_whitespace)
                    && !tail.trim().is_empty();
                if ends_sentence && !blocks_sentence_break(&chars, index) {
                    breaks.push(after);
                }
                index = last + 1;
                continue;
            }
            _ => {}
        }
        index += 1;
    }
    breaks
}

/// Characters that terminate a clause.
fn is_clause_boundary(character: char) -> bool {
    matches!(character, ',' | ';' | ':')
}

/// Find byte positions at which to break a line at clause boundaries.
///
/// A break is placed immediately after a comma, semicolon, or colon that is
/// followed by whitespace and further text. Breaks are only taken at the top
/// level: positions inside inline math, braces, brackets, or parentheses are
/// skipped so that punctuation within `\cite{a, b}`, `[a, b]`, math, or a
/// parenthetical is never split. Breaks are never placed inside a comment.
fn clause_breaks(line: &str, pattern: &Pattern) -> Vec<usize> {
    let limit = find_comment_index(line, pattern).unwrap_or(line.len());
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    let mut breaks = Vec::new();
    let mut in_math = false;
    let (mut brace, mut bracket, mut paren) = (0_i32, 0_i32, 0_i32);
    for index in 0..chars.len() {
        let (byte, character) = chars[index];
        if byte >= limit {
            break;
        }
        let escaped = index > 0 && chars[index - 1].1 == '\\';
        match character {
            '$' if !escaped => in_math = !in_math,
            '(' if escaped => in_math = true,
            ')' if escaped => in_math = false,
            '{' if !escaped => brace += 1,
            '}' if !escaped => brace = (brace - 1).max(0),
            '[' if !escaped => bracket += 1,
            ']' if !escaped => bracket = (bracket - 1).max(0),
            '(' if !escaped => paren += 1,
            ')' if !escaped => paren = (paren - 1).max(0),
            _ if !in_math
                && brace == 0
                && bracket == 0
                && paren == 0
                && is_clause_boundary(character) =>
            {
                let after = byte + character.len_utf8();
                let tail = line.get(after..limit).unwrap_or("");
                if tail.starts_with(char::is_whitespace)
                    && !tail.trim().is_empty()
                {
                    breaks.push(after);
                }
            }
            _ => {}
        }
    }
    breaks
}

/// Greedily choose clause breaks that keep each line within `wraplen`.
///
/// Given a too-long sentence segment `line[start..end]` and the clause
/// boundaries it contains, this packs as many whole clauses onto each line as
/// fit and returns the interior break positions. A single clause longer than
/// `wraplen` is emitted on its own line and left to the wrapping pass. Length is
/// measured on the trimmed text of each candidate line.
fn pack_clauses(
    line: &str,
    start: usize,
    end: usize,
    clauses: &[usize],
    wraplen: usize,
) -> Vec<usize> {
    let mut candidates: Vec<usize> = clauses
        .iter()
        .copied()
        .filter(|&c| start < c && c < end)
        .collect();
    candidates.push(end);

    let mut cuts = Vec::new();
    let mut line_start = start;
    let mut index = 0;
    while index < candidates.len() {
        // Extend the current line to the farthest clause boundary that fits.
        let mut fit = None;
        while index < candidates.len()
            && line[line_start..candidates[index]].trim().chars().count()
                <= wraplen
        {
            fit = Some(candidates[index]);
            index += 1;
        }
        // If not even the first clause fits, emit it anyway and move on.
        let position = fit.unwrap_or_else(|| {
            let position = candidates[index];
            index += 1;
            position
        });
        if position != end {
            cuts.push(position);
        }
        line_start = position;
    }
    cuts
}

/// Reflow a paragraph by keeping authored breaks and adding sentence breaks.
///
/// Each authored line is preserved as its own break and additionally split at
/// internal sentence boundaries, so this mode only ever adds breaks. A sentence
/// that would still exceed `wraplen` is additionally split at clause boundaries,
/// packing whole clauses onto each line (`pack_clauses`); any clause that
/// remains too long even alone is left to the wrapping pass, which breaks it at
/// word boundaries. Length is measured on the trimmed text, ignoring the
/// indentation added later, so the wrapping pass remains the final authority on
/// width.
fn semantically_reflow(lines: &[String], args: &Args) -> String {
    let mut output = String::new();
    for line in lines {
        let pattern = Pattern::new(line);
        let sentences = sentence_breaks(line, &pattern);
        let clauses = clause_breaks(line, &pattern);

        // Sentence cuts always apply; clause cuts apply only within a sentence
        // segment that is still longer than the wrap limit.
        let mut segments = Vec::with_capacity(sentences.len() + 2);
        segments.push(0);
        segments.extend(sentences);
        segments.push(line.len());
        let mut cuts = Vec::new();
        for window in segments.windows(2) {
            let (start, end) = (window[0], window[1]);
            cuts.push(start);
            if line[start..end].trim().chars().count() > args.wraplen {
                cuts.extend(pack_clauses(
                    line,
                    start,
                    end,
                    &clauses,
                    args.wraplen,
                ));
            }
        }
        cuts.push(line.len());
        cuts.sort_unstable();
        cuts.dedup();

        for window in cuts.windows(2) {
            let fragment = line[window[0]..window[1]].trim();
            if !fragment.is_empty() {
                output.push_str(fragment);
                output.push_str(LINE_END);
            }
        }
    }
    output
}

fn flush_paragraph(
    output: &mut String,
    paragraph: &mut Vec<String>,
    args: &Args,
) {
    if paragraph.is_empty() {
        return;
    }
    match args.reflow {
        ReflowMode::Off => {
            unreachable!("off mode does not enter the reflow pass")
        }
        ReflowMode::Minimal => {
            output.push_str(&minimally_reflow(paragraph, args));
        }
        ReflowMode::Canonical => output.push_str(&canonical_reflow(paragraph)),
        ReflowMode::Semantic => {
            output.push_str(&semantically_reflow(paragraph, args));
        }
    }
    paragraph.clear();
}

/// Reflow eligible prose paragraphs using the configured strategy.
pub fn reflow_lines(
    text: &str,
    file: &Path,
    args: &Args,
    logs: &mut Vec<Log>,
    verbatims_begin: &[String],
    verbatims_end: &[String],
) -> String {
    if file.extension().is_some_and(|extension| {
        NO_REFLOW_EXTENSIONS.iter().any(|item| extension == *item)
    }) {
        return text.to_string();
    }

    let mut state = State::new();
    let mut output = String::with_capacity(text.len());
    let mut paragraph = Vec::new();
    let mut after_boundary = true;
    let mut math_depth = 0_i8;
    let mut in_display = false;
    let mut in_inline_math = false;

    for line in text.lines() {
        let pattern = Pattern::new(line);
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
        let bare_equals = if protected || trimmed.is_empty() {
            in_inline_math = false;
            false
        } else {
            let comment = find_comment_index(line, &pattern);
            let scanned = comment.map_or(line, |index| &line[..index]);
            scan_bare_equals(scanned, &mut in_inline_math)
        };

        let reflowable = is_reflowable(line, &pattern, protected, bare_equals);
        let begins = can_begin_reflow(line, &pattern, after_boundary);
        after_boundary = !reflowable || line.trim_end().ends_with('{');

        if !paragraph.is_empty() {
            if reflowable {
                if args.reflow == ReflowMode::Minimal
                    && find_comment_index(line, &pattern).is_some()
                {
                    flush_paragraph(&mut output, &mut paragraph, args);
                    output.push_str(line);
                    output.push_str(LINE_END);
                    continue;
                }
                paragraph.push(line.to_string());
                if !can_absorb(line, &pattern) {
                    flush_paragraph(&mut output, &mut paragraph, args);
                }
                continue;
            }
            flush_paragraph(&mut output, &mut paragraph, args);
        }

        if reflowable && begins {
            paragraph.push(line.to_string());
        } else {
            output.push_str(line);
            output.push_str(LINE_END);
        }
    }

    flush_paragraph(&mut output, &mut paragraph, args);
    output
}
