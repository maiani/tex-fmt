//! Utilities for redistributing text across paragraph lines.

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

fn canonical_reflow(lines: &[String]) -> String {
    let (text, _) = join_paragraph(lines);
    format!("{text}{LINE_END}")
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
