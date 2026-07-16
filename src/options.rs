//! Utilities for formatting multiline optional arguments

use crate::args::{Args, TabChar};
use crate::format::{Pattern, State};
use crate::ignore::get_ignore;
use crate::logging::Log;
use crate::table::is_inside_table;
use crate::verbatim::get_verbatim;
use crate::LINE_END;
use std::path::Path;

/// Return the byte ranges which must not be formatted.
fn protected_ranges(
    text: &str,
    file: &Path,
    logs: &mut Vec<Log>,
    verbatims_begin: &[String],
    verbatims_end: &[String],
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut state = State::new();
    let mut start = 0;

    for raw_line in text.split_inclusive('\n') {
        let line = raw_line.trim_end_matches(['\r', '\n']);
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

        let end = start + raw_line.len();
        if state.ignore.visual || state.verbatim.visual || state.table.visual {
            ranges.push((start, end));
        }
        start = end;
    }

    ranges
}

/// Check whether an opening bracket begins a command's optional argument.
fn is_optional_argument_open(text: &str, open: usize) -> bool {
    let line_start = text[..open].rfind('\n').map_or(0, |i| i + 1);
    let prefix = text[line_start..open].trim_start();
    if !prefix.starts_with('\\') || prefix.ends_with('\\') {
        return false;
    }

    // An optional argument begins at command level, rather than inside a
    // preceding brace or another optional argument.
    let mut brace_depth = 0_usize;
    let mut square_depth = 0_usize;
    let mut escaped = false;
    for character in prefix.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        match character {
            '\\' => escaped = true,
            '{' => brace_depth += 1,
            '}' => brace_depth = brace_depth.saturating_sub(1),
            '[' => square_depth += 1,
            ']' => square_depth = square_depth.saturating_sub(1),
            _ => {}
        }
    }
    brace_depth == 0 && square_depth == 0
}

/// Find the matching closing bracket, respecting nested braces and brackets.
fn find_close(text: &str, open: usize) -> Option<usize> {
    let mut square_depth = 1_usize;
    let mut escaped = false;
    let mut comment = false;

    for (offset, character) in text[open + 1..].char_indices() {
        if comment {
            if character == '\n' {
                comment = false;
            }
            continue;
        }
        if escaped {
            escaped = false;
            continue;
        }
        match character {
            '\\' => escaped = true,
            '%' => comment = true,
            '[' => square_depth += 1,
            ']' => {
                square_depth -= 1;
                if square_depth == 0 {
                    return Some(open + 1 + offset);
                }
            }
            _ => {}
        }
    }
    None
}

/// Split an optional argument at top-level commas.
///
/// Lists containing comments or items which themselves span lines are left
/// untouched because rewriting them could alter meaningful whitespace.
fn split_items(inner: &str) -> Option<(Vec<&str>, bool)> {
    let mut items = Vec::new();
    let mut start = 0;
    let mut brace_depth = 0_usize;
    let mut square_depth = 0_usize;
    let mut escaped = false;

    for (index, character) in inner.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match character {
            '\\' => escaped = true,
            '%' => return None,
            '{' => brace_depth += 1,
            '}' => brace_depth = brace_depth.saturating_sub(1),
            '[' => square_depth += 1,
            ']' => square_depth = square_depth.saturating_sub(1),
            ',' if brace_depth == 0 && square_depth == 0 => {
                items.push(inner[start..index].trim());
                start = index + character.len_utf8();
            }
            _ => {}
        }
    }

    // A single item is already canonical and should remain compact.
    if items.is_empty() {
        return None;
    }

    let last = inner[start..].trim();
    let trailing_comma = last.is_empty();
    if !trailing_comma {
        items.push(last);
    }

    if items.is_empty()
        || items
            .iter()
            .any(|item| item.is_empty() || item.contains(['\n', '\r']))
    {
        return None;
    }

    Some((items, trailing_comma))
}

/// Put each item in multiline optional arguments on its own line.
#[must_use]
pub fn format_options(
    text: &str,
    file: &Path,
    args: &Args,
    logs: &mut Vec<Log>,
    verbatims_begin: &[String],
    verbatims_end: &[String],
) -> String {
    let protected =
        protected_ranges(text, file, logs, verbatims_begin, verbatims_end);
    let indent_unit = match args.tabchar {
        TabChar::Tab => "\t".to_string(),
        TabChar::Space => " ".repeat(usize::from(args.tabsize)),
    };
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;

    while let Some(relative_open) = text[cursor..].find('[') {
        let open = cursor + relative_open;
        output.push_str(&text[cursor..open]);

        let Some(close) = find_close(text, open) else {
            output.push_str(&text[open..]);
            return output;
        };
        let overlaps_protected = protected
            .iter()
            .any(|&(start, end)| start <= close && end > open);
        let inner = &text[open + 1..close];
        let multiline = inner.contains('\n');

        if !is_optional_argument_open(text, open)
            || !multiline
            || overlaps_protected
        {
            output.push('[');
            cursor = open + 1;
            continue;
        }

        let Some((items, trailing_comma)) = split_items(inner) else {
            output.push('[');
            cursor = open + 1;
            continue;
        };

        let line_start = text[..open].rfind('\n').map_or(0, |i| i + 1);
        let base_indent: String = text[line_start..open]
            .chars()
            .take_while(|character| character.is_whitespace())
            .collect();
        let item_indent = format!("{base_indent}{indent_unit}");

        output.push('[');
        output.push_str(LINE_END);
        for (index, item) in items.iter().enumerate() {
            output.push_str(&item_indent);
            output.push_str(item);
            if index + 1 < items.len() || trailing_comma {
                output.push(',');
            }
            output.push_str(LINE_END);
        }
        output.push_str(&base_indent);
        output.push(']');
        cursor = close + 1;
    }

    output.push_str(&text[cursor..]);
    output
}
