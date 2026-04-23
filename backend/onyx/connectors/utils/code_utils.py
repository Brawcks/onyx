"""Syntax-aware splitting of source code files into logical sections.

For Python, uses the built-in `ast` module for accurate boundary detection.
For other languages, uses regex-based heuristics targeting top-level
function/class declarations.

Each section maps to a top-level construct (function, class, etc.) so
the downstream chunker produces semantically meaningful embeddings instead
of splitting arbitrarily in the middle of a function or class body.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from onyx.utils.logger import setup_logger

logger = setup_logger()


# Extensions considered text-based source code worth indexing.
# Files whose extension is NOT in this set are skipped during code indexing
# to avoid polluting the index with binary data (images, compiled artifacts, etc.).
TEXT_INDEXABLE_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Python
        ".py",
        ".pyi",
        # JavaScript / TypeScript
        ".js",
        ".mjs",
        ".cjs",
        ".jsx",
        ".ts",
        ".tsx",
        # JVM
        ".java",
        ".kt",
        ".kts",
        ".scala",
        ".groovy",
        # Systems
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".cxx",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".swift",
        # Scripting
        ".rb",
        ".php",
        ".lua",
        ".pl",
        # Shell
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        # Web / Styles
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".vue",
        ".svelte",
        # Config / Data
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".ini",
        ".cfg",
        ".env",
        # Docs / Markup
        ".md",
        ".rst",
        ".txt",
        # Query / Schema
        ".sql",
        ".graphql",
        ".gql",
        ".proto",
        # Infrastructure
        ".tf",
        ".hcl",
    }
)

# Maps file extensions to an internal language key used to select the right splitter.
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
}

# Files shorter than this are returned as a single section — no point splitting tiny files.
_MIN_LINES_TO_SPLIT = 20


@dataclass
class CodeSection:
    """A logical section of source code (top-level function, class, module block, etc.)."""

    identifier: str  # e.g. "MyClass", "my_function", "<module>"
    content: str
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed, inclusive


# ---------------------------------------------------------------------------
# Per-language regex patterns for top-level definitions.
# Patterns are anchored at column 0 (^) so nested defs are not matched.
# Each pattern may contain multiple capturing groups; _first_group() picks
# the first non-None one as the section identifier.
# ---------------------------------------------------------------------------
_PATTERNS: dict[str, re.Pattern[str]] = {
    # def foo(  /  async def foo(  /  class Foo(  /  class Foo:
    "python": re.compile(
        r"^(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)"
    ),
    # function foo(  /  async function foo(  /  export function foo(  /  class Foo
    "javascript": re.compile(
        r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*[(<]"
        r"|^(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)"
    ),
    "typescript": re.compile(
        r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*[(<]"
        r"|^(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)"
        r"|^(?:export\s+)?interface\s+([A-Za-z_$][A-Za-z0-9_$]*)"
    ),
    # public class Foo  /  interface Bar  /  enum Baz  /  record Qux
    "java": re.compile(
        r"^(?:(?:public|private|protected|static|abstract|final|sealed|strictfp)\s+)*"
        r"(?:class|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)"
    ),
    # func (r *Receiver) MethodName(  /  func FuncName(  /  type Foo struct
    "go": re.compile(
        r"^func\s+(?:\([^)]+\)\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[(\[]"
        r"|^type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:struct|interface)"
    ),
    "ruby": re.compile(
        r"^def\s+([A-Za-z_][A-Za-z0-9_!?]*)"
        r"|^class\s+([A-Za-z_][A-Za-z0-9_:]*)"
        r"|^module\s+([A-Za-z_][A-Za-z0-9_:]*)"
    ),
    # pub fn foo  /  pub struct Foo  /  impl Foo  /  pub trait Bar
    "rust": re.compile(
        r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)"
        r"|^(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)"
        r"|^(?:pub(?:\([^)]*\))?\s+)?impl(?:<[^>]*>)?\s+(?:[A-Za-z_][A-Za-z0-9_:<>, ]*\s+for\s+)?([A-Za-z_][A-Za-z0-9_]*)"
    ),
    "csharp": re.compile(
        r"^(?:(?:public|private|protected|internal|static|abstract|sealed|partial|virtual|override)\s+)*"
        r"(?:class|struct|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)"
    ),
    # class Foo  /  struct Bar  — free functions have varied signatures, skip them
    "cpp": re.compile(
        r"^(?:(?:template\s*<[^>]*>\s*)?(?:inline|static|virtual|explicit|constexpr)\s+)*"
        r"(?:class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)"
    ),
    "php": re.compile(
        r"^(?:(?:abstract|final|readonly)\s+)?(?:class|interface|trait|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"
        r"|^(?:(?:public|private|protected|static|abstract|final)\s+)*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
    ),
    "swift": re.compile(
        r"^(?:(?:public|private|internal|fileprivate|open|final|@\w+(?:\([^)]*\))?\s*)*)"
        r"(?:func|class|struct|enum|extension|protocol|actor)\s+([A-Za-z_][A-Za-z0-9_]*)"
    ),
    "kotlin": re.compile(
        r"^(?:(?:public|private|protected|internal|abstract|final|open|sealed|data|inner|inline|suspend|override)\s+)*"
        r"(?:fun|class|interface|object|enum\s+class)\s+([A-Za-z_][A-Za-z0-9_]*)"
    ),
    "scala": re.compile(
        r"^(?:(?:private|protected|override|abstract|sealed|final|implicit|lazy|case)\s+)*"
        r"(?:def|class|trait|object)\s+([A-Za-z_][A-Za-z0-9_]*)"
    ),
    # function_name() {  /  function function_name() {
    "shell": re.compile(
        r"^(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\)\s*(?:\{|$)"
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_language_for_file(file_path: str) -> str | None:
    """Return the language key for a file, or None if unsupported."""
    return EXTENSION_TO_LANGUAGE.get(Path(file_path).suffix.lower())


def split_file_into_code_sections(content: str, file_path: str) -> list[CodeSection]:
    """Split source code into logical top-level sections.

    Falls back to a single whole-file section when:
    - The file extension is not supported.
    - The file has fewer than _MIN_LINES_TO_SPLIT lines.
    - Parsing fails for any reason.
    """
    lines = content.splitlines()
    file_name = Path(file_path).name
    total = max(len(lines), 1)

    if len(lines) < _MIN_LINES_TO_SPLIT:
        return [
            CodeSection(
                identifier=file_name, content=content, start_line=1, end_line=total
            )
        ]

    language = get_language_for_file(file_path)
    if language is None:
        return [
            CodeSection(
                identifier=file_name, content=content, start_line=1, end_line=total
            )
        ]

    try:
        if language == "python":
            sections = _split_python(content, file_name)
        else:
            pattern = _PATTERNS.get(language)
            if pattern is None:
                return [
                    CodeSection(
                        identifier=file_name,
                        content=content,
                        start_line=1,
                        end_line=total,
                    )
                ]
            sections = _split_by_regex(content, pattern, file_name)
    except Exception:
        logger.warning(
            f"Code section splitting failed for {file_path!r}, using whole-file fallback",
            exc_info=True,
        )
        return [
            CodeSection(
                identifier=file_name, content=content, start_line=1, end_line=total
            )
        ]

    return sections or [
        CodeSection(identifier=file_name, content=content, start_line=1, end_line=total)
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _first_group(m: re.Match[str]) -> str:
    """Return the first non-None capturing group from a match."""
    for g in m.groups():
        if g is not None:
            return g
    return "<unknown>"


def _split_at_starts(
    lines: list[str],
    starts: list[tuple[int, str]],
    file_name: str,
) -> list[CodeSection]:
    """Slice *lines* into sections delimited by *starts* (1-indexed line, identifier).

    Content before the first start becomes a "<module>" prologue section.
    Each start extends to just before the next start (or end of file).
    """
    total = len(lines)
    sections: list[CodeSection] = []

    # Prologue: anything before the first top-level definition
    if starts[0][0] > 1:
        prologue = "\n".join(lines[: starts[0][0] - 1]).strip()
        if prologue:
            sections.append(
                CodeSection(
                    identifier="<module>",
                    content=prologue,
                    start_line=1,
                    end_line=starts[0][0] - 1,
                )
            )

    for i, (start_line, identifier) in enumerate(starts):
        end_line = starts[i + 1][0] - 1 if i + 1 < len(starts) else total
        body = "\n".join(lines[start_line - 1 : end_line]).rstrip()
        if body:
            sections.append(
                CodeSection(
                    identifier=identifier,
                    content=body,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

    return sections


def _split_by_regex(
    content: str,
    pattern: re.Pattern[str],
    file_name: str,
) -> list[CodeSection]:
    lines = content.splitlines()
    starts: list[tuple[int, str]] = [
        (i + 1, _first_group(m))
        for i, line in enumerate(lines)
        if (m := pattern.match(line))
    ]
    if not starts:
        return [
            CodeSection(
                identifier=file_name,
                content=content,
                start_line=1,
                end_line=max(len(lines), 1),
            )
        ]
    return _split_at_starts(lines, starts, file_name)


def _split_python(content: str, file_name: str) -> list[CodeSection]:
    """Python splitter using `ast` for accurate top-level boundary detection."""
    lines = content.splitlines()
    try:
        tree = ast.parse(content)
    except SyntaxError:
        # Fall back to regex for files with syntax errors
        return _split_by_regex(content, _PATTERNS["python"], file_name)

    starts: list[tuple[int, str]] = sorted(
        (node.lineno, node.name)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )

    if not starts:
        return [
            CodeSection(
                identifier=file_name,
                content=content,
                start_line=1,
                end_line=max(len(lines), 1),
            )
        ]

    return _split_at_starts(lines, starts, file_name)
