from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tree_sitter import Language, Parser


@dataclass
class CodeChunk:
    """Represents a parsed, self-contained code unit."""

    name: str
    type: str  # "function" | "class" | "method"
    code: str
    docstring: Optional[str]
    file: str
    start_line: int
    end_line: int


class TreeSitterCodeParser:
    """Parse source files into function/class level chunks using Tree-sitter."""

    def __init__(self) -> None:
        self._parsers: Dict[str, Parser] = {}
        self._languages: Dict[str, Language] = {}
        self._init_languages()

    def _init_languages(self) -> None:
        """
        Lazily load Tree-sitter languages for Python, JavaScript, and TypeScript.

        This implementation prefers the `tree-sitter-language-pack` package and
        falls back to `tree_sitter_languages` if available. At least one of
        these must be installed to provide pre-built grammars.
        """
        try:
            # Preferred, actively maintained package.
            from tree_sitter_language_pack import get_language  # type: ignore
        except ImportError:
            try:
                # Fallback to legacy package if present.
                from tree_sitter_languages import get_language  # type: ignore
            except ImportError as exc:  # pragma: no cover - runtime safeguard
                raise RuntimeError(
                    "A Tree-sitter language bundle is required for TreeSitterCodeParser. "
                    "Install one of:\n"
                    "- pip install tree-sitter-language-pack\n"
                    "- pip install tree_sitter_languages"
                ) from exc

        language_map = {
            "python": "python",
            "javascript": "javascript",
            "typescript": "typescript",
        }

        for key, lang_name in language_map.items():
            language = get_language(lang_name)
            # Newer versions of `tree_sitter` take the language in the
            # constructor instead of exposing `set_language`.
            parser = Parser(language)
            self._languages[key] = language
            self._parsers[key] = parser

    @staticmethod
    def _detect_language(path: Path) -> Optional[str]:
        ext = path.suffix.lower()
        if ext == ".py":
            return "python"
        if ext in {".js", ".jsx"}:
            return "javascript"
        if ext in {".ts", ".tsx"}:
            return "typescript"
        return None

    def parse_file(self, path: Path) -> List[CodeChunk]:
        """
        Parse a single file into syntax-aware chunks.

        Each chunk is a complete function, method, or class definition with
        associated metadata.
        """
        language_key = self._detect_language(path)
        if language_key is None:
            raise ValueError(f"Unsupported file type for Tree-sitter parsing: {path}")

        parser = self._parsers.get(language_key)
        if parser is None:
            raise RuntimeError(f"No Tree-sitter parser initialised for language: {language_key}")

        source_text = path.read_text(encoding="utf-8")
        source_bytes = source_text.encode("utf-8")
        tree = parser.parse(source_bytes)
        root = tree.root_node

        chunks: List[CodeChunk] = []

        if language_key == "python":
            self._collect_python_chunks(root, source_bytes, str(path), chunks)
        else:
            self._collect_js_ts_chunks(root, source_bytes, str(path), chunks, language_key)

        return chunks

    def parse_files(self, paths: Iterable[Path]) -> List[CodeChunk]:
        """Parse multiple files and flatten all chunks."""
        all_chunks: List[CodeChunk] = []
        for p in paths:
            language_key = self._detect_language(p)
            if language_key is None:
                continue
            all_chunks.extend(self.parse_file(p))
        return all_chunks

    # -------------------------- Python helpers -------------------------- #

    def _collect_python_chunks(
        self,
        root,
        source_bytes: bytes,
        file_path: str,
        out: List[CodeChunk],
    ) -> None:
        """Collect function and class definitions from a Python syntax tree."""

        def walk(node, enclosing_class: Optional[str] = None) -> None:
            for child in node.children:
                if child.type == "function_definition":
                    name_node = _find_child_by_type(child, "identifier")
                    name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"
                    full_name = f"{enclosing_class}.{name}" if enclosing_class else name
                    code, start_line, end_line = _slice_source(child, source_bytes)
                    docstring = self._extract_python_docstring(child, source_bytes)

                    out.append(
                        CodeChunk(
                            name=full_name,
                            type="method" if enclosing_class else "function",
                            code=code,
                            docstring=docstring,
                            file=file_path,
                            start_line=start_line,
                            end_line=end_line,
                        )
                    )
                    walk(child, enclosing_class=enclosing_class)
                elif child.type == "class_definition":
                    name_node = _find_child_by_type(child, "identifier")
                    class_name = _node_text(name_node, source_bytes) if name_node else "<anonymous_class>"
                    code, start_line, end_line = _slice_source(child, source_bytes)
                    docstring = self._extract_python_docstring(child, source_bytes)

                    out.append(
                        CodeChunk(
                            name=class_name,
                            type="class",
                            code=code,
                            docstring=docstring,
                            file=file_path,
                            start_line=start_line,
                            end_line=end_line,
                        )
                    )
                    walk(child, enclosing_class=class_name)
                else:
                    walk(child, enclosing_class=enclosing_class)

        walk(root)

    def _extract_python_docstring(self, node, source_bytes: bytes) -> Optional[str]:
        """
        Extract a Python-style docstring from a function or class node.

        Looks for the first expression statement in the body that is a string
        literal and returns its text content without surrounding quotes.
        """
        # Find block / suite node which holds the body
        block = _find_child_by_type(node, "block") or _find_child_by_type(node, "suite")
        if block is None:
            return None

        for child in block.children:
            if child.type != "expression_statement":
                continue
            string_node = _find_child_by_type(child, "string")
            if string_node is None:
                continue
            raw = _node_text(string_node, source_bytes).strip()
            # Naive unquoting; good enough for doc generation context
            if (raw.startswith('"""') and raw.endswith('"""')) or (raw.startswith("'''") and raw.endswith("'''")):
                return raw[3:-3]
            if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
                return raw[1:-1]
        return None

    # --------------------- JavaScript / TypeScript helpers --------------------- #

    def _collect_js_ts_chunks(
        self,
        root,
        source_bytes: bytes,
        file_path: str,
        out: List[CodeChunk],
        language_key: str,
    ) -> None:
        """
        Collect function and class-like constructs for JS/TS.

        We consider:
        - function_declaration
        - method_definition
        - arrow_function (when assigned to a variable)
        - class_declaration
        """

        interesting_types = {
            "function_declaration",
            "method_definition",
            "class_declaration",
            "lexical_declaration",
            "variable_declaration",
        }

        def walk(node, enclosing_class: Optional[str] = None) -> None:
            for child in node.children:
                if child.type not in interesting_types:
                    walk(child, enclosing_class=enclosing_class)
                    continue

                if child.type == "class_declaration":
                    name_node = _find_child_by_type(child, "identifier")
                    class_name = _node_text(name_node, source_bytes) if name_node else "<anonymous_class>"
                    code, start_line, end_line = _slice_source(child, source_bytes)
                    jsdoc = self._extract_leading_jsdoc(child, source_bytes)
                    out.append(
                        CodeChunk(
                            name=class_name,
                            type="class",
                            code=code,
                            docstring=jsdoc,
                            file=file_path,
                            start_line=start_line,
                            end_line=end_line,
                        )
                    )
                    walk(child, enclosing_class=class_name)
                elif child.type in {"function_declaration", "method_definition"}:
                    name_node = _find_child_by_type(child, "identifier")
                    name = _node_text(name_node, source_bytes) if name_node else "<anonymous>"
                    full_name = f"{enclosing_class}.{name}" if enclosing_class else name
                    code, start_line, end_line = _slice_source(child, source_bytes)
                    jsdoc = self._extract_leading_jsdoc(child, source_bytes)
                    out.append(
                        CodeChunk(
                            name=full_name,
                            type="method" if enclosing_class else "function",
                            code=code,
                            docstring=jsdoc,
                            file=file_path,
                            start_line=start_line,
                            end_line=end_line,
                        )
                    )
                    walk(child, enclosing_class=enclosing_class)
                elif child.type in {"lexical_declaration", "variable_declaration"}:
                    # Look for arrow functions assigned to identifiers: const foo = () => {}
                    for decl in child.children:
                        if decl.type != "variable_declarator":
                            continue
                        identifier = _find_child_by_type(decl, "identifier")
                        arrow = _find_child_by_type(decl, "arrow_function")
                        if identifier is None or arrow is None:
                            continue
                        name = _node_text(identifier, source_bytes)
                        full_name = f"{enclosing_class}.{name}" if enclosing_class else name
                        code, start_line, end_line = _slice_source(decl, source_bytes)
                        jsdoc = self._extract_leading_jsdoc(decl, source_bytes)
                        out.append(
                            CodeChunk(
                                name=full_name,
                                type="function",
                                code=code,
                                docstring=jsdoc,
                                file=file_path,
                                start_line=start_line,
                                end_line=end_line,
                            )
                        )
                else:
                    walk(child, enclosing_class=enclosing_class)

        walk(root)

    def _extract_leading_jsdoc(self, node, source_bytes: bytes) -> Optional[str]:
        """
        Extract a leading JSDoc-style comment (`/** ... */`) immediately
        preceding the given node.
        """
        parent = node.parent
        if parent is None:
            return None

        siblings = parent.children
        try:
            idx = siblings.index(node)
        except ValueError:
            return None

        if idx == 0:
            return None

        prev_sibling = siblings[idx - 1]
        if prev_sibling.type != "comment":
            return None

        raw = _node_text(prev_sibling, source_bytes).strip()
        if raw.startswith("/**") and raw.endswith("*/"):
            inner = raw[3:-2].strip()
            return inner

        return None


def _node_text(node, source_bytes: bytes) -> str:
    """Decode the exact source text for a node."""
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")


def _slice_source(node, source_bytes: bytes) -> tuple[str, int, int]:
    """
    Return the source code and 1-based line range for a node.

    Tree-sitter points are 0-based; convert to 1-based to align with typical
    editor line numbers.
    """
    code = _node_text(node, source_bytes)
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    return code, start_line, end_line


def _find_child_by_type(node, node_type: str):
    """Return the first direct child of a given type, if any."""
    for child in node.children:
        if child.type == node_type:
            return child
    return None


__all__ = ["CodeChunk", "TreeSitterCodeParser"]

