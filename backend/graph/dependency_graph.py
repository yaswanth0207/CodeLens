from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
from pyvis.network import Network

from backend.parser.tree_sitter_parser import TreeSitterCodeParser


@dataclass
class CallEdge:
    """Represents a function/method call between two symbols."""

    caller_file: str
    caller_symbol: str
    callee_name: str


@dataclass
class ImportEdge:
    """Represents an import relationship between two files/modules."""

    src_file: str
    dst_module: str


class DependencyGraph:
    """
    Build and query a directed graph of imports and function calls.

    Nodes represent files and symbols; edges represent imports and calls.
    The graph is primarily used to visualise dependencies via pyvis and to
    answer "who calls X?" style questions.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.graph = nx.DiGraph()
        self._parser = TreeSitterCodeParser()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build(self, files: Optional[Iterable[Path]] = None) -> None:
        """
        Build or rebuild the dependency graph for the given files.

        If `files` is None, all supported files under `root` are scanned.
        """
        self.graph.clear()
        file_paths = list(files) if files is not None else list(self._iter_supported_files(self.root))

        # First add file nodes.
        for path in file_paths:
            file_str = str(path)
            self.graph.add_node(file_str, type="file", label=Path(file_str).name)

        # Then extract imports and calls.
        for path in file_paths:
            if path.suffix == ".py":
                self._process_python_file(path)
            else:
                self._process_js_like_file(path)

    def get_graph(self, file_path: Path) -> str:
        """
        Return an HTML representation of the dependency subgraph for a file.

        The HTML is suitable for embedding in a web view (e.g. Streamlit).
        """
        file_str = str(file_path)
        if file_str not in self.graph:
            # If file not present, return an empty graph HTML.
            net = Network(height="600px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
            return net.generate_html()

        # Collect the ego network for the file: its immediate neighbours.
        neighbours: Set[str] = set(self.graph.predecessors(file_str)) | set(self.graph.successors(file_str))
        sub_nodes = {file_str, *neighbours}

        subgraph = self.graph.subgraph(sub_nodes).copy()

        net = Network(height="600px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")

        for node_id, data in subgraph.nodes(data=True):
            node_type = data.get("type", "file")
            label = data.get("label", Path(node_id).name if node_type == "file" else node_id)
            color = "#1976d2" if node_type == "file" else "#9c27b0"
            net.add_node(node_id, label=label, color=color, title=node_id)

        for src, dst, data in subgraph.edges(data=True):
            edge_type = data.get("type", "dependency")
            color = "#4caf50" if edge_type == "import" else "#ff9800"
            title = data.get("label", edge_type)
            net.add_edge(src, dst, color=color, title=title)

        return net.generate_html()

    def get_callers(self, function_name: str) -> List[Tuple[str, str]]:
        """
        Return a list of (caller_file, caller_symbol) for the given function name.

        We perform a simple suffix match on symbol names so that queries like
        'do_something' will match 'MyClass.do_something'.
        """
        callers: List[Tuple[str, str]] = []
        for src, dst, data in self.graph.edges(data=True):
            if data.get("type") != "call":
                continue
            callee = data.get("callee", "")
            if callee.endswith(function_name):
                caller_symbol = data.get("caller_symbol", "")
                callers.append((src, caller_symbol))
        return callers

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _iter_supported_files(self, root: Path) -> Iterable[Path]:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() in {".py", ".js", ".ts", ".jsx", ".tsx"}:
                yield path

    # ----------------------------- Python ----------------------------- #

    def _process_python_file(self, path: Path) -> None:
        """Extract import and call relationships from a Python file."""
        file_str = str(path)

        try:
            source = path.read_text(encoding="utf-8")
            module = ast.parse(source, filename=file_str)
        except Exception:
            return

        # Imports.
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name
                    self._add_import_edge(file_str, target)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self._add_import_edge(file_str, node.module)

        # Calls: we track the current function/class context to build symbol names.
        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self, outer: "DependencyGraph", filename: str) -> None:
                self.outer = outer
                self.filename = filename
                self.scope: List[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

            def visit_Call(self, node: ast.Call) -> None:
                callee_name = self._extract_callee_name(node.func)
                if callee_name:
                    caller_symbol = ".".join(self.scope) if self.scope else "<module>"
                    self.outer._add_call_edge(self.filename, caller_symbol, callee_name)
                self.generic_visit(node)

            @staticmethod
            def _extract_callee_name(expr: ast.expr) -> Optional[str]:
                if isinstance(expr, ast.Name):
                    return expr.id
                if isinstance(expr, ast.Attribute):
                    return expr.attr
                return None

        FunctionVisitor(self, file_str).visit(module)

    # ------------------------- JavaScript / TS ------------------------ #

    def _process_js_like_file(self, path: Path) -> None:
        """
        Extract import and call relationships for JS/TS/JSX/TSX files.

        For simplicity and portability we use lightweight regular expressions
        over the raw source rather than a full Tree-sitter query language.
        """
        file_str = str(path)

        try:
            source = path.read_text(encoding="utf-8")
        except Exception:
            return

        # ES module & CommonJS imports.
        import_patterns = [
            r"""import\s+.+?\s+from\s+['"](?P<mod>[^'"]+)['"]""",
            r"""import\s+['"](?P<mod>[^'"]+)['"]""",
            r"""require\(\s*['"](?P<mod>[^'"]+)['"]\s*\)""",
        ]
        for pat in import_patterns:
            for m in re.finditer(pat, source):
                module_name = m.group("mod")
                self._add_import_edge(file_str, module_name)

        # Function calls: best-effort heuristic (identifier or dotted name followed by '(').
        call_pattern = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_\.]*)\s*\(")
        for m in call_pattern.finditer(source):
            callee = m.group("name").split(".")[-1]
            if not callee:
                continue
            # We do not track fine-grained JS/TS scopes here; treat as module-level calls.
            caller_symbol = "<module>"
            self._add_call_edge(file_str, caller_symbol, callee)

    # ------------------------------ Edges ----------------------------- #

    def _add_import_edge(self, src_file: str, dst_module: str) -> None:
        """
        Add an import edge from `src_file` to a module name.

        Module names are resolved to file-like node IDs where possible but we
        keep the raw module as an attribute.
        """
        dst_id = self._resolve_module_to_file(dst_module)
        if dst_id not in self.graph:
            self.graph.add_node(dst_id, type="file", label=Path(dst_id).name)

        self.graph.add_edge(
            src_file,
            dst_id,
            type="import",
            label=f"import {dst_module}",
            module=dst_module,
        )

    def _add_call_edge(self, caller_file: str, caller_symbol: str, callee_name: str) -> None:
        """
        Add a call edge between a caller symbol within a file and a callee name.
        """
        caller_node = f"{caller_file}::{caller_symbol}"
        if caller_node not in self.graph:
            self.graph.add_node(caller_node, type="symbol", label=caller_symbol)

        # Use a generic callee node; resolution to specific files is non-trivial
        # across languages, so we keep the callee as a symbol node.
        callee_node = callee_name
        if callee_node not in self.graph:
            self.graph.add_node(callee_node, type="symbol", label=callee_name)

        # Connect file to its symbol for visual clarity.
        if not self.graph.has_edge(caller_file, caller_node):
            self.graph.add_edge(caller_file, caller_node, type="contains", label="contains")

        self.graph.add_edge(
            caller_node,
            callee_node,
            type="call",
            label=f"calls {callee_name}",
            caller_symbol=caller_symbol,
            callee=callee_name,
        )

    def _resolve_module_to_file(self, module: str) -> str:
        """
        Best-effort resolution of a Python/JS-style module name to a file path.

        If resolution fails, we fall back to treating the module string itself
        as the node identifier.
        """
        # Normalise common path-like modules (e.g. './foo/bar')
        if module.startswith("."):
            candidate = (self.root / module).with_suffix(".py")
            return str(candidate)

        # For dotted modules, assume Python-style and map to path.
        if "." in module:
            candidate = self.root / (module.replace(".", "/") + ".py")
            return str(candidate)

        # Fallback: return the raw module string.
        return module


__all__ = ["DependencyGraph"]

