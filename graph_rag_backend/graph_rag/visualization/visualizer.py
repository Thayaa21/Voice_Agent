"""
Graph Visualizer — Step 18
============================
Renders the knowledge graph as an interactive HTML file using Pyvis.

TEACHING NOTES
--------------
Why Pyvis?
    Pyvis wraps the JavaScript library vis.js to create interactive
    network visualizations in HTML. You get:
    - Draggable nodes
    - Zoom in/out
    - Hover tooltips with node/edge details
    - Physics simulation (nodes repel each other, edges attract)
    All rendered as a single self-contained HTML file — no server needed.

Color coding:
    Different colors help users instantly understand node types:
    PERSON        = #4A90D9 (blue)
    DOCUMENT      = #27AE60 (green)
    ID_NUMBER     = #E67E22 (orange)
    DATE          = #9B59B6 (purple)
    ORGANIZATION  = #E74C3C (red)
    LOCATION      = #1ABC9C (teal)

Edge styles:
    mentions  = dashed grey (entity found in document)
    same_as   = solid, color from red→green based on confidence
                (red=barely matched, green=confident match)
    conflict  = dotted red (data disagreement)

Tooltips:
    Every node shows all its attributes when you hover over it.
    This lets users explore the data without reading the JSON file.

Fallback behavior:
    If Pyvis is not installed, we log an error and return empty string.
    The calling code (API, CLI) handles the empty string gracefully.
    We never crash — the rest of the pipeline still works without visualization.

Subgraph rendering:
    render_subgraph() creates a focused view centered on specific entities.
    Useful for exploring a single person's document network.
"""

import logging
from pathlib import Path
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)

# ---- Node colors by entity type ----
_NODE_COLORS: dict[str, str] = {
    "PERSON":       "#4A90D9",   # blue
    "document":     "#27AE60",   # green
    "ID_NUMBER":    "#E67E22",   # orange
    "DATE":         "#9B59B6",   # purple
    "ORGANIZATION": "#E74C3C",   # red
    "LOCATION":     "#1ABC9C",   # teal
    "default":      "#95A5A6",   # grey
}

# ---- Edge colors ----
_EDGE_MENTIONS_COLOR  = "#BDC3C7"   # light grey
_EDGE_CONFLICT_COLOR  = "#E74C3C"   # red
_EDGE_SAME_AS_LOW     = "#E74C3C"   # red (low confidence)
_EDGE_SAME_AS_HIGH    = "#27AE60"   # green (high confidence)


def _confidence_to_color(confidence: float) -> str:
    """
    Interpolate between red (low) and green (high) based on confidence.

    confidence = 0.0 → pure red #E74C3C
    confidence = 0.5 → orange #E67E22
    confidence = 1.0 → pure green #27AE60

    TEACHING: We decompose the hex colors to RGB and linearly interpolate.
    """
    confidence = max(0.0, min(1.0, confidence))

    # Red: (231, 76, 60) → Green: (39, 174, 96)
    r = int(231 + (39  - 231) * confidence)
    g = int(76  + (174 - 76)  * confidence)
    b = int(60  + (96  - 60)  * confidence)
    return f"#{r:02x}{g:02x}{b:02x}"


def _node_label(node_id: str, data: dict) -> str:
    """Short label for the node (shown on the node itself)."""
    if data.get("node_type") == "entity":
        name = data.get("name", "?")
        # Truncate long names
        return name[:20] + ("..." if len(name) > 20 else "")
    elif data.get("node_type") == "document":
        filename = data.get("filename", "?")
        return filename[:20] + ("..." if len(filename) > 20 else "")
    return node_id[:8]


def _node_tooltip(node_id: str, data: dict) -> str:
    """
    Full tooltip HTML shown on hover.
    Shows all node attributes except large text fields.
    """
    lines = [f"<b>ID:</b> {node_id[:12]}..."]

    for key, value in data.items():
        if key in ("text", "embedding", "paragraph_text"):
            continue  # Skip large fields
        if key == "attributes" and isinstance(value, dict):
            lines.append("<b>attributes:</b>")
            for k, v in value.items():
                lines.append(f"  &nbsp; {k}: {v}")
        elif isinstance(value, (str, int, float, bool)):
            lines.append(f"<b>{key}:</b> {value}")

    return "<br>".join(lines)


def _node_color(data: dict) -> str:
    """Determine node color from its type attributes."""
    if data.get("node_type") == "document":
        return _NODE_COLORS["document"]
    entity_type = data.get("entity_type", "default")
    return _NODE_COLORS.get(entity_type, _NODE_COLORS["default"])


def _node_size(data: dict) -> int:
    """Larger size for document nodes, smaller for entities."""
    if data.get("node_type") == "document":
        return 30
    return 20


class GraphVisualizer:
    """
    Renders a NetworkX graph as an interactive HTML file.

    Usage:
        visualizer = GraphVisualizer(graph)

        # Render the full graph
        path = visualizer.render("graph.html")

        # Render only a subgraph
        path = visualizer.render_subgraph(["entity_id_1", "entity_id_2"], "sub.html")
    """

    def __init__(self, graph: nx.Graph):
        """
        Args:
            graph — NetworkX graph from KnowledgeGraphBuilder
        """
        self._graph = graph

    def render(self, output_path: str = "graph.html") -> str:
        """
        Render the full graph as an interactive HTML file.

        Args:
            output_path — where to save the HTML file

        Returns:
            Path to the generated HTML file, or "" if rendering failed.
        """
        return self._render_graph(self._graph, output_path)

    def render_subgraph(
        self, entity_ids: list[str], output_path: str = "subgraph.html"
    ) -> str:
        """
        Render a subgraph centered on specific entity IDs.

        Includes all directly connected nodes (documents, same_as neighbors).

        Args:
            entity_ids  — entity node IDs to center the subgraph on
            output_path — where to save the HTML file

        Returns:
            Path to the generated HTML file, or "" if rendering failed.
        """
        # Build subgraph: include given nodes + their neighbors
        nodes_to_include: set[str] = set(entity_ids)
        for eid in entity_ids:
            if eid in self._graph:
                nodes_to_include.update(self._graph.neighbors(eid))

        subgraph = self._graph.subgraph(nodes_to_include)
        return self._render_graph(subgraph, output_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _render_graph(self, graph: nx.Graph, output_path: str) -> str:
        """
        Core rendering logic: converts NetworkX graph to Pyvis HTML.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            logger.error(
                "Install pyvis for graph visualization: pip install pyvis"
            )
            return ""

        if graph.number_of_nodes() == 0:
            logger.warning("Cannot render empty graph.")
            return ""

        output_path = str(Path(output_path).resolve())

        try:
            # ---- Create Pyvis network ----
            net = Network(
                height         = "800px",
                width          = "100%",
                bgcolor        = "#1a1a2e",   # dark background
                font_color     = "white",
                notebook       = False,
                cdn_resources  = "in_line",   # self-contained HTML
            )

            # Enable physics for organic layout
            net.force_atlas_2based(
                gravity        = -50,
                central_gravity = 0.01,
                spring_length  = 150,
                spring_strength = 0.05,
                damping        = 0.4,
            )

            # ---- Add nodes ----
            for node_id, data in graph.nodes(data=True):
                label   = _node_label(node_id, data)
                tooltip = _node_tooltip(node_id, data)
                color   = _node_color(data)
                size    = _node_size(data)

                net.add_node(
                    node_id,
                    label   = label,
                    title   = tooltip,
                    color   = color,
                    size    = size,
                    font    = {"color": "white", "size": 12},
                )

            # ---- Add edges ----
            for source, target, data in graph.edges(data=True):
                edge_type = data.get("edge_type", "")

                if edge_type == "mentions":
                    # Dashed grey line: entity → document
                    net.add_edge(
                        source, target,
                        title  = "mentions",
                        color  = _EDGE_MENTIONS_COLOR,
                        dashes = True,
                        width  = 1,
                    )

                elif edge_type == "same_as":
                    confidence = float(data.get("confidence", 0.5))
                    color      = _confidence_to_color(confidence)
                    label      = f"{confidence:.0%}"
                    tooltip    = (
                        f"same_as (confidence={confidence:.3f})<br>"
                        f"name_score={data.get('name_score', 0):.2f}<br>"
                        f"semantic_score={data.get('semantic_score', 0):.2f}<br>"
                        f"llm_confirmed={data.get('llm_confirmed', False)}"
                    )
                    net.add_edge(
                        source, target,
                        label  = label,
                        title  = tooltip,
                        color  = color,
                        width  = max(1, int(confidence * 4)),
                        dashes = False,
                    )

                elif edge_type == "conflict":
                    severity = data.get("severity", "minor")
                    attr_key = data.get("attribute_key", "")
                    tooltip  = (
                        f"CONFLICT: {data.get('conflict_type', 'mismatch')}<br>"
                        f"key={attr_key}<br>"
                        f"severity={severity}<br>"
                        f"value_a={data.get('value_a', '')}<br>"
                        f"value_b={data.get('value_b', '')}"
                    )
                    net.add_edge(
                        source, target,
                        label  = f"⚠ {attr_key}",
                        title  = tooltip,
                        color  = _EDGE_CONFLICT_COLOR,
                        dashes = True,
                        width  = 3,
                    )

                else:
                    # lives_with or unknown edge type
                    if edge_type == "lives_with":
                        address = data.get("address", "")
                        net.add_edge(
                            source, target,
                            title  = f"lives_with: {address}",
                            label  = "🏠",
                            color  = "#1ABC9C",
                            dashes = True,
                            width  = 2,
                        )
                    else:
                        net.add_edge(source, target, color="#555555", width=1)

            # ---- Write HTML ----
            net.write_html(output_path)

            logger.info(
                "Graph rendered: %s (%d nodes, %d edges)",
                output_path, graph.number_of_nodes(), graph.number_of_edges()
            )
            return output_path

        except Exception as e:
            logger.error("Visualization rendering failed: %s", e)
            return ""
