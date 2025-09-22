"""Streamlit dashboard for exploring Ford-Fulkerson strategy outcomes."""

from __future__ import annotations

import os
import random
import sys
import types
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _install_matplotlib_stub() -> None:
    """Provide a minimal matplotlib stub when the dependency is missing."""

    if "matplotlib" in sys.modules:
        return

    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("pyplot")

    def _warn(*_args, **_kwargs) -> None:  # pragma: no cover - defensive stub
        raise ModuleNotFoundError(
            "Matplotlib is required for graph visualisation utilities. Install it "
            "to use matplotlib-based helpers."
        )

    for name in ("scatter", "arrow", "show", "xlabel", "ylabel", "title", "legend"):
        setattr(pyplot, name, _warn)

    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot


try:
    from ford_fulkerson.runner import (  # type: ignore
        DEFAULT_STRATEGIES,
        GraphRunResult,
        StrategyMetrics,
        run_strategies_for_graph,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - import fallback
    if exc.name != "matplotlib.pyplot":
        raise
    _install_matplotlib_stub()
    from ford_fulkerson.runner import (  # type: ignore
        DEFAULT_STRATEGIES,
        GraphRunResult,
        StrategyMetrics,
        run_strategies_for_graph,
    )


Vertex = Tuple[float, float]
REPO_ROOT = Path(__file__).resolve().parent
os.chdir(REPO_ROOT)
DATA_DIR = REPO_ROOT / "graph_data"

st.set_page_config(page_title="Ford-Fulkerson Strategy Explorer", layout="wide")


def _available_graph_numbers() -> Sequence[int]:
    numbers = {
        int(path.stem.split("_")[2])
        for path in DATA_DIR.glob("sim_val_*_vertices.csv")
        if path.stem.split("_")[2].isdigit()
    }
    return tuple(sorted(numbers))


@st.cache_data(show_spinner=False)
def load_graph_result(graph_no: int) -> tuple[GraphRunResult, pd.DataFrame]:
    """Execute all strategies once and tabulate their metrics."""

    random.seed(graph_no)
    result = run_strategies_for_graph(graph_no)
    metrics_rows: List[dict[str, float]] = []
    for strategy_name, metrics in result.strategies.items():
        metrics_rows.append(
            {
                "strategy": strategy_name,
                "max_flow": metrics.max_flow,
                "augmenting_paths": metrics.augmenting_paths,
                "mean_path_length": metrics.mean_length,
                "mpl": metrics.mpl,
            }
        )

    metrics_df = (
        pd.DataFrame(metrics_rows)
        .set_index("strategy")
        .sort_index()
    )
    return result, metrics_df


@st.cache_data(show_spinner=False)
def compute_strategy_edges(graph_no: int, strategy_key: str) -> pd.DataFrame:
    """Run ``strategy_key`` on a fresh residual network and capture edge flows."""

    graph_result, _ = load_graph_result(graph_no)
    graph = graph_result.graph
    residual_network = graph.create_residual_network().clone()

    random.seed(graph_no)
    strategy = DEFAULT_STRATEGIES[strategy_key]
    strategy(residual_network)

    records: List[dict[str, float]] = []
    for (tail, head), capacity in graph.capacities.items():
        residual = residual_network.get_capacity(tail, head)
        reverse_residual = residual_network.get_capacity(head, tail)
        flow = capacity - residual
        records.append(
            {
                "tail_x": tail[0],
                "tail_y": tail[1],
                "head_x": head[0],
                "head_y": head[1],
                "capacity": capacity,
                "flow": flow,
                "residual": residual,
                "reverse_residual": reverse_residual,
            }
        )

    return pd.DataFrame.from_records(records)


def _strategy_color_palette() -> Mapping[str, str]:
    base_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    strategies = list(DEFAULT_STRATEGIES.keys())
    palette: dict[str, str] = {}
    for index, strategy in enumerate(strategies):
        palette[strategy] = base_colors[index % len(base_colors)]
    return palette


def _build_metrics_charts(metrics_df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    metrics_reset = metrics_df.reset_index(names="strategy")
    palette = _strategy_color_palette()

    max_flow_fig = px.bar(
        metrics_reset,
        x="strategy",
        y="max_flow",
        color="strategy",
        color_discrete_map=palette,
        title="Maximum flow by strategy",
    )
    max_flow_fig.update_layout(showlegend=False, yaxis_title="Max flow")

    paths_fig = px.bar(
        metrics_reset,
        x="strategy",
        y="augmenting_paths",
        color="strategy",
        color_discrete_map=palette,
        title="Augmenting paths explored",
    )
    paths_fig.update_layout(showlegend=False, yaxis_title="Paths")

    return max_flow_fig, paths_fig


def _build_network_figure(
    graph_result: GraphRunResult, edges_df: pd.DataFrame, strategy_key: str
) -> go.Figure:
    graph = graph_result.graph
    source = graph.source
    sink = graph.sink

    residual_edges = edges_df[edges_df["flow"] <= 1e-9]
    flowing_edges = edges_df[edges_df["flow"] > 1e-9]

    residual_x: List[float] = []
    residual_y: List[float] = []
    residual_text: List[str | None] = []

    for row in residual_edges.itertuples(index=False):
        tail = (row.tail_x, row.tail_y)
        head = (row.head_x, row.head_y)
        hover = (
            f"{tail} → {head}<br>capacity: {row.capacity:.2f}<br>"
            f"flow: {row.flow:.2f}<br>residual: {row.residual:.2f}<br>"
            f"reverse residual: {row.reverse_residual:.2f}"
        )
        residual_x.extend([tail[0], head[0], None])
        residual_y.extend([tail[1], head[1], None])
        residual_text.extend([hover, hover, None])

    figure = go.Figure()
    if residual_x:
        figure.add_trace(
            go.Scatter(
                x=residual_x,
                y=residual_y,
                mode="lines",
                line=dict(color="rgba(180, 180, 180, 0.45)", width=1),
                hoverinfo="text",
                hovertext=residual_text,
                name="Residual",
                showlegend=False,
            )
        )

    palette = _strategy_color_palette()
    flow_color = palette.get(strategy_key, "#ff7f0e")
    for row in flowing_edges.itertuples(index=False):
        tail = (row.tail_x, row.tail_y)
        head = (row.head_x, row.head_y)
        capacity = row.capacity if row.capacity else 1.0
        ratio = max(min(row.flow / capacity, 1.0), 0.0)
        width = 2 + 6 * ratio
        hover = (
            f"{tail} → {head}<br>capacity: {row.capacity:.2f}<br>"
            f"flow: {row.flow:.2f}<br>residual: {row.residual:.2f}<br>"
            f"reverse residual: {row.reverse_residual:.2f}"
        )
        figure.add_trace(
            go.Scatter(
                x=[tail[0], head[0]],
                y=[tail[1], head[1]],
                mode="lines",
                line=dict(color=flow_color, width=width),
                hoverinfo="text",
                hovertext=[hover, hover],
                showlegend=False,
            )
        )

    vertices_x: List[float] = []
    vertices_y: List[float] = []
    node_colors: List[str] = []
    node_text: List[str] = []
    for vertex in graph.vertices:
        vertices_x.append(vertex[0])
        vertices_y.append(vertex[1])
        if vertex == source:
            node_colors.append("#2ca02c")
            node_text.append(f"Source {vertex}")
        elif vertex == sink:
            node_colors.append("#d62728")
            node_text.append(f"Sink {vertex}")
        else:
            node_colors.append("#1f77b4")
            node_text.append(f"Vertex {vertex}")

    figure.add_trace(
        go.Scatter(
            x=vertices_x,
            y=vertices_y,
            mode="markers",
            marker=dict(size=9, color=node_colors),
            hoverinfo="text",
            text=node_text,
            showlegend=False,
        )
    )

    figure.update_layout(
        title=f"Residual network for {strategy_key}",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    )
    figure.update_layout(plot_bgcolor="white")

    return figure


def _format_vertex(vertex: Vertex) -> str:
    return f"({vertex[0]:.3f}, {vertex[1]:.3f})"


def main() -> None:
    st.title("Ford-Fulkerson strategy explorer")
    st.caption(
        "Inspect benchmark graphs and compare augmenting-path strategies side by side."
    )

    graph_numbers = _available_graph_numbers()
    if not graph_numbers:
        st.error("No graph datasets found in graph_data/.")
        return

    strategy_options = list(DEFAULT_STRATEGIES.keys())

    with st.sidebar:
        st.header("Dataset")
        graph_no = st.selectbox("Simulation graph", graph_numbers, format_func=lambda x: f"Graph {x}")
        strategy_key = st.selectbox("Strategy", strategy_options, index=0)

    graph_result, metrics_df = load_graph_result(graph_no)
    graph = graph_result.graph

    with st.sidebar.expander("Graph details", expanded=True):
        st.write(
            {
                "nodes": graph.n,
                "edges": graph.total_edges,
                "radius": round(graph.r, 4),
                "upper capacity": graph.upper_cap,
                "max distance": graph.max_distance,
            }
        )
        st.write(
            {
                "source": _format_vertex(graph.source),
                "sink": _format_vertex(graph.sink),
            }
        )

    st.subheader("Strategy metrics")
    st.dataframe(metrics_df, use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)
    max_flow_fig, paths_fig = _build_metrics_charts(metrics_df)
    with chart_col1:
        st.plotly_chart(max_flow_fig, use_container_width=True)
    with chart_col2:
        st.plotly_chart(paths_fig, use_container_width=True)

    selected_metrics: StrategyMetrics = graph_result.strategies[strategy_key]
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Max flow", f"{selected_metrics.max_flow:.2f}")
    metric_col2.metric("Augmenting paths", int(selected_metrics.augmenting_paths))
    metric_col3.metric("Mean path length", f"{selected_metrics.mean_length:.2f}")
    metric_col4.metric("MPL", f"{selected_metrics.mpl:.2f}")

    edges_df = compute_strategy_edges(graph_no, strategy_key)
    network_fig = _build_network_figure(graph_result, edges_df, strategy_key)
    st.plotly_chart(network_fig, use_container_width=True)

    positive_flow_df = (
        edges_df[edges_df["flow"] > 1e-9]
        .copy()
        .sort_values("flow", ascending=False)
    )
    if not positive_flow_df.empty:
        st.subheader("Edges carrying flow")
        display_columns = [
            "tail_x",
            "tail_y",
            "head_x",
            "head_y",
            "capacity",
            "flow",
            "residual",
        ]
        st.dataframe(positive_flow_df[display_columns], use_container_width=True)
    else:
        st.info("The selected strategy did not route any flow across this network.")


if __name__ == "__main__":
    main()
