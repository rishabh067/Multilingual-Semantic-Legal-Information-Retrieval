"""Generate report-ready comparison markdown and plots from the model comparison CSV."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "model_comparison_100_query.csv"
SUMMARY_PATH = BASE_DIR / "comparison_summary.md"
P1_PLOT_PATH = BASE_DIR / "comparison_p_at_1.png"
NDCG3_PLOT_PATH = BASE_DIR / "comparison_ndcg_at_3.png"


def load_table() -> pd.DataFrame:
    """Load comparison table and coerce metrics to numeric where possible."""
    dataframe = pd.read_csv(CSV_PATH)
    metric_columns = ["P@1", "P@3", "P@5", "R@1", "R@3", "R@5", "NDCG@1", "NDCG@3", "NDCG@5"]
    for column in metric_columns:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    return dataframe


def save_bar_plot(dataframe: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    """Create a simple horizontal bar chart for one metric."""
    plot_df = dataframe.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(metric, ascending=True)
    plt.figure(figsize=(10, 5))
    plt.barh(plot_df["approach"], plot_df[metric], color="#2f6fed")
    plt.xlim(0, 1.05)
    plt.xlabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_summary(dataframe: pd.DataFrame) -> str:
    """Create a markdown summary for the report appendix."""
    available = dataframe.dropna(subset=["P@1"]).copy().sort_values("P@1", ascending=False)
    lines = [
        "# Model Comparison Summary",
        "",
        "This summary is generated from `model_comparison_100_query.csv`.",
        "",
        "## Available measured rows",
        "",
    ]

    if available.empty:
        lines.append("No measured rows are available yet.")
        return "\n".join(lines)

    for _, row in available.iterrows():
        lines.extend(
            [
                f"### {row['approach']}",
                "",
                f"- Source: `{row['source']}`",
                f"- P@1: `{row['P@1']:.4f}`",
                f"- P@3: `{row['P@3']:.4f}`" if pd.notna(row["P@3"]) else "- P@3: `N/A`",
                f"- P@5: `{row['P@5']:.4f}`" if pd.notna(row["P@5"]) else "- P@5: `N/A`",
                f"- NDCG@3: `{row['NDCG@3']:.4f}`" if pd.notna(row["NDCG@3"]) else "- NDCG@3: `N/A`",
                f"- Notes: {row['notes']}",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    """Generate report assets."""
    dataframe = load_table()
    SUMMARY_PATH.write_text(build_summary(dataframe), encoding="utf-8")
    save_bar_plot(dataframe, "P@1", P1_PLOT_PATH, "Comparison of P@1 Across Retrieval Approaches")
    save_bar_plot(dataframe, "NDCG@3", NDCG3_PLOT_PATH, "Comparison of NDCG@3 Across Retrieval Approaches")
    print(f"Saved summary to {SUMMARY_PATH}")
    print(f"Saved plot to {P1_PLOT_PATH}")
    print(f"Saved plot to {NDCG3_PLOT_PATH}")


if __name__ == "__main__":
    main()
