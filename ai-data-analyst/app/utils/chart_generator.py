"""
chart_generator.py
──────────────────
Standalone chart-generation functions.
Every function returns a base64-encoded PNG string suitable for JSON responses.
"""

import io
import base64
import logging
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Design tokens ─────────────────────────────────────────────────────────────
_PRIMARY   = "#2E5090"
_SECONDARY = "#5B9BD5"
_ACCENT    = "#ED7D31"
_PALETTE   = [_PRIMARY, _SECONDARY, _ACCENT,
              "#70AD47", "#FFC000", "#A5A5A5", "#FF0000", "#843C0C"]
_FIGSIZE   = (10, 6)
_DPI       = 100
_TITLE_STYLE = {"fontsize": 14, "fontweight": "bold", "pad": 10}
_LABEL_SIZE  = 12

# Apply style once at module load — NOT inside request handlers (global state is not thread-safe)
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": _LABEL_SIZE,
})


# ── Helper ────────────────────────────────────────────────────────────────────

def fig_to_base64(fig: plt.Figure) -> str:
    """Serialise a Matplotlib figure to a base64-encoded PNG string then close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=_DPI)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _new_fig(figsize: tuple = _FIGSIZE) -> tuple[plt.Figure, plt.Axes]:
    # Use Figure() directly instead of plt.subplots() to avoid touching shared pyplot state
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ── 2. Histogram ──────────────────────────────────────────────────────────────

def create_histogram(
    data: pd.Series,
    title: str,
    color: str = _PRIMARY,
) -> str:
    """Histogram with optional KDE overlay."""
    logger.debug("create_histogram — title=%r, n=%d", title, len(data))
    try:
        series = data.dropna()
        fig, ax = _new_fig()

        ax.hist(series, bins="auto", color=color, edgecolor="white",
                alpha=0.85, density=True, label="Frequency")

        try:
            series.plot.kde(ax=ax, color=_ACCENT, linewidth=2, label="KDE")
            ax.legend(fontsize=10)
        except Exception:
            pass

        ax.set_title(title, **_TITLE_STYLE)
        ax.set_xlabel(data.name or "", fontsize=_LABEL_SIZE)
        ax.set_ylabel("Density", fontsize=_LABEL_SIZE)
        plt.tight_layout()
        return fig_to_base64(fig)

    except Exception as exc:
        logger.error("create_histogram failed: %s", exc)
        return _error_chart(f"Histogram error: {exc}")


# ── 3. Bar chart ──────────────────────────────────────────────────────────────

def create_bar_chart(
    categories: list,
    values: list,
    title: str,
    horizontal: bool = False,
    color: str = _PRIMARY,
) -> str:
    """Vertical or horizontal bar chart with value labels."""
    logger.debug("create_bar_chart — title=%r, n=%d", title, len(categories))
    try:
        n = len(categories)
        fig_h = max(5, n * 0.35 + 1) if horizontal else 6
        fig_w = 10 if horizontal else max(8, n * 0.55 + 2)
        fig, ax = _new_fig((fig_w, fig_h))

        cat_str = [str(c) for c in categories]
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(n)]

        if horizontal:
            bars = ax.barh(cat_str, values, color=colors, edgecolor="white")
            ax.bar_label(bars, fmt="%g", padding=4, fontsize=10)
            ax.set_xlabel("Value", fontsize=_LABEL_SIZE)
            ax.invert_yaxis()
        else:
            bars = ax.bar(cat_str, values, color=colors, edgecolor="white")
            ax.bar_label(bars, fmt="%g", padding=3, fontsize=10)
            ax.set_ylabel("Value", fontsize=_LABEL_SIZE)
            plt.xticks(rotation=45, ha="right", fontsize=10)

        ax.set_title(title, **_TITLE_STYLE)
        plt.tight_layout()
        return fig_to_base64(fig)

    except Exception as exc:
        logger.error("create_bar_chart failed: %s", exc)
        return _error_chart(f"Bar chart error: {exc}")


# ── 4. Scatter plot ───────────────────────────────────────────────────────────

def create_scatter(
    x: pd.Series,
    y: pd.Series,
    title: str,
    color_by: pd.Series = None,
    xlabel: str = "",
    ylabel: str = "",
) -> str:
    """Scatter plot with optional color grouping."""
    logger.debug("create_scatter — title=%r", title)
    try:
        fig, ax = _new_fig()

        if color_by is not None:
            unique_vals = color_by.unique()
            for i, val in enumerate(unique_vals):
                mask = color_by == val
                ax.scatter(
                    x[mask], y[mask],
                    c=_PALETTE[i % len(_PALETTE)],
                    s=25, alpha=0.7, label=str(val),
                )
            ax.legend(fontsize=9, title=color_by.name or "", title_fontsize=9)
        else:
            ax.scatter(x, y, c=_PRIMARY, s=25, alpha=0.6)

        # Regression line
        try:
            valid = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(valid) >= 4:
                m, b = np.polyfit(valid["x"], valid["y"], 1)
                x_line = np.linspace(valid["x"].min(), valid["x"].max(), 200)
                ax.plot(x_line, m * x_line + b, color=_ACCENT,
                        linewidth=1.5, linestyle="--", label="Trend")
                if color_by is None:
                    ax.legend(fontsize=10)
        except Exception:
            pass

        ax.set_title(title, **_TITLE_STYLE)
        ax.set_xlabel(xlabel or (x.name or ""), fontsize=_LABEL_SIZE)
        ax.set_ylabel(ylabel or (y.name or ""), fontsize=_LABEL_SIZE)
        plt.tight_layout()
        return fig_to_base64(fig)

    except Exception as exc:
        logger.error("create_scatter failed: %s", exc)
        return _error_chart(f"Scatter error: {exc}")


# ── 5. Heatmap ────────────────────────────────────────────────────────────────

def create_heatmap(data: pd.DataFrame, title: str) -> str:
    """Annotated correlation-style heatmap (lower-triangle mask)."""
    logger.debug("create_heatmap — title=%r, shape=%s", title, data.shape)
    try:
        n = max(data.shape)
        size = max(7, min(n, 18))
        fig, ax = plt.subplots(figsize=(size, size * 0.8))
        plt.style.use("seaborn-v0_8-whitegrid")

        mask = np.triu(np.ones_like(data, dtype=bool)) if data.shape[0] == data.shape[1] else None

        sns.heatmap(
            data,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            ax=ax,
            linewidths=0.5,
            annot_kws={"size": max(7, 11 - n // 4)},
            vmin=-1 if data.values.min() >= -1 else None,
            vmax=1  if data.values.max() <= 1  else None,
        )
        ax.set_title(title, **_TITLE_STYLE)
        plt.tight_layout()
        return fig_to_base64(fig)

    except Exception as exc:
        logger.error("create_heatmap failed: %s", exc)
        return _error_chart(f"Heatmap error: {exc}")


# ── 6. Line chart ─────────────────────────────────────────────────────────────

def create_line_chart(
    x: pd.Series,
    y: pd.Series,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
) -> str:
    """Simple line chart with marker dots."""
    logger.debug("create_line_chart — title=%r, n=%d", title, len(x))
    try:
        fig, ax = _new_fig((12, 6))

        n = len(x)
        marker = "o" if n <= 60 else None
        ms = 4 if n <= 60 else None

        ax.plot(x, y, color=_PRIMARY, linewidth=2,
                marker=marker, markersize=ms, alpha=0.9)
        ax.fill_between(x, y, alpha=0.08, color=_PRIMARY)

        ax.set_title(title, **_TITLE_STYLE)
        ax.set_xlabel(xlabel or (x.name or ""), fontsize=_LABEL_SIZE)
        ax.set_ylabel(ylabel or (y.name or ""), fontsize=_LABEL_SIZE)
        plt.xticks(rotation=30, ha="right", fontsize=10)
        plt.tight_layout()
        return fig_to_base64(fig)

    except Exception as exc:
        logger.error("create_line_chart failed: %s", exc)
        return _error_chart(f"Line chart error: {exc}")


# ── 7. Box plot ───────────────────────────────────────────────────────────────

def create_box_plot(
    data: pd.DataFrame,
    column: str,
    group_by: str = None,
    title: str = "",
) -> str:
    """Box plot — single column or grouped by a categorical column."""
    logger.debug("create_box_plot — column=%r, group_by=%r", column, group_by)
    try:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found.")

        fig, ax = _new_fig()
        t = title or f"Box Plot — {column}"

        if group_by and group_by in data.columns:
            top_cats = data[group_by].value_counts().head(10).index.tolist()
            groups = [
                data.loc[data[group_by] == cat, column].dropna().values
                for cat in top_cats
            ]
            bp = ax.boxplot(
                groups, labels=top_cats, patch_artist=True, vert=True
            )
            for patch, c in zip(bp["boxes"], _PALETTE):
                patch.set_facecolor(c)
                patch.set_alpha(0.8)
            for median in bp["medians"]:
                median.set_color(_ACCENT)
                median.set_linewidth(2)
            plt.xticks(rotation=40, ha="right", fontsize=10)
            ax.set_xlabel(group_by, fontsize=_LABEL_SIZE)
        else:
            series = data[column].dropna()
            bp = ax.boxplot(
                series, vert=False, patch_artist=True,
                boxprops=dict(facecolor=_SECONDARY, color=_PRIMARY, alpha=0.8),
                medianprops=dict(color=_ACCENT, linewidth=2),
                whiskerprops=dict(color=_PRIMARY),
                capprops=dict(color=_PRIMARY),
                flierprops=dict(marker="o", color=_ACCENT, markersize=4, alpha=0.5),
            )
            ax.set_yticks([])

        ax.set_title(t, **_TITLE_STYLE)
        ax.set_ylabel(column if group_by else "", fontsize=_LABEL_SIZE)
        plt.tight_layout()
        return fig_to_base64(fig)

    except Exception as exc:
        logger.error("create_box_plot failed: %s", exc)
        return _error_chart(f"Box plot error: {exc}")


# ── 8. Pie chart ──────────────────────────────────────────────────────────────

def create_pie_chart(labels: list, values: list, title: str) -> str:
    """Donut-style pie chart."""
    logger.debug("create_pie_chart — title=%r, n=%d", title, len(labels))
    try:
        fig, ax = _new_fig((8, 8))
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]

        wedges, texts, autotexts = ax.pie(
            values,
            labels=[str(l) for l in labels],
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
            pctdistance=0.82,
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
        )

        # Donut hole
        centre_circle = plt.Circle((0, 0), 0.60, fc="white")
        ax.add_patch(centre_circle)

        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")

        ax.set_title(title, **_TITLE_STYLE)
        plt.tight_layout()
        return fig_to_base64(fig)

    except Exception as exc:
        logger.error("create_pie_chart failed: %s", exc)
        return _error_chart(f"Pie chart error: {exc}")


# ── 9. Confusion matrix ───────────────────────────────────────────────────────

def create_confusion_matrix(
    y_true,
    y_pred,
    labels: list,
    title: str = "Confusion Matrix",
) -> str:
    """Annotated confusion matrix heatmap."""
    logger.debug("create_confusion_matrix — title=%r, classes=%d", title, len(labels))
    try:
        cm = sk_confusion_matrix(y_true, y_pred)
        str_labels = [str(l) for l in labels]
        n = len(str_labels)
        size = max(6, n * 0.9 + 1)
        fig, ax = plt.subplots(figsize=(size, size * 0.85))
        plt.style.use("seaborn-v0_8-whitegrid")

        # Normalised values for colour, raw counts for annotation
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        annot = np.array([
            [f"{cm[i, j]}\n({cm_norm[i, j]:.0%})" for j in range(n)]
            for i in range(n)
        ])

        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=str_labels,
            yticklabels=str_labels,
            ax=ax,
            linewidths=0.5,
            annot_kws={"size": max(8, 12 - n)},
            vmin=0,
            vmax=1,
        )
        ax.set_title(title, **_TITLE_STYLE)
        ax.set_xlabel("Predicted", fontsize=_LABEL_SIZE)
        ax.set_ylabel("Actual", fontsize=_LABEL_SIZE)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        return fig_to_base64(fig)

    except Exception as exc:
        logger.error("create_confusion_matrix failed: %s", exc)
        return _error_chart(f"Confusion matrix error: {exc}")


# ── 10. Feature importance ────────────────────────────────────────────────────

def create_feature_importance(
    features: list,
    importances: list,
    title: str = "Feature Importance",
) -> str:
    """Horizontal bar chart sorted by importance."""
    logger.debug("create_feature_importance — title=%r, n=%d", title, len(features))
    try:
        paired = sorted(zip(features, importances), key=lambda x: x[1])
        sorted_features = [p[0] for p in paired]
        sorted_values   = [p[1] for p in paired]
        n = len(sorted_features)

        fig, ax = _new_fig((10, max(5, n * 0.4 + 1)))

        max_val = max(sorted_values) if sorted_values else 1
        colors = [
            _PRIMARY if v == max_val else (_SECONDARY if v >= max_val * 0.5 else "#A5A5A5")
            for v in sorted_values
        ]

        bars = ax.barh(sorted_features, sorted_values, color=colors, edgecolor="white")
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)

        ax.set_title(title, **_TITLE_STYLE)
        ax.set_xlabel("Importance", fontsize=_LABEL_SIZE)
        ax.set_xlim(0, max_val * 1.15)
        plt.tight_layout()
        return fig_to_base64(fig)

    except Exception as exc:
        logger.error("create_feature_importance failed: %s", exc)
        return _error_chart(f"Feature importance error: {exc}")


# ── Error fallback ────────────────────────────────────────────────────────────

def _error_chart(message: str) -> str:
    """Return a minimal base64 PNG that displays an error message."""
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis("off")
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center",
        fontsize=11, color="#C00000",
        transform=ax.transAxes,
        wrap=True,
    )
    plt.tight_layout()
    return fig_to_base64(fig)
