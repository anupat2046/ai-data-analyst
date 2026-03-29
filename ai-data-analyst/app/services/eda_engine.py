import io
import base64
import logging
import warnings
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Style constants ──────────────────────────────────────────────────────────
_C1, _C2, _C3 = "#2E5090", "#5B9BD5", "#ED7D31"
_PALETTE = [_C1, _C2, _C3, "#A5A5A5", "#FFC000", "#70AD47", "#FF0000"]
_FIGSIZE = (10, 6)
_FONT_SIZE = 12

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"font.size": _FONT_SIZE})


# ── Helper ───────────────────────────────────────────────────────────────────

def fig_to_base64(fig: plt.Figure) -> str:
    """Serialize a Matplotlib figure to a base64-encoded PNG string, then close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _new_fig(figsize: tuple = _FIGSIZE) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ── Main class ───────────────────────────────────────────────────────────────

class EDAEngine:
    """Automatic EDA — all chart methods return base64-encoded PNG strings."""

    # ------------------------------------------------------------------ #
    # 1. auto_eda                                                          #
    # ------------------------------------------------------------------ #

    def auto_eda(self, df: pd.DataFrame) -> dict:
        """Run a full automatic EDA and return a comprehensive report dict."""
        logger.info("Starting auto_eda for DataFrame %s", df.shape)
        try:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = df.select_dtypes(include="object").columns.tolist()

            result: dict[str, Any] = {
                "summary_stats": self._summary_stats(df),
                "missing_analysis": self._missing_analysis(df),
                "correlation_matrix": "",
                "distribution_charts": [],
                "categorical_charts": [],
                "insights": [],
            }

            # Correlation heatmap
            if len(num_cols) >= 2:
                corr_data = self.correlation_analysis(df)
                result["correlation_matrix"] = corr_data["chart"]
                result["insights"] += corr_data["insights"]

            # Distribution charts for numeric columns
            for col in num_cols:
                try:
                    dist = self.distribution_analysis(df, col)
                    result["distribution_charts"].append({
                        "column": col,
                        "charts": dist["charts"],
                    })
                    result["insights"] += dist["insights"]
                except Exception as exc:
                    logger.warning("distribution_analysis skipped for %s: %s", col, exc)

            # Bar charts for categorical columns
            for col in cat_cols:
                try:
                    cat = self.categorical_analysis(df, col)
                    result["categorical_charts"].append({
                        "column": col,
                        "charts": cat["charts"],
                    })
                except Exception as exc:
                    logger.warning("categorical_analysis skipped for %s: %s", col, exc)

            # Generic insights
            result["insights"] = list(dict.fromkeys(result["insights"]))  # deduplicate
            result["insights"] += self._generic_insights(df)

            logger.info("auto_eda complete — %d insights generated", len(result["insights"]))
            return result

        except Exception as exc:
            logger.error("auto_eda failed: %s", exc)
            raise RuntimeError(f"auto_eda failed: {exc}") from exc

    def _summary_stats(self, df: pd.DataFrame) -> dict:
        num_df = df.select_dtypes(include="number")
        if num_df.empty:
            return {}
        return num_df.describe().round(4).to_dict()

    def _missing_analysis(self, df: pd.DataFrame) -> dict:
        counts = df.isnull().sum()
        pct = (df.isnull().mean() * 100).round(2)
        return {
            col: {"count": int(counts[col]), "percent": float(pct[col])}
            for col in df.columns
        }

    def _generic_insights(self, df: pd.DataFrame) -> list[str]:
        insights: list[str] = []
        n = len(df)
        # Missing
        for col in df.columns:
            pct = df[col].isnull().mean() * 100
            if pct >= 50:
                insights.append(f"Column '{col}' has {pct:.1f}% missing values — consider dropping it.")
            elif pct >= 20:
                insights.append(f"Column '{col}' has {pct:.1f}% missing values.")
        # Duplicates
        dups = df.duplicated().sum()
        if dups:
            insights.append(f"Dataset contains {dups} duplicate rows ({dups/n*100:.1f}%).")
        # Dataset size
        insights.append(f"Dataset has {n:,} rows and {df.shape[1]} columns.")
        return insights

    # ------------------------------------------------------------------ #
    # 2. correlation_analysis                                              #
    # ------------------------------------------------------------------ #

    def correlation_analysis(self, df: pd.DataFrame) -> dict:
        """Heatmap + top correlated pairs + insights."""
        logger.info("Running correlation_analysis")
        try:
            num_df = df.select_dtypes(include="number")
            if num_df.shape[1] < 2:
                return {"chart": "", "top_correlations": [], "insights": [
                    "Not enough numeric columns for correlation analysis (need ≥ 2)."
                ]}

            corr = num_df.corr()

            # ── Heatmap ──
            size = max(8, min(corr.shape[0], 16))
            fig, ax = plt.subplots(figsize=(size, size * 0.8))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(
                corr,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap=sns.diverging_palette(220, 20, as_cmap=True),
                ax=ax,
                linewidths=0.5,
                annot_kws={"size": 9},
            )
            ax.set_title("Correlation Matrix", fontsize=_FONT_SIZE + 2, pad=12)
            plt.tight_layout()
            chart_b64 = fig_to_base64(fig)

            # ── Top correlated pairs ──
            pairs: list[dict] = []
            cols = corr.columns.tolist()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    r = corr.iloc[i, j]
                    if not np.isnan(r):
                        pairs.append({"col1": cols[i], "col2": cols[j], "r": round(float(r), 4)})
            pairs.sort(key=lambda x: abs(x["r"]), reverse=True)
            top10 = pairs[:10]

            # ── Insights ──
            insights: list[str] = []
            for p in top10:
                r, c1, c2 = p["r"], p["col1"], p["col2"]
                if abs(r) >= 0.9:
                    insights.append(f"'{c1}' and '{c2}' are very strongly correlated (r={r:.2f}).")
                elif abs(r) >= 0.7:
                    insights.append(f"'{c1}' and '{c2}' are strongly correlated (r={r:.2f}).")

            logger.info("correlation_analysis complete — %d pairs analysed", len(pairs))
            return {"chart": chart_b64, "top_correlations": top10, "insights": insights}

        except Exception as exc:
            logger.error("correlation_analysis failed: %s", exc)
            raise RuntimeError(f"correlation_analysis failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # 3. distribution_analysis                                             #
    # ------------------------------------------------------------------ #

    def distribution_analysis(self, df: pd.DataFrame, column: str) -> dict:
        """Histogram+KDE, boxplot, QQ-plot, stats, normality test."""
        logger.info("distribution_analysis for column '%s'", column)
        try:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")

            series = df[column].dropna()
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(f"Column '{column}' is not numeric.")

            charts: list[str] = []
            insights: list[str] = []

            # ── Histogram + KDE ──
            fig, ax = _new_fig()
            ax.hist(series, bins="auto", color=_C1, edgecolor="white", alpha=0.85, density=True)
            try:
                series.plot.kde(ax=ax, color=_C3, linewidth=2)
            except Exception:
                pass
            ax.set_title(f"Distribution of {column}", fontsize=_FONT_SIZE + 1)
            ax.set_xlabel(column)
            ax.set_ylabel("Density")
            plt.tight_layout()
            charts.append(fig_to_base64(fig))

            # ── Box plot ──
            fig, ax = _new_fig((8, 5))
            ax.boxplot(series, orientation="horizontal", patch_artist=True,
                       boxprops=dict(facecolor=_C2, color=_C1),
                       medianprops=dict(color=_C3, linewidth=2))
            ax.set_title(f"Box Plot — {column}", fontsize=_FONT_SIZE + 1)
            ax.set_xlabel(column)
            ax.set_yticks([])
            plt.tight_layout()
            charts.append(fig_to_base64(fig))

            # ── QQ plot ──
            fig, ax = _new_fig((7, 6))
            (osm, osr), (slope, intercept, r) = scipy_stats.probplot(series, dist="norm")
            ax.scatter(osm, osr, color=_C1, s=10, alpha=0.7, label="Data")
            fit_line = np.array(osm) * slope + intercept
            ax.plot(osm, fit_line, color=_C3, linewidth=1.5, label="Normal fit")
            ax.set_title(f"Q-Q Plot — {column}", fontsize=_FONT_SIZE + 1)
            ax.set_xlabel("Theoretical quantiles")
            ax.set_ylabel("Sample quantiles")
            ax.legend(fontsize=10)
            plt.tight_layout()
            charts.append(fig_to_base64(fig))

            # ── Statistics ──
            skewness = float(series.skew())
            kurt = float(series.kurtosis())
            col_stats = {
                "count": int(series.count()),
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "skewness": round(skewness, 4),
                "kurtosis": round(kurt, 4),
                "q1": round(float(series.quantile(0.25)), 4),
                "q3": round(float(series.quantile(0.75)), 4),
            }

            # ── Normality test ──
            n = len(series)
            if n < 5000:
                stat, p_val = scipy_stats.shapiro(series.sample(min(n, 5000), random_state=42))
                test_name = "Shapiro-Wilk"
            else:
                stat, p_val = scipy_stats.kstest(
                    series, "norm",
                    args=(float(series.mean()), float(series.std()))
                )
                test_name = "Kolmogorov-Smirnov"

            col_stats["normality_test"] = {
                "test": test_name,
                "statistic": round(float(stat), 6),
                "p_value": round(float(p_val), 6),
                "is_normal": bool(p_val > 0.05),
            }

            # ── Insights ──
            if abs(skewness) > 2:
                direction = "right" if skewness > 0 else "left"
                insights.append(f"'{column}' is heavily {direction}-skewed (skewness={skewness:.2f}).")
            elif abs(skewness) > 1:
                direction = "right" if skewness > 0 else "left"
                insights.append(f"'{column}' is moderately {direction}-skewed (skewness={skewness:.2f}).")

            if p_val <= 0.05:
                insights.append(f"'{column}' does not follow a normal distribution ({test_name} p={p_val:.4f}).")
            else:
                insights.append(f"'{column}' appears normally distributed ({test_name} p={p_val:.4f}).")

            # Outlier rough count via IQR
            q1, q3 = col_stats["q1"], col_stats["q3"]
            iqr = q3 - q1
            outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
            if len(outliers):
                insights.append(
                    f"'{column}' has {len(outliers)} potential outliers ({len(outliers)/n*100:.1f}%) by IQR rule."
                )

            logger.info("distribution_analysis done for '%s'", column)
            return {"charts": charts, "stats": col_stats, "insights": insights}

        except Exception as exc:
            logger.error("distribution_analysis failed for '%s': %s", column, exc)
            raise RuntimeError(f"distribution_analysis failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # 4. categorical_analysis                                              #
    # ------------------------------------------------------------------ #

    def categorical_analysis(self, df: pd.DataFrame, column: str) -> dict:
        """Value-counts bar chart, optional pie chart, and stats."""
        logger.info("categorical_analysis for column '%s'", column)
        try:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")

            series = df[column].dropna()
            vc = series.value_counts()
            n_unique = int(series.nunique())
            charts: list[str] = []

            # ── Bar chart (top 20) ──
            top_vc = vc.head(20)
            fig, ax = _new_fig((max(10, min(n_unique, 20)) * 0.6 + 2, 6))
            colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(top_vc))]
            bars = ax.bar(top_vc.index.astype(str), top_vc.values, color=colors, edgecolor="white")
            ax.bar_label(bars, padding=3, fontsize=9)
            ax.set_title(f"Value Counts — {column}", fontsize=_FONT_SIZE + 1)
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right", fontsize=9)
            plt.tight_layout()
            charts.append(fig_to_base64(fig))

            # ── Pie chart (only for low-cardinality columns) ──
            if n_unique <= 10:
                fig, ax = _new_fig((8, 8))
                wedge_colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(vc))]
                ax.pie(
                    vc.values,
                    labels=vc.index.astype(str),
                    autopct="%1.1f%%",
                    colors=wedge_colors,
                    startangle=140,
                    pctdistance=0.82,
                )
                ax.set_title(f"Distribution — {column}", fontsize=_FONT_SIZE + 1)
                plt.tight_layout()
                charts.append(fig_to_base64(fig))

            # ── Stats ──
            mode_val = series.mode().iloc[0] if not series.mode().empty else None
            col_stats = {
                "total_count": int(series.count()),
                "unique_count": n_unique,
                "missing_count": int(df[column].isnull().sum()),
                "missing_percent": round(df[column].isnull().mean() * 100, 2),
                "mode": str(mode_val) if mode_val is not None else None,
                "mode_frequency": int(vc.iloc[0]) if not vc.empty else 0,
                "mode_percent": round(float(vc.iloc[0] / len(series) * 100), 2) if not vc.empty else 0.0,
                "top_values": vc.head(10).to_dict(),
            }

            logger.info("categorical_analysis done for '%s'", column)
            return {"charts": charts, "stats": col_stats}

        except Exception as exc:
            logger.error("categorical_analysis failed for '%s': %s", column, exc)
            raise RuntimeError(f"categorical_analysis failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # 5. bivariate_analysis                                                #
    # ------------------------------------------------------------------ #

    def bivariate_analysis(self, df: pd.DataFrame, col1: str, col2: str) -> dict:
        """Scatter / boxplot-by-group / count-heatmap depending on column types."""
        logger.info("bivariate_analysis: '%s' vs '%s'", col1, col2)
        try:
            for col in (col1, col2):
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame.")

            is_num1 = pd.api.types.is_numeric_dtype(df[col1])
            is_num2 = pd.api.types.is_numeric_dtype(df[col2])

            pair_df = df[[col1, col2]].dropna()
            insights: list[str] = []
            col_stats: dict[str, Any] = {}

            # ── Both numeric → scatter + regression line ──
            if is_num1 and is_num2:
                fig, ax = _new_fig()
                ax.scatter(pair_df[col1], pair_df[col2], alpha=0.5, color=_C1, s=20, label="Data")
                # Regression line
                m, b, r, p, _ = scipy_stats.linregress(pair_df[col1], pair_df[col2])
                x_line = np.linspace(pair_df[col1].min(), pair_df[col1].max(), 200)
                ax.plot(x_line, m * x_line + b, color=_C3, linewidth=2, label=f"Fit (r={r:.2f})")
                ax.set_title(f"Scatter — {col1} vs {col2}", fontsize=_FONT_SIZE + 1)
                ax.set_xlabel(col1)
                ax.set_ylabel(col2)
                ax.legend(fontsize=10)
                plt.tight_layout()
                chart_b64 = fig_to_base64(fig)

                corr = round(float(pair_df[col1].corr(pair_df[col2])), 4)
                col_stats = {"pearson_r": corr, "r_squared": round(corr ** 2, 4),
                             "slope": round(m, 4), "intercept": round(b, 4),
                             "p_value": round(p, 6)}
                if abs(corr) >= 0.7:
                    insights.append(f"'{col1}' and '{col2}' are strongly correlated (r={corr}).")
                elif abs(corr) >= 0.4:
                    insights.append(f"'{col1}' and '{col2}' have moderate correlation (r={corr}).")
                else:
                    insights.append(f"'{col1}' and '{col2}' have weak correlation (r={corr}).")

            # ── One numeric + one categorical → grouped box plot ──
            elif is_num1 != is_num2:
                num_col, cat_col = (col1, col2) if is_num1 else (col2, col1)
                top_cats = pair_df[cat_col].value_counts().head(10).index
                filtered = pair_df[pair_df[cat_col].isin(top_cats)]

                fig, ax = _new_fig()
                groups = [filtered[filtered[cat_col] == cat][num_col].dropna().values
                          for cat in top_cats]
                bp = ax.boxplot(groups, patch_artist=True, vert=True)
                for patch, color in zip(bp["boxes"], _PALETTE):
                    patch.set_facecolor(color)
                for median in bp["medians"]:
                    median.set_color(_C3)
                ax.set_xticklabels(top_cats, rotation=45, ha="right", fontsize=9)
                ax.set_title(f"Box Plot — {num_col} by {cat_col}", fontsize=_FONT_SIZE + 1)
                ax.set_xlabel(cat_col)
                ax.set_ylabel(num_col)
                plt.tight_layout()
                chart_b64 = fig_to_base64(fig)

                group_means = filtered.groupby(cat_col)[num_col].mean().round(4).to_dict()
                col_stats = {"group_means": group_means, "groups_shown": len(top_cats)}
                insights.append(
                    f"Median '{num_col}' varies across '{cat_col}' categories."
                )

            # ── Both categorical → count heatmap ──
            else:
                top1 = df[col1].value_counts().head(10).index
                top2 = df[col2].value_counts().head(10).index
                ct = pd.crosstab(
                    pair_df[col1][pair_df[col1].isin(top1)],
                    pair_df[col2][pair_df[col2].isin(top2)],
                )
                fig, ax = _new_fig((10, max(5, len(ct) * 0.5 + 2)))
                sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax, linewidths=0.4)
                ax.set_title(f"Count Heatmap — {col1} × {col2}", fontsize=_FONT_SIZE + 1)
                plt.tight_layout()
                chart_b64 = fig_to_base64(fig)

                col_stats = {"crosstab_shape": list(ct.shape)}
                insights.append(
                    f"Cross-tabulation of '{col1}' vs '{col2}' shows {ct.shape[0]}×{ct.shape[1]} combinations."
                )

            logger.info("bivariate_analysis complete for '%s' vs '%s'", col1, col2)
            return {"chart": chart_b64, "stats": col_stats, "insights": insights}

        except Exception as exc:
            logger.error("bivariate_analysis failed: %s", exc)
            raise RuntimeError(f"bivariate_analysis failed: {exc}") from exc

    # ------------------------------------------------------------------ #
    # 6. time_series_analysis                                              #
    # ------------------------------------------------------------------ #

    def time_series_analysis(
        self, df: pd.DataFrame, date_col: str, value_col: str
    ) -> dict:
        """Line trend, aggregations, moving averages, and optional decomposition."""
        logger.info("time_series_analysis: date='%s', value='%s'", date_col, value_col)
        try:
            for col in (date_col, value_col):
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame.")

            ts_df = df[[date_col, value_col]].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
            ts_df = ts_df.dropna().sort_values(date_col).set_index(date_col)
            series = ts_df[value_col].astype(float)

            charts: list[str] = []
            insights: list[str] = []
            ts_stats: dict[str, Any] = {
                "start_date": str(series.index.min().date()),
                "end_date": str(series.index.max().date()),
                "n_points": len(series),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
            }

            # ── 1. Main trend line + moving averages ──
            fig, ax = _new_fig((12, 6))
            ax.plot(series.index, series.values, color=_C1, linewidth=1.2,
                    alpha=0.8, label=value_col)

            n = len(series)
            if n >= 7:
                ma7 = series.rolling(7).mean()
                ax.plot(ma7.index, ma7.values, color=_C3, linewidth=2, label="7-day MA")
            if n >= 30:
                ma30 = series.rolling(30).mean()
                ax.plot(ma30.index, ma30.values, color=_C2, linewidth=2,
                        linestyle="--", label="30-day MA")

            ax.set_title(f"Time Series — {value_col}", fontsize=_FONT_SIZE + 1)
            ax.set_xlabel("Date")
            ax.set_ylabel(value_col)
            ax.legend(fontsize=10)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            charts.append(fig_to_base64(fig))

            # ── 2. Monthly aggregation ──
            try:
                monthly = series.resample("ME").mean()
                if len(monthly) >= 2:
                    fig, ax = _new_fig((12, 5))
                    ax.bar(monthly.index, monthly.values, color=_C2,
                           edgecolor="white", width=20)
                    ax.set_title(f"Monthly Average — {value_col}", fontsize=_FONT_SIZE + 1)
                    ax.set_xlabel("Month")
                    ax.set_ylabel(f"Avg {value_col}")
                    plt.xticks(rotation=30, ha="right")
                    plt.tight_layout()
                    charts.append(fig_to_base64(fig))
                    ts_stats["monthly_avg"] = {
                        str(k.date()): round(float(v), 4)
                        for k, v in monthly.items()
                        if not np.isnan(v)
                    }
            except Exception as exc:
                logger.warning("Monthly aggregation skipped: %s", exc)

            # ── 3. Weekly aggregation ──
            try:
                weekly = series.resample("W").mean()
                if len(weekly) >= 4:
                    fig, ax = _new_fig((12, 5))
                    ax.plot(weekly.index, weekly.values, color=_C1,
                            linewidth=1.5, marker="o", markersize=4)
                    ax.set_title(f"Weekly Average — {value_col}", fontsize=_FONT_SIZE + 1)
                    ax.set_xlabel("Week")
                    ax.set_ylabel(f"Avg {value_col}")
                    plt.xticks(rotation=30, ha="right")
                    plt.tight_layout()
                    charts.append(fig_to_base64(fig))
            except Exception as exc:
                logger.warning("Weekly aggregation skipped: %s", exc)

            # ── 4. Seasonal decomposition (need ≥ 2 full periods) ──
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose

                # Require at least 2 * period data points
                period = 12 if n >= 24 else 7 if n >= 14 else None
                if period and len(series.dropna()) >= 2 * period:
                    decomp = seasonal_decompose(
                        series.dropna(), model="additive", period=period, extrapolate_trend="freq"
                    )
                    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                    for ax_, component, label, color in zip(
                        axes,
                        [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
                        ["Observed", "Trend", "Seasonal", "Residual"],
                        [_C1, _C2, _C3, "#A5A5A5"],
                    ):
                        ax_.plot(component.index, component.values, color=color, linewidth=1.2)
                        ax_.set_ylabel(label, fontsize=10)
                        ax_.grid(True, alpha=0.3)
                    axes[0].set_title(
                        f"Seasonal Decomposition — {value_col} (period={period})",
                        fontsize=_FONT_SIZE + 1,
                    )
                    plt.tight_layout()
                    charts.append(fig_to_base64(fig))
                    insights.append(
                        f"Seasonal decomposition applied with period={period} on '{value_col}'."
                    )
            except ImportError:
                logger.warning("statsmodels not installed — skipping seasonal decomposition.")
            except Exception as exc:
                logger.warning("Seasonal decomposition skipped: %s", exc)

            # ── Trend direction insight ──
            if n >= 2:
                first_half = series.iloc[: n // 2].mean()
                second_half = series.iloc[n // 2 :].mean()
                change_pct = (second_half - first_half) / abs(first_half) * 100 if first_half != 0 else 0
                direction = "upward" if change_pct > 5 else "downward" if change_pct < -5 else "stable"
                insights.append(
                    f"'{value_col}' shows a {direction} trend "
                    f"({change_pct:+.1f}% change from first to second half)."
                )

            logger.info("time_series_analysis complete — %d charts", len(charts))
            return {"charts": charts, "stats": ts_stats, "insights": insights}

        except Exception as exc:
            logger.error("time_series_analysis failed: %s", exc)
            raise RuntimeError(f"time_series_analysis failed: {exc}") from exc
