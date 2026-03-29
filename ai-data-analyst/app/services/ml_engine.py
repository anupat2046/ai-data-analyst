import json
import logging
import os
import warnings
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report as sk_classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.services.eda_engine import fig_to_base64

# Directory where trained models are persisted (same as uploads so the volume covers it)
_MODEL_DIR = os.environ.get("UPLOAD_DIR", "uploads")

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_RS = 42  # global random_state
_C_NORMAL = "#2E5090"
_C_ANOMALY = "#ED7D31"
_CLUSTER_PALETTE = [
    "#2E5090", "#ED7D31", "#5B9BD5", "#70AD47",
    "#FFC000", "#FF0000", "#A5A5A5", "#4472C4",
    "#44546A", "#843C0C",
]


def _new_fig(figsize=(10, 6)):
    return plt.subplots(figsize=figsize)


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _validate_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")


# ─────────────────────────────────────────────────────────────────────────────

class MLEngine:
    """ML utilities: anomaly detection, clustering, and classification."""

    # ------------------------------------------------------------------ #
    # 1. anomaly_detection                                                 #
    # ------------------------------------------------------------------ #

    def anomaly_detection(
        self,
        df: pd.DataFrame,
        features: list[str] | None = None,
        contamination: float = 0.05,
    ) -> dict:
        logger.info("anomaly_detection started — shape=%s", df.shape)
        try:
            features = self._resolve_features(df, features)
            if not features:
                raise ValueError("No numeric columns available for anomaly detection.")

            X_raw = df[features].copy().dropna()
            if X_raw.empty:
                raise ValueError("All rows contain NaN after dropping — cannot proceed.")

            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)

            model = IsolationForest(contamination=contamination, random_state=_RS, n_jobs=-1)
            preds = model.fit_predict(X)          # 1 = normal, -1 = anomaly
            scores = model.score_samples(X)        # lower = more anomalous

            is_anomaly = preds == -1
            anomaly_count = int(is_anomaly.sum())
            anomaly_pct = round(anomaly_count / len(X_raw) * 100, 2)
            anomaly_indices = X_raw.index[is_anomaly].tolist()

            # ── Chart: 2-D scatter via PCA ──
            charts = [self._anomaly_scatter(X, is_anomaly, features)]

            # ── Top 10 anomaly rows ──
            score_series = pd.Series(scores[is_anomaly], index=anomaly_indices, name="anomaly_score")
            top10_idx = score_series.nsmallest(10).index
            sample_anomalies = (
                df.loc[top10_idx]
                .replace({float("nan"): None})
                .to_dict(orient="records")
            )

            # ── Feature contribution (mean absolute Z-score for anomaly vs normal) ──
            z_scores = pd.DataFrame(np.abs(X), columns=features, index=X_raw.index)
            anomaly_mean_z = z_scores.loc[anomaly_indices].mean().sort_values(ascending=False)
            top_features = anomaly_mean_z.head(5).index.tolist()

            insights = [
                f"Found {anomaly_count} anomalies ({anomaly_pct}% of {len(X_raw):,} rows).",
                f"Top anomalous features (by mean |Z-score|): {', '.join(top_features)}.",
            ]
            if anomaly_pct > 10:
                insights.append(
                    f"Anomaly rate {anomaly_pct}% is high — consider reviewing data quality "
                    "or lowering the contamination parameter."
                )

            logger.info("anomaly_detection done — %d anomalies found", anomaly_count)
            return {
                "total_rows": len(X_raw),
                "anomaly_count": anomaly_count,
                "anomaly_percentage": anomaly_pct,
                "anomaly_indices": anomaly_indices,
                "features_used": features,
                "charts": charts,
                "insights": insights,
                "sample_anomalies": sample_anomalies,
            }

        except Exception as exc:
            logger.error("anomaly_detection failed: %s", exc)
            return {"error": str(exc), "charts": [], "insights": [], "anomaly_count": 0}

    def _anomaly_scatter(
        self, X: np.ndarray, is_anomaly: np.ndarray, features: list[str]
    ) -> str:
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=_RS)
            X2d = pca.fit_transform(X)
            explained = pca.explained_variance_ratio_.sum() * 100
            xlabel = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
            ylabel = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
            title_suffix = f" — PCA 2D ({explained:.1f}% variance)"
        else:
            X2d = X if X.shape[1] == 2 else np.column_stack([X[:, 0], np.zeros(len(X))])
            xlabel = features[0]
            ylabel = features[1] if len(features) > 1 else ""
            title_suffix = ""

        fig, ax = _new_fig()
        ax.scatter(X2d[~is_anomaly, 0], X2d[~is_anomaly, 1],
                   c=_C_NORMAL, s=18, alpha=0.5, label=f"Normal ({(~is_anomaly).sum():,})")
        ax.scatter(X2d[is_anomaly, 0], X2d[is_anomaly, 1],
                   c=_C_ANOMALY, s=40, alpha=0.85, marker="x",
                   linewidths=1.5, label=f"Anomaly ({is_anomaly.sum():,})")
        ax.set_title(f"Anomaly Detection{title_suffix}", fontsize=13)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)
        plt.tight_layout()
        return fig_to_base64(fig)

    # ------------------------------------------------------------------ #
    # 2. clustering                                                        #
    # ------------------------------------------------------------------ #

    def clustering(
        self,
        df: pd.DataFrame,
        features: list[str] | None = None,
        n_clusters: int | None = None,
    ) -> dict:
        logger.info("clustering started — shape=%s, n_clusters=%s", df.shape, n_clusters)
        try:
            features = self._resolve_features(df, features)
            if not features:
                raise ValueError("No numeric columns available for clustering.")

            X_raw = df[features].dropna()
            if len(X_raw) < 4:
                raise ValueError("Need at least 4 rows (after dropping NaN) for clustering.")

            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)

            charts: list[str] = []

            # ── Elbow method to find optimal k ──
            k_range = range(2, min(11, len(X_raw)))
            inertias: list[float] = []
            sil_scores: list[float] = []

            for k in k_range:
                km = KMeans(n_clusters=k, random_state=_RS, n_init="auto")
                labels = km.fit_predict(X)
                inertias.append(km.inertia_)
                sil_scores.append(silhouette_score(X, labels))

            # Elbow curve chart
            charts.append(self._elbow_chart(list(k_range), inertias, sil_scores))

            # Choose k: highest silhouette if not provided
            if n_clusters is None:
                best_idx = int(np.argmax(sil_scores))
                n_clusters = list(k_range)[best_idx]
                logger.info("Optimal k chosen by silhouette: %d", n_clusters)

            # ── Final KMeans ──
            km_final = KMeans(n_clusters=n_clusters, random_state=_RS, n_init="auto")
            cluster_labels = km_final.fit_predict(X)
            final_sil = round(float(silhouette_score(X, cluster_labels)), 4)

            # Attach labels back to original index
            label_series = pd.Series(cluster_labels, index=X_raw.index, name="cluster")
            df_clustered = df.loc[X_raw.index].copy()
            df_clustered["cluster"] = label_series

            # ── Scatter chart (PCA 2D) ──
            charts.append(self._cluster_scatter(X, cluster_labels, n_clusters, features))

            # ── Cluster profile chart ──
            profile_df = pd.DataFrame(X, columns=features)
            profile_df["cluster"] = cluster_labels
            cluster_means = profile_df.groupby("cluster")[features].mean()
            charts.append(self._cluster_profile_chart(cluster_means, features))

            # ── Cluster sizes ──
            sizes = label_series.value_counts().sort_index()
            cluster_sizes = {f"cluster_{k}": int(v) for k, v in sizes.items()}

            # ── Cluster centers (inverse-transformed) ──
            centers_scaled = km_final.cluster_centers_
            centers_original = scaler.inverse_transform(centers_scaled)
            cluster_centers = [
                {feat: round(float(val), 4) for feat, val in zip(features, row)}
                for row in centers_original
            ]

            # ── Insights ──
            insights = [
                f"Optimal number of clusters: {n_clusters} (silhouette score={final_sil:.3f}).",
            ]
            for cid, center in enumerate(cluster_centers):
                top_feat = max(center, key=lambda k_: abs(center[k_]))
                insights.append(
                    f"Cluster {cid} ({cluster_sizes.get(f'cluster_{cid}', '?')} rows): "
                    f"highest '{top_feat}' = {center[top_feat]:.3g}."
                )
            if final_sil < 0.25:
                insights.append(
                    "Silhouette score is low — clusters may overlap. "
                    "Consider feature selection or a different k."
                )

            logger.info("clustering done — k=%d, silhouette=%.4f", n_clusters, final_sil)
            return {
                "n_clusters": n_clusters,
                "cluster_sizes": cluster_sizes,
                "cluster_centers": cluster_centers,
                "silhouette_score": final_sil,
                "features_used": features,
                "charts": charts,
                "insights": insights,
            }

        except Exception as exc:
            logger.error("clustering failed: %s", exc)
            return {"error": str(exc), "charts": [], "insights": [], "n_clusters": 0}

    def _elbow_chart(
        self, k_range: list[int], inertias: list[float], sil_scores: list[float]
    ) -> str:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(k_range, inertias, marker="o", color="#2E5090", linewidth=2)
        ax1.set_title("Elbow Curve — Inertia", fontsize=13)
        ax1.set_xlabel("Number of Clusters (k)")
        ax1.set_ylabel("Inertia")
        ax1.set_xticks(k_range)

        ax2.plot(k_range, sil_scores, marker="s", color="#ED7D31", linewidth=2)
        ax2.set_title("Silhouette Score vs k", fontsize=13)
        ax2.set_xlabel("Number of Clusters (k)")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_xticks(k_range)

        best_k_idx = int(np.argmax(sil_scores))
        ax2.axvline(k_range[best_k_idx], color="#5B9BD5", linestyle="--",
                    linewidth=1.5, label=f"Best k={k_range[best_k_idx]}")
        ax2.legend(fontsize=10)

        plt.tight_layout()
        return fig_to_base64(fig)

    def _cluster_scatter(
        self, X: np.ndarray, labels: np.ndarray, n_clusters: int, features: list[str]
    ) -> str:
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=_RS)
            X2d = pca.fit_transform(X)
            explained = pca.explained_variance_ratio_.sum() * 100
            xlabel = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
            ylabel = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
            title = f"Cluster Scatter — PCA 2D ({explained:.1f}% variance)"
        else:
            X2d = X if X.shape[1] == 2 else np.column_stack([X[:, 0], np.zeros(len(X))])
            xlabel = features[0]
            ylabel = features[1] if len(features) > 1 else ""
            title = "Cluster Scatter"

        fig, ax = _new_fig()
        for cid in range(n_clusters):
            mask = labels == cid
            ax.scatter(X2d[mask, 0], X2d[mask, 1],
                       c=_CLUSTER_PALETTE[cid % len(_CLUSTER_PALETTE)],
                       s=20, alpha=0.7, label=f"Cluster {cid} ({mask.sum():,})")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, markerscale=1.5)
        plt.tight_layout()
        return fig_to_base64(fig)

    def _cluster_profile_chart(
        self, cluster_means: pd.DataFrame, features: list[str]
    ) -> str:
        n_clusters = len(cluster_means)
        n_features = len(features)
        fig, ax = _new_fig((max(10, n_features * 1.2), 5))

        x = np.arange(n_features)
        bar_w = 0.8 / n_clusters
        for i, cid in enumerate(cluster_means.index):
            offset = (i - n_clusters / 2 + 0.5) * bar_w
            ax.bar(
                x + offset,
                cluster_means.loc[cid, features].values,
                width=bar_w,
                label=f"Cluster {cid}",
                color=_CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)],
                edgecolor="white",
            )

        ax.set_title("Cluster Profiles — Normalized Feature Means", fontsize=13)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Mean (standardized)")
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=10)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        plt.tight_layout()
        return fig_to_base64(fig)

    # ------------------------------------------------------------------ #
    # 3. classification_report                                             #
    # ------------------------------------------------------------------ #

    def classification_report(
        self,
        df: pd.DataFrame,
        target: str,
        features: list[str] | None = None,
        filename: str = "default",
    ) -> dict:
        logger.info("classification_report started — target='%s', file='%s'", target, filename)
        try:
            if target not in df.columns:
                raise ValueError(f"Target column '{target}' not found.")

            if features is None:
                features = [c for c in _numeric_cols(df) if c != target]
            else:
                _validate_columns(df, features)
                features = [c for c in features if c != target]

            if not features:
                raise ValueError("No feature columns available.")

            work = df[features + [target]].dropna()
            if len(work) < 20:
                raise ValueError("Need at least 20 rows (after dropping NaN) for classification.")

            X = work[features].values
            y_raw = work[target].values

            # Encode labels if needed
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            classes = le.classes_.tolist()
            is_binary = len(classes) == 2

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_scaled, y, test_size=0.2, random_state=_RS, stratify=y
            )

            # ── Train 3 models ──
            candidates = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, random_state=_RS, class_weight="balanced"
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, random_state=_RS, class_weight="balanced", n_jobs=-1
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=100, random_state=_RS
                ),
            }

            results: dict[str, dict] = {}
            for name, clf in candidates.items():
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)
                acc = round(float(accuracy_score(y_te, y_pred)), 4)
                results[name] = {"model": clf, "accuracy": acc, "y_pred": y_pred}
                logger.debug("%s accuracy=%.4f", name, acc)

            best_name = max(results, key=lambda k: results[k]["accuracy"])
            best = results[best_name]
            best_clf = best["model"]
            best_pred = best["y_pred"]
            best_acc = best["accuracy"]

            # ── Classification report ──
            cr = sk_classification_report(
                y_te, best_pred,
                target_names=[str(c) for c in classes],
                output_dict=True,
                zero_division=0,
            )

            # ── Feature importance ──
            feat_importance: dict[str, float] = {}
            if hasattr(best_clf, "feature_importances_"):
                feat_importance = {
                    f: round(float(v), 6)
                    for f, v in zip(features, best_clf.feature_importances_)
                }
            elif hasattr(best_clf, "coef_"):
                coefs = np.abs(best_clf.coef_).mean(axis=0)
                feat_importance = {
                    f: round(float(v), 6) for f, v in zip(features, coefs)
                }

            charts: list[str] = []

            # ── Confusion matrix chart ──
            charts.append(
                self._confusion_matrix_chart(y_te, best_pred, classes, best_name)
            )

            # ── Feature importance chart ──
            if feat_importance:
                charts.append(self._feature_importance_chart(feat_importance, best_name))

            # ── ROC curve (binary only) ──
            if is_binary and hasattr(best_clf, "predict_proba"):
                y_prob = best_clf.predict_proba(X_te)[:, 1]
                charts.append(self._roc_chart(y_te, y_prob, best_name))

            # ── Accuracy comparison chart ──
            charts.append(
                self._accuracy_comparison_chart({n: r["accuracy"] for n, r in results.items()})
            )

            # ── Insights ──
            insights = [
                f"Best model: {best_name} (accuracy={best_acc:.2%}).",
            ]
            model_summary = " | ".join(
                f"{n}: {r['accuracy']:.2%}" for n, r in results.items()
            )
            insights.append(f"All models — {model_summary}.")
            if feat_importance:
                top_feat = max(feat_importance, key=lambda k: feat_importance[k])
                insights.append(
                    f"Most important feature: '{top_feat}' "
                    f"(importance={feat_importance[top_feat]:.4f})."
                )
            if best_acc < 0.6:
                insights.append(
                    "Accuracy is below 60% — consider feature engineering, "
                    "more data, or hyperparameter tuning."
                )

            # ── Persist best model to disk (worker-safe, survives restarts) ──────
            # Key = stem(filename) + target — prevents two users with the same
            # target column name (e.g. "status") from overwriting each other's model.
            file_stem = os.path.splitext(os.path.basename(filename))[0]
            model_key = f"{file_stem}__{target}"
            model_bundle = {
                "model": best_clf,
                "scaler": scaler,
                "le": le,
                "features": features,
                "model_name": best_name,
                "model_key": model_key,
            }
            model_path = os.path.join(_MODEL_DIR, f"{model_key}_model.joblib")
            try:
                os.makedirs(_MODEL_DIR, exist_ok=True)
                joblib.dump(model_bundle, model_path)
                logger.info("Model saved → %s", model_path)
            except Exception as save_exc:
                logger.warning("Could not save model to disk: %s", save_exc)

            logger.info("classification_report done — best=%s acc=%.4f", best_name, best_acc)
            return {
                "best_model": best_name,
                "accuracy": best_acc,
                "all_model_accuracies": {n: r["accuracy"] for n, r in results.items()},
                "classification_report": cr,
                "feature_importance": feat_importance,
                "features_used": features,
                "classes": [str(c) for c in classes],
                "model_key": model_key,   # pass to quick_predict
                "charts": charts,
                "insights": insights,
            }

        except Exception as exc:
            logger.error("classification_report failed: %s", exc)
            return {"error": str(exc), "charts": [], "insights": []}

    def _confusion_matrix_chart(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: list,
        model_name: str,
    ) -> str:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = _new_fig((max(6, len(classes)), max(5, len(classes))))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[str(c) for c in classes],
            yticklabels=[str(c) for c in classes],
            ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        plt.tight_layout()
        return fig_to_base64(fig)

    def _feature_importance_chart(
        self, importance: dict[str, float], model_name: str
    ) -> str:
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
        labels, vals = zip(*sorted_imp)

        fig, ax = _new_fig((10, max(5, len(labels) * 0.4 + 1)))
        colors = ["#2E5090" if v == max(vals) else "#5B9BD5" for v in vals]
        bars = ax.barh(list(reversed(labels)), list(reversed(vals)), color=list(reversed(colors)),
                       edgecolor="white")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_title(f"Feature Importance — {model_name}", fontsize=13)
        ax.set_xlabel("Importance")
        plt.tight_layout()
        return fig_to_base64(fig)

    def _roc_chart(
        self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str
    ) -> str:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = round(float(roc_auc_score(y_true, y_prob)), 4)

        fig, ax = _new_fig((8, 6))
        ax.plot(fpr, tpr, color="#2E5090", linewidth=2, label=f"ROC (AUC={auc})")
        ax.plot([0, 1], [0, 1], color="#A5A5A5", linestyle="--", linewidth=1, label="Random")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#2E5090")
        ax.set_title(f"ROC Curve — {model_name}", fontsize=13)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=10)
        plt.tight_layout()
        return fig_to_base64(fig)

    def _accuracy_comparison_chart(self, accuracies: dict[str, float]) -> str:
        names = list(accuracies.keys())
        vals = list(accuracies.values())
        best_val = max(vals)

        fig, ax = _new_fig((9, 5))
        colors = ["#2E5090" if v == best_val else "#5B9BD5" for v in vals]
        bars = ax.bar(names, vals, color=colors, edgecolor="white", width=0.5)
        ax.bar_label(bars, fmt="%.2%", padding=4, fontsize=10)
        ax.set_ylim(0, min(1.15, max(vals) + 0.15))
        ax.set_title("Model Accuracy Comparison", fontsize=13)
        ax.set_ylabel("Accuracy")
        ax.axhline(0.5, color="#ED7D31", linestyle="--", linewidth=1, label="Baseline 50%")
        ax.legend(fontsize=10)
        plt.tight_layout()
        return fig_to_base64(fig)

    # ------------------------------------------------------------------ #
    # 4. quick_predict                                                     #
    # ------------------------------------------------------------------ #

    def quick_predict(
        self,
        df: pd.DataFrame,
        target: str,
        new_data: dict,
        filename: str = "default",
    ) -> dict:
        """Load saved model from disk (worker-safe) and predict on new_data.

        Model file is keyed by stem(filename) + target to prevent cross-user
        collisions when different datasets share the same target column name.
        Falls back to training a new model if no saved file exists.
        """
        logger.info("quick_predict called — target='%s', file='%s'", target, filename)
        try:
            file_stem = os.path.splitext(os.path.basename(filename))[0]
            model_key = f"{file_stem}__{target}"
            model_path = os.path.join(_MODEL_DIR, f"{model_key}_model.joblib")

            if not os.path.exists(model_path):
                logger.info("No saved model for key '%s' — training first.", model_key)
                train_result = self.classification_report(df, target, filename=filename)
                if "error" in train_result:
                    return {"error": f"Training failed: {train_result['error']}"}
                # classification_report saves the file; verify it now exists
                if not os.path.exists(model_path):
                    return {"error": "Model file was not saved correctly."}

            bundle = joblib.load(model_path)
            features:   list[str]    = bundle["features"]
            model                    = bundle["model"]
            scaler:     StandardScaler = bundle["scaler"]
            le:         LabelEncoder   = bundle["le"]
            model_name: str            = bundle["model_name"]

            # Build input row — fill missing features with 0
            row = np.array([[float(new_data.get(f, 0)) for f in features]])
            row_scaled = scaler.transform(row)

            raw_pred = model.predict(row_scaled)[0]
            prediction = le.inverse_transform([raw_pred])[0]

            probability: dict[str, float] = {}
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(row_scaled)[0]
                probability = {
                    str(cls): round(float(p), 4)
                    for cls, p in zip(le.classes_, proba)
                }

            logger.info("quick_predict done — prediction=%s", prediction)
            return {
                "prediction": str(prediction),
                "probability": probability,
                "model_used": model_name,
                "features_used": features,
            }

        except Exception as exc:
            logger.error("quick_predict failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------ #
    # Shared helpers                                                       #
    # ------------------------------------------------------------------ #

    def _resolve_features(
        self, df: pd.DataFrame, features: list[str] | None
    ) -> list[str]:
        if features is None:
            return _numeric_cols(df)
        _validate_columns(df, features)
        non_numeric = [
            c for c in features
            if not pd.api.types.is_numeric_dtype(df[c])
        ]
        if non_numeric:
            raise ValueError(f"Non-numeric columns in features: {non_numeric}")
        return features
