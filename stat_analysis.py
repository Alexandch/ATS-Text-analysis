from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_labels import feature_label
from plot_style import ATS_SCORE_LABEL, PEARSON_CORRELATION_LABEL, apply_standard_plot_style, wrap_label


TARGET_COLUMN = "score"
TOP_HEATMAP_FEATURES = 12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Статистический анализ связи текстовых признаков с ATS-оценкой."
    )
    parser.add_argument(
        "--input",
        default="features.csv",
        help="Путь к CSV с признаками.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Папка для сохранения результатов анализа.",
    )
    parser.add_argument(
        "--jaccard-threshold",
        type=float,
        default=None,
        help="Порог для разделения групп по jaccard_similarity. По умолчанию берется медиана.",
    )
    return parser.parse_args()


def load_data(path):
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"В файле отсутствует колонка {TARGET_COLUMN}")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [column for column in numeric_columns if column != TARGET_COLUMN]
    if not feature_columns:
        raise ValueError("Не найдены числовые признаки для анализа.")

    return df[feature_columns + [TARGET_COLUMN]].dropna().copy(), feature_columns


def prettify_label(label):
    """Делает длинные имена признаков более читаемыми на графиках."""
    return wrap_label(feature_label(label), width=18)


def save_descriptive_stats(df, output_dir):
    descriptive = df.describe().T
    descriptive["variance"] = df.var()
    descriptive.to_csv(output_dir / "descriptive_stats.csv", encoding="utf-8-sig")
    return descriptive


def save_correlations(df, feature_columns, output_dir):
    rows = []
    for feature in feature_columns:
        pearson_r, pearson_p = stats.pearsonr(df[feature], df[TARGET_COLUMN])
        spearman_rho, spearman_p = stats.spearmanr(df[feature], df[TARGET_COLUMN])
        rows.append(
            {
                "feature": feature,
                "pearson_r": pearson_r,
                "pearson_p_value": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p_value": spearman_p,
            }
        )

    correlations = pd.DataFrame(rows).sort_values(
        by="pearson_r", key=lambda series: series.abs(), ascending=False
    )
    correlations.to_csv(output_dir / "correlations_with_score.csv", index=False, encoding="utf-8-sig")

    corr_matrix = df.corr(numeric_only=True)
    corr_matrix.to_csv(output_dir / "correlation_matrix.csv", encoding="utf-8-sig")
    return correlations, corr_matrix


def run_group_test(df, output_dir, threshold=None):
    if threshold is None:
        threshold = float(df["jaccard_similarity"].median())

    low_group = df[df["jaccard_similarity"] < threshold][TARGET_COLUMN]
    high_group = df[df["jaccard_similarity"] >= threshold][TARGET_COLUMN]

    t_stat, t_p = stats.ttest_ind(high_group, low_group, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(high_group, low_group, alternative="two-sided")

    result = pd.DataFrame(
        [
            {
                "threshold": threshold,
                "low_group_size": len(low_group),
                "high_group_size": len(high_group),
                "low_group_mean_score": low_group.mean(),
                "high_group_mean_score": high_group.mean(),
                "mean_difference": high_group.mean() - low_group.mean(),
                "welch_t_statistic": t_stat,
                "welch_t_p_value": t_p,
                "mannwhitney_u_statistic": u_stat,
                "mannwhitney_u_p_value": u_p,
            }
        ]
    )
    result.to_csv(output_dir / "jaccard_group_test.csv", index=False, encoding="utf-8-sig")
    return result


def run_anova(df, output_dir):
    anova_df = df.copy()
    anova_df["cosine_quartile"] = pd.qcut(
        anova_df["cosine_similarity"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    )

    grouped_scores = [
        anova_df.loc[anova_df["cosine_quartile"] == label, TARGET_COLUMN]
        for label in ["Q1", "Q2", "Q3", "Q4"]
    ]
    f_stat, p_value = stats.f_oneway(*grouped_scores)

    summary = (
        anova_df.groupby("cosine_quartile", observed=False)[TARGET_COLUMN]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    summary["anova_f_statistic"] = f_stat
    summary["anova_p_value"] = p_value
    summary.to_csv(output_dir / "cosine_quartile_anova.csv", index=False, encoding="utf-8-sig")
    return summary, f_stat, p_value


def fit_ols_with_pvalues(df, feature_columns, output_dir):
    x = df[feature_columns].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)

    x_design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.pinv(x_design.T @ x_design) @ x_design.T @ y
    y_pred = x_design @ beta
    residuals = y - y_pred

    n = len(y)
    p = x_design.shape[1]
    sigma_squared = (residuals @ residuals) / (n - p)
    covariance = sigma_squared * np.linalg.pinv(x_design.T @ x_design)
    standard_errors = np.sqrt(np.diag(covariance))
    t_stats = beta / standard_errors
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p))

    result = pd.DataFrame(
        {
            "term": ["intercept"] + feature_columns,
            "coefficient": beta,
            "std_error": standard_errors,
            "t_statistic": t_stats,
            "p_value": p_values,
        }
    )
    result.to_csv(output_dir / "ols_coefficients.csv", index=False, encoding="utf-8-sig")
    return result


def evaluate_predictive_models(df, feature_columns, output_dir):
    x = df[feature_columns]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = LinearRegression()
    model.fit(x_train_scaled, y_train)
    linear_predictions = model.predict(x_test_scaled)

    linear_metrics = {
        "model": "LinearRegression",
        "r2_test": r2_score(y_test, linear_predictions),
        "rmse_test": np.sqrt(mean_squared_error(y_test, linear_predictions)),
    }

    coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "standardized_coefficient": model.coef_,
        }
    ).sort_values(by="standardized_coefficient", key=lambda series: series.abs(), ascending=False)
    coefficients.to_csv(
        output_dir / "regression_standardized_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    random_forest = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=3,
    )
    random_forest.fit(x_train, y_train)
    rf_predictions = random_forest.predict(x_test)

    rf_metrics = {
        "model": "RandomForestRegressor",
        "r2_test": r2_score(y_test, rf_predictions),
        "rmse_test": np.sqrt(mean_squared_error(y_test, rf_predictions)),
    }

    rf_importances = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": random_forest.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)
    rf_importances.to_csv(
        output_dir / "random_forest_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = pd.DataFrame([linear_metrics, rf_metrics]).sort_values(
        by="r2_test", ascending=False
    )
    metrics.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    return metrics, coefficients, rf_importances


def save_plots(df, corr_matrix, output_dir):
    sns.set_theme(style="whitegrid")

    formatted_labels = [prettify_label(label) for label in corr_matrix.columns]
    heatmap_size = max(12, int(len(corr_matrix.columns) * 0.7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(heatmap_size, heatmap_size))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        square=False,
        linewidths=0.3,
        xticklabels=formatted_labels,
        yticklabels=formatted_labels,
        cbar_kws={"shrink": 0.8, "label": PEARSON_CORRELATION_LABEL},
    )
    apply_standard_plot_style("Полная корреляционная матрица", "Признаки", "Признаки")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=200)
    plt.close()

    top_score_features = (
        corr_matrix[TARGET_COLUMN]
        .drop(labels=[TARGET_COLUMN])
        .sort_values(key=lambda series: series.abs(), ascending=False)
        .head(TOP_HEATMAP_FEATURES)
    )
    top_feature_names = top_score_features.index.tolist() + [TARGET_COLUMN]
    top_corr_matrix = corr_matrix.loc[top_feature_names, top_feature_names]
    top_corr_matrix.to_csv(
        output_dir / "top_feature_correlation_matrix.csv",
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        top_corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        square=True,
        xticklabels=[prettify_label(label) for label in top_corr_matrix.columns],
        yticklabels=[prettify_label(label) for label in top_corr_matrix.index],
        cbar_kws={"shrink": 0.8, "label": PEARSON_CORRELATION_LABEL},
    )
    apply_standard_plot_style("Корреляции ключевых признаков с ATS-оценкой", "Признаки", "Признаки")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap_top_features.png", dpi=220)
    plt.close()

    score_heatmap_height = max(6, int(len(top_score_features) * 0.85))
    plt.figure(figsize=(9, score_heatmap_height))
    sns.heatmap(
        top_score_features.to_frame(name=TARGET_COLUMN),
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": PEARSON_CORRELATION_LABEL},
        xticklabels=[ATS_SCORE_LABEL],
        yticklabels=[prettify_label(label) for label in top_score_features.index],
    )
    apply_standard_plot_style("Корреляции признаков с ATS-оценкой", PEARSON_CORRELATION_LABEL, "Признаки")
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "score_correlation_heatmap.png", dpi=220)
    plt.close()

    for feature in ["jaccard_similarity", "cosine_similarity"]:
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=df.sample(min(len(df), 5000), random_state=42),
            x=feature,
            y=TARGET_COLUMN,
            scatter_kws={"alpha": 0.25, "s": 18},
            line_kws={"color": "darkred"},
        )
        apply_standard_plot_style(f"{feature_label(feature)} и ATS-оценка", feature_label(feature), ATS_SCORE_LABEL)
        plt.tight_layout()
        plt.savefig(output_dir / f"{feature}_vs_score.png", dpi=200)
        plt.close()


def write_summary(
    feature_columns,
    descriptive,
    correlations,
    group_test,
    anova_summary,
    anova_f_stat,
    anova_p_value,
    ols_result,
    model_metrics,
    regression_coefficients,
    rf_importances,
    output_dir,
):
    strongest_corr = correlations.iloc[0]
    group_row = group_test.iloc[0]
    best_model_row = model_metrics.iloc[0]
    strongest_positive = correlations.sort_values(by="pearson_r", ascending=False).iloc[0]
    strongest_negative = correlations.sort_values(by="pearson_r", ascending=True).iloc[0]

    lines = [
        "Статистический анализ связи текстовых признаков с ATS-оценкой",
        "",
        f"Количество наблюдений: {int(descriptive.loc['score', 'count'])}",
        f"Количество использованных признаков: {len(feature_columns)}",
        f"Средняя ATS-оценка: {descriptive.loc['score', 'mean']:.3f}",
        f"Стандартное отклонение ATS-оценки: {descriptive.loc['score', 'std']:.3f}",
        "",
        "Корреляционный анализ:",
        (
            f"Наиболее сильная линейная связь с ATS-оценкой у признака "
            f"{feature_label(strongest_corr['feature'])} (Pearson r={strongest_corr['pearson_r']:.4f}, "
            f"p={strongest_corr['pearson_p_value']:.4g})."
        ),
        (
            f"Самая сильная положительная связь: {feature_label(strongest_positive['feature'])} "
            f"(r={strongest_positive['pearson_r']:.4f})."
        ),
        (
            f"Самая сильная отрицательная связь: {feature_label(strongest_negative['feature'])} "
            f"(r={strongest_negative['pearson_r']:.4f})."
        ),
        "",
        "Проверка гипотезы по Jaccard:",
        (
            f"При пороге {group_row['threshold']:.4f} средняя ATS-оценка в группе высокой схожести "
            f"равен {group_row['high_group_mean_score']:.3f}, а в группе низкой схожести "
            f"{group_row['low_group_mean_score']:.3f}."
        ),
        (
            f"Разница средних = {group_row['mean_difference']:.3f}; "
            f"Welch t-test p={group_row['welch_t_p_value']:.4g}; "
            f"Mann-Whitney p={group_row['mannwhitney_u_p_value']:.4g}."
        ),
        "",
        "ANOVA по квартилям cosine similarity:",
        f"F={anova_f_stat:.4f}, p={anova_p_value:.4g}.",
        "",
        "Сравнение моделей:",
        (
            f"Лучшая модель: {best_model_row['model']} "
            f"(R^2={best_model_row['r2_test']:.4f}, RMSE={best_model_row['rmse_test']:.4f})."
        ),
        "Наиболее влиятельные стандартизованные коэффициенты линейной модели:",
    ]

    for _, row in regression_coefficients.head(4).iterrows():
        lines.append(
            f"- {row['feature']}: {row['standardized_coefficient']:.4f}"
            .replace(str(row['feature']), feature_label(row['feature']))
        )

    lines.extend(
        [
            "",
            "Наиболее важные признаки по Random Forest:",
        ]
    )

    for _, row in rf_importances.head(5).iterrows():
        lines.append(f"- {feature_label(row['feature'])}: {row['importance']:.4f}")

    lines.extend(
        [
            "",
            "Коэффициенты OLS и p-value см. в файле ols_coefficients.csv.",
            "Подробные таблицы и графики сохранены в папке analysis_outputs.",
        ]
    )

    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Чтение данных из {input_path}...")
    df, feature_columns = load_data(input_path)
    print(f"Загружено строк: {len(df)}")
    print(f"Числовых признаков для анализа: {len(feature_columns)}")

    print("Сохранение описательной статистики...")
    descriptive = save_descriptive_stats(df, output_dir)

    print("Расчет корреляций...")
    correlations, corr_matrix = save_correlations(df, feature_columns, output_dir)

    print("Проверка гипотезы по группам Jaccard...")
    group_test = run_group_test(df, output_dir, args.jaccard_threshold)

    print("Выполнение ANOVA по квартилям cosine similarity...")
    anova_summary, anova_f_stat, anova_p_value = run_anova(df, output_dir)

    print("Оценка OLS-регрессии с p-value...")
    ols_result = fit_ols_with_pvalues(df, feature_columns, output_dir)

    print("Оценка предсказательных моделей...")
    model_metrics, regression_coefficients, rf_importances = evaluate_predictive_models(
        df, feature_columns, output_dir
    )

    print("Сохранение графиков...")
    save_plots(df, corr_matrix, output_dir)

    print("Формирование текстового summary...")
    write_summary(
        feature_columns,
        descriptive,
        correlations,
        group_test,
        anova_summary,
        anova_f_stat,
        anova_p_value,
        ols_result,
        model_metrics,
        regression_coefficients,
        rf_importances,
        output_dir,
    )

    print("Готово. Результаты сохранены в папке analysis_outputs.")
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
