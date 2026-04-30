from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from feature_labels import feature_label
from ml_modeling import build_models, load_data, TARGET_COLUMN, RANDOM_STATE
from phrase_and_sentiment_analysis import extract_vacancy_group, normalize_text_series

from plot_style import (
    ABSOLUTE_ERROR_LABEL,
    ATS_SCORE_LABEL,
    PREDICTED_SCORE_LABEL,
    apply_standard_plot_style,
)


DEFAULT_DATASET_NAME = "VaishnaviGude/ats-resume-dataset-1lakh"
DEFAULT_MODEL_NAME = "RandomForestRegressor"
DEFAULT_OUTPUT_DIR = "model_error_outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Анализ ошибок модели прогнозирования ATS-оценки."
    )
    parser.add_argument("--input", default="features.csv", help="Путь к CSV с признаками.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME, help="Имя датасета Hugging Face для загрузки исходных текстов.")
    parser.add_argument("--split", default="train", help="Сплит датасета.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Папка для результатов.")
    parser.add_argument("--sample-rows", type=int, default=0, help="Если > 0, использовать только первые N строк.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Имя модели из ml_modeling.build_models().")
    parser.add_argument("--min-group-size", type=int, default=200, help="Минимальный размер группы вакансий на тесте.")
    return parser.parse_args()


def load_raw_dataset(dataset_name, split_name, expected_rows):
    raw_df = load_dataset(dataset_name, split=split_name).to_pandas()
    raw_df = raw_df.head(expected_rows).copy()
    raw_df["vacancy_group"] = normalize_text_series(raw_df["jd_text"]).apply(extract_vacancy_group)
    return raw_df.reset_index(drop=True)


def prepare_prediction_frame(args):
    features_df, feature_columns = load_data(args.input, sample_rows=args.sample_rows)
    raw_df = load_raw_dataset(args.dataset, args.split, len(features_df))

    models = build_models()
    if args.model not in models:
        raise ValueError(f"Неизвестная модель {args.model}. Доступно: {sorted(models)}")

    model = clone(models[args.model])
    x = features_df[feature_columns]
    y = features_df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x,
        y,
        features_df.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    residuals = y_test.to_numpy() - predictions
    prediction_df = features_df.loc[idx_test].copy().reset_index(drop=True)
    prediction_df["actual_score"] = y_test.to_numpy()
    prediction_df["predicted_score"] = predictions
    prediction_df["residual"] = residuals
    prediction_df["absolute_error"] = np.abs(residuals)

    raw_test = raw_df.loc[idx_test].reset_index(drop=True)
    merged = pd.concat(
        [
            prediction_df,
            raw_test[["resume_text", "jd_text", "vacancy_group"]].copy(),
        ],
        axis=1,
    )

    metrics = pd.DataFrame(
        [
            {
                "model": args.model,
                "r2_test": r2_score(y_test, predictions),
                "rmse_test": np.sqrt(mean_squared_error(y_test, predictions)),
                "mae_test": np.abs(residuals).mean(),
            }
        ]
    )
    return merged, metrics


def error_by_group(df, min_group_size):
    grouped = (
        df.groupby("vacancy_group")
        .agg(
            count=("absolute_error", "size"),
            mean_absolute_error=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
            rmse=("residual", lambda series: np.sqrt(np.mean(np.square(series)))),
            mean_actual_score=("actual_score", "mean"),
            mean_predicted_score=("predicted_score", "mean"),
        )
        .reset_index()
    )
    return grouped[grouped["count"] >= min_group_size].sort_values(
        by="mean_absolute_error", ascending=False
    )


def error_by_quantiles(df, feature_name, q=4):
    quantile_df = df.copy()
    quantile_df[f"{feature_name}_quartile"] = pd.qcut(
        quantile_df[feature_name],
        q=q,
        labels=[f"Q{i}" for i in range(1, q + 1)],
        duplicates="drop",
    )
    return (
        quantile_df.groupby(f"{feature_name}_quartile", observed=False)
        .agg(
            count=("absolute_error", "size"),
            mean_absolute_error=("absolute_error", "mean"),
            rmse=("residual", lambda series: np.sqrt(np.mean(np.square(series)))),
            mean_actual_score=("actual_score", "mean"),
        )
        .reset_index()
    )


def error_by_score_quantile(df):
    quantile_df = df.copy()
    quantile_df["actual_score_quartile"] = pd.qcut(
        quantile_df["actual_score"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )
    return (
        quantile_df.groupby("actual_score_quartile", observed=False)
        .agg(
            count=("absolute_error", "size"),
            mean_absolute_error=("absolute_error", "mean"),
            rmse=("residual", lambda series: np.sqrt(np.mean(np.square(series)))),
            mean_predicted_score=("predicted_score", "mean"),
        )
        .reset_index()
    )


def save_plots(df, group_errors, cosine_errors, sentence_errors, output_dir):
    sns.set_theme(style="whitegrid")

    sampled = df.sample(min(len(df), 5000), random_state=RANDOM_STATE)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=sampled, x="predicted_score", y="absolute_error", alpha=0.3, s=20)
    apply_standard_plot_style("Абсолютная ошибка и предсказанная ATS-оценка", PREDICTED_SCORE_LABEL, ABSOLUTE_ERROR_LABEL)
    plt.tight_layout()
    plt.savefig(output_dir / "absolute_error_vs_predicted.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.histplot(df["absolute_error"], bins=40, kde=True)
    apply_standard_plot_style("Распределение абсолютной ошибки", ABSOLUTE_ERROR_LABEL, "Частота")
    plt.tight_layout()
    plt.savefig(output_dir / "absolute_error_distribution.png", dpi=220)
    plt.close()

    if not group_errors.empty:
        plot_groups = group_errors.head(12).iloc[::-1]
        plt.figure(figsize=(10, 8))
        sns.barplot(data=plot_groups, x="mean_absolute_error", y="vacancy_group", orient="h", color="#E45756")
        apply_standard_plot_style("Группы вакансий с наибольшей средней абсолютной ошибкой", ABSOLUTE_ERROR_LABEL, "Группа вакансий")
        plt.tight_layout()
        plt.savefig(output_dir / "worst_vacancy_groups.png", dpi=220)
        plt.close()

    for table, title, filename, column in [
        (cosine_errors, "Ошибка по квартилям косинусного сходства", "error_by_cosine_quartile.png", "cosine_similarity_quartile"),
        (sentence_errors, "Ошибка по квартилям средней длины предложения в резюме", "error_by_sentence_length_quartile.png", "resume_avg_sentence_length_quartile"),
    ]:
        if not table.empty:
            plt.figure(figsize=(8, 6))
            sns.barplot(data=table, x=column, y="mean_absolute_error", color="#4C78A8")
            apply_standard_plot_style(title, "Квартиль", ABSOLUTE_ERROR_LABEL)
            plt.tight_layout()
            plt.savefig(output_dir / filename, dpi=220)
            plt.close()


def write_summary(metrics, group_errors, score_errors, cosine_errors, sentence_errors, output_dir):
    metric_row = metrics.iloc[0]
    lines = [
        "Анализ ошибок модели",
        "",
        f"Модель: {metric_row['model']}",
        f"R^2={metric_row['r2_test']:.4f}, RMSE={metric_row['rmse_test']:.4f}, MAE={metric_row['mae_test']:.4f}",
        "",
        "Ошибки по квартилям фактической ATS-оценки:",
    ]

    for _, row in score_errors.iterrows():
        lines.append(
            f"- {row['actual_score_quartile']}: MAE={row['mean_absolute_error']:.4f}, RMSE={row['rmse']:.4f}"
        )

    if not group_errors.empty:
        lines.extend(["", "Группы вакансий с наибольшей ошибкой:"])
        for _, row in group_errors.head(8).iterrows():
            lines.append(
                f"- {row['vacancy_group']}: MAE={row['mean_absolute_error']:.4f}, count={int(row['count'])}"
            )

    lines.extend(["", "Ошибки по квартилям cosine similarity:"])
    for _, row in cosine_errors.iterrows():
        lines.append(
            f"- {row['cosine_similarity_quartile']}: MAE={row['mean_absolute_error']:.4f}"
        )

    lines.extend(["", "Ошибки по квартилям средней длины предложения в резюме:"])
    for _, row in sentence_errors.iterrows():
        lines.append(
            f"- {row['resume_avg_sentence_length_quartile']}: MAE={row['mean_absolute_error']:.4f}"
        )

    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Подготовка предсказаний для диагностики ошибок...")
    prediction_df, metrics = prepare_prediction_frame(args)
    prediction_df.to_csv(output_dir / "prediction_diagnostics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "model_error_metrics.csv", index=False, encoding="utf-8-sig")

    print("Агрегация ошибок по группам вакансий...")
    group_errors = error_by_group(prediction_df, args.min_group_size)
    group_errors.to_csv(output_dir / "error_by_vacancy_group.csv", index=False, encoding="utf-8-sig")

    print("Агрегация ошибок по квартилям ATS-оценки и признаков...")
    score_errors = error_by_score_quantile(prediction_df)
    score_errors.to_csv(output_dir / "error_by_score_quartile.csv", index=False, encoding="utf-8-sig")

    cosine_errors = error_by_quantiles(prediction_df, "cosine_similarity")
    cosine_errors.to_csv(output_dir / "error_by_cosine_similarity_quartile.csv", index=False, encoding="utf-8-sig")

    sentence_errors = error_by_quantiles(prediction_df, "resume_avg_sentence_length")
    sentence_errors.to_csv(output_dir / "error_by_sentence_length_quartile.csv", index=False, encoding="utf-8-sig")

    print("Сохранение графиков...")
    save_plots(prediction_df, group_errors, cosine_errors, sentence_errors, output_dir)

    print("Формирование summary...")
    write_summary(metrics, group_errors, score_errors, cosine_errors, sentence_errors, output_dir)

    print(f"Готово. Результаты сохранены в {output_dir}.")
    print(group_errors.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
