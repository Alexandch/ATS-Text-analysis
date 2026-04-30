from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

from feature_labels import feature_group_label, scenario_label
from ml_modeling import build_models, load_data
from plot_style import CV_R2_LABEL, apply_standard_plot_style


TARGET_COLUMN = "score"
RANDOM_STATE = 42
DEFAULT_MODELS = ["Ridge", "RandomForestRegressor"]
FEATURE_GROUP_ORDER = [
    "structural",
    "lexical",
    "sentiment",
    "pair_alignment",
    "similarity",
    "other",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Абляционный анализ групп признаков для ATS-оценки."
    )
    parser.add_argument("--input", default="features.csv", help="Путь к CSV с признаками.")
    parser.add_argument("--output-dir", default="ablation_outputs", help="Папка для результатов.")
    parser.add_argument("--sample-rows", type=int, default=0, help="Если > 0, использовать только первые N строк.")
    parser.add_argument("--cv-folds", type=int, default=3, help="Количество фолдов для кросс-валидации.")
    return parser.parse_args()


def group_name(feature_name):
    if feature_name in {"jaccard_similarity", "cosine_similarity"}:
        return "similarity"

    if (
        feature_name.startswith("resume_")
        or feature_name.startswith("job_")
    ):
        if "sentiment" in feature_name or "positive_ratio" in feature_name or "negative_ratio" in feature_name:
            return "sentiment"
        if any(token in feature_name for token in ["word_count", "char_count", "sentence_count", "avg_sentence_length", "avg_word_length"]):
            return "structural"
        if any(token in feature_name for token in ["lexical_diversity", "numeric_ratio", "flesch_reading_ease"]):
            return "lexical"

    if (
        feature_name.startswith("overlap_")
        or feature_name.endswith("_difference")
        or feature_name.endswith("_gap")
        or feature_name.endswith("_ratio")
        or feature_name == "shared_token_count"
        or feature_name == "sentiment_difference"
        or feature_name == "sentiment_gap"
    ):
        return "pair_alignment"

    return "other"


def build_feature_groups(feature_columns):
    groups = {name: [] for name in FEATURE_GROUP_ORDER}
    for feature in feature_columns:
        groups[group_name(feature)].append(feature)
    return {name: features for name, features in groups.items() if features}


def evaluate_subset(model, x, y, cv_folds):
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        model,
        x,
        y,
        cv=cv,
        scoring=("r2", "neg_mean_squared_error"),
        n_jobs=1,
        return_train_score=False,
    )
    rmse_scores = np.sqrt(-scores["test_neg_mean_squared_error"])
    return scores["test_r2"].mean(), scores["test_r2"].std(), rmse_scores.mean(), rmse_scores.std()


def holdout_score(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    fitted = clone(model)
    fitted.fit(x_train, y_train)
    predictions = fitted.predict(x_test)
    return r2_score(y_test, predictions), np.sqrt(mean_squared_error(y_test, predictions))


def run_ablation(df, feature_columns, output_dir, cv_folds):
    feature_groups = build_feature_groups(feature_columns)
    x_full = df[feature_columns]
    y = df[TARGET_COLUMN]
    models = build_models()
    models = {name: model for name, model in models.items() if name in DEFAULT_MODELS}

    rows = []
    for model_name, model in models.items():
        cv_r2, cv_r2_std, cv_rmse, cv_rmse_std = evaluate_subset(model, x_full, y, cv_folds)
        holdout_r2, holdout_rmse = holdout_score(model, x_full, y)
        rows.append(
            {
                "model": model_name,
                "scenario": "all_features",
                "feature_group": "all",
                "feature_count": len(feature_columns),
                "cv_r2_mean": cv_r2,
                "cv_r2_std": cv_r2_std,
                "cv_rmse_mean": cv_rmse,
                "cv_rmse_std": cv_rmse_std,
                "holdout_r2": holdout_r2,
                "holdout_rmse": holdout_rmse,
            }
        )

        for group, group_features in feature_groups.items():
            only_features = group_features
            if only_features:
                x_only = df[only_features]
                cv_r2, cv_r2_std, cv_rmse, cv_rmse_std = evaluate_subset(model, x_only, y, cv_folds)
                holdout_r2, holdout_rmse = holdout_score(model, x_only, y)
                rows.append(
                    {
                        "model": model_name,
                        "scenario": "only_group",
                        "feature_group": group,
                        "feature_count": len(only_features),
                        "cv_r2_mean": cv_r2,
                        "cv_r2_std": cv_r2_std,
                        "cv_rmse_mean": cv_rmse,
                        "cv_rmse_std": cv_rmse_std,
                        "holdout_r2": holdout_r2,
                        "holdout_rmse": holdout_rmse,
                    }
                )

            remaining_features = [feature for feature in feature_columns if feature not in group_features]
            if remaining_features:
                x_without = df[remaining_features]
                cv_r2, cv_r2_std, cv_rmse, cv_rmse_std = evaluate_subset(model, x_without, y, cv_folds)
                holdout_r2, holdout_rmse = holdout_score(model, x_without, y)
                rows.append(
                    {
                        "model": model_name,
                        "scenario": "without_group",
                        "feature_group": group,
                        "feature_count": len(remaining_features),
                        "cv_r2_mean": cv_r2,
                        "cv_r2_std": cv_r2_std,
                        "cv_rmse_mean": cv_rmse,
                        "cv_rmse_std": cv_rmse_std,
                        "holdout_r2": holdout_r2,
                        "holdout_rmse": holdout_rmse,
                    }
                )

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "ablation_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [(group, len(features)) for group, features in feature_groups.items()],
        columns=["feature_group", "feature_count"],
    ).to_csv(output_dir / "feature_groups.csv", index=False, encoding="utf-8-sig")
    return result, feature_groups


def save_plots(results, output_dir):
    sns.set_theme(style="whitegrid")

    for model_name in results["model"].unique():
        model_df = results[results["model"] == model_name].copy()
        plot_df = model_df[model_df["scenario"] != "all_features"].copy()
        if plot_df.empty:
            continue

        plot_df["label"] = plot_df["scenario"].map(scenario_label) + " | " + plot_df["feature_group"].map(feature_group_label)
        plt.figure(figsize=(12, 8))
        sns.barplot(data=plot_df, x="cv_r2_mean", y="label", orient="h", color="#4C78A8")
        baseline = model_df.loc[model_df["scenario"] == "all_features", "cv_r2_mean"].iloc[0]
        plt.axvline(baseline, color="darkred", linewidth=2, linestyle="--")
        apply_standard_plot_style(
            f"Абляционный анализ признаков для модели {model_name}",
            CV_R2_LABEL,
            "Сценарий и группа признаков",
        )
        plt.tight_layout()
        plt.savefig(output_dir / f"ablation_{model_name}.png", dpi=220)
        plt.close()


def write_summary(results, feature_groups, output_dir):
    lines = [
        "Абляционный анализ признаков",
        "",
    ]

    for model_name in results["model"].unique():
        model_df = results[results["model"] == model_name].copy()
        baseline = model_df[model_df["scenario"] == "all_features"].iloc[0]
        lines.append(
            f"Модель {model_name}: baseline CV R^2={baseline['cv_r2_mean']:.4f}, holdout R^2={baseline['holdout_r2']:.4f}"
        )

        without_rows = model_df[model_df["scenario"] == "without_group"].copy()
        without_rows["r2_drop"] = baseline["cv_r2_mean"] - without_rows["cv_r2_mean"]
        strongest_drop = without_rows.sort_values("r2_drop", ascending=False).iloc[0]

        only_rows = model_df[model_df["scenario"] == "only_group"].copy()
        strongest_only = only_rows.sort_values("cv_r2_mean", ascending=False).iloc[0]

        lines.append(
            f"- Самый важный блок по падению качества при удалении: {feature_group_label(strongest_drop['feature_group'])} (ΔR^2={strongest_drop['r2_drop']:.4f})"
        )
        lines.append(
            f"- Самый сильный блок сам по себе: {feature_group_label(strongest_only['feature_group'])} (CV R^2={strongest_only['cv_r2_mean']:.4f})"
        )
        lines.append("")

    lines.append("Состав групп признаков:")
    for group, features in feature_groups.items():
        lines.append(f"- {feature_group_label(group)}: {len(features)} признаков")

    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Чтение данных из {args.input}...")
    df, feature_columns = load_data(args.input, sample_rows=args.sample_rows)
    print(f"Загружено строк: {len(df)}; признаков: {len(feature_columns)}")

    print("Выполнение абляционного анализа...")
    results, feature_groups = run_ablation(df, feature_columns, output_dir, args.cv_folds)

    print("Сохранение графиков...")
    save_plots(results, output_dir)

    print("Формирование summary...")
    write_summary(results, feature_groups, output_dir)

    print(f"Готово. Результаты сохранены в {output_dir}.")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
