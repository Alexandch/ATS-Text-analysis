from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_labels import feature_label
from plot_style import (
    ACTUAL_SCORE_LABEL,
    FEATURE_IMPORTANCE_LABEL,
    PREDICTED_SCORE_LABEL,
    RESIDUAL_LABEL,
    apply_standard_plot_style,
)


TARGET_COLUMN = "score"
RANDOM_STATE = 42
TOP_FEATURES_TO_SHOW = 12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Сравнение ML-моделей для предсказания ATS-оценки."
    )
    parser.add_argument(
        "--input",
        default="features.csv",
        help="Путь к CSV с признаками.",
    )
    parser.add_argument(
        "--output-dir",
        default="ml_outputs",
        help="Папка для результатов ML-моделирования.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="Если > 0, использовать только первые N строк для быстрых экспериментов.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Количество фолдов для кросс-валидации.",
    )
    return parser.parse_args()


def load_data(path, sample_rows=0):
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"В файле отсутствует колонка {TARGET_COLUMN}")

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [column for column in numeric_columns if column != TARGET_COLUMN]
    if not feature_columns:
        raise ValueError("Не найдены числовые признаки для ML-моделирования.")

    if sample_rows and sample_rows > 0:
        df = df.head(sample_rows).copy()

    return df[feature_columns + [TARGET_COLUMN]].dropna().copy(), feature_columns


def rmse_value(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def build_models():
    return {
        "LinearRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=3,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
        ),
    }


def run_cross_validation(x, y, models, cv_folds, output_dir):
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, model in models.items():
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
        rows.append(
            {
                "model": name,
                "cv_r2_mean": scores["test_r2"].mean(),
                "cv_r2_std": scores["test_r2"].std(),
                "cv_rmse_mean": rmse_scores.mean(),
                "cv_rmse_std": rmse_scores.std(),
            }
        )

    results = pd.DataFrame(rows).sort_values(by="cv_r2_mean", ascending=False)
    results.to_csv(output_dir / "cv_model_comparison.csv", index=False, encoding="utf-8-sig")
    return results


def fit_best_model(x, y, feature_columns, models, best_model_name, output_dir):
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    best_model = clone(models[best_model_name])
    best_model.fit(x_train, y_train)
    predictions = best_model.predict(x_test)
    residuals = y_test - predictions

    metrics = pd.DataFrame(
        [
            {
                "model": best_model_name,
                "r2_test": r2_score(y_test, predictions),
                "rmse_test": rmse_value(y_test, predictions),
                "mae_test": np.abs(residuals).mean(),
            }
        ]
    )
    metrics.to_csv(output_dir / "best_model_holdout_metrics.csv", index=False, encoding="utf-8-sig")

    prediction_df = pd.DataFrame(
        {
            "actual_score": y_test.to_numpy(),
            "predicted_score": predictions,
            "residual": residuals.to_numpy(),
        }
    )
    prediction_df.to_csv(output_dir / "best_model_predictions.csv", index=False, encoding="utf-8-sig")

    importance_df = extract_feature_importance(
        best_model,
        x_test,
        y_test,
        feature_columns,
    )
    importance_df.to_csv(
        output_dir / "best_model_feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return best_model, metrics, prediction_df, importance_df


def extract_feature_importance(model, x_test, y_test, feature_columns):
    final_model = model.named_steps["model"] if isinstance(model, Pipeline) else model

    if hasattr(final_model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": final_model.feature_importances_,
                "importance_type": "native_importance",
            }
        )
    elif hasattr(final_model, "coef_"):
        coef = np.ravel(final_model.coef_)
        importance_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": np.abs(coef),
                "signed_value": coef,
                "importance_type": "absolute_coefficient",
            }
        )
    else:
        result = permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=5,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        importance_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": result.importances_mean,
                "importance_std": result.importances_std,
                "importance_type": "permutation_importance",
            }
        )

    return importance_df.sort_values(by="importance", ascending=False)


def save_plots(prediction_df, importance_df, output_dir):
    sns.set_theme(style="whitegrid")

    sampled_predictions = prediction_df.sample(
        min(len(prediction_df), 5000),
        random_state=RANDOM_STATE,
    )

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=sampled_predictions,
        x="actual_score",
        y="predicted_score",
        alpha=0.35,
        s=25,
    )
    diagonal_min = min(sampled_predictions["actual_score"].min(), sampled_predictions["predicted_score"].min())
    diagonal_max = max(sampled_predictions["actual_score"].max(), sampled_predictions["predicted_score"].max())
    plt.plot([diagonal_min, diagonal_max], [diagonal_min, diagonal_max], color="darkred", linewidth=2)
    apply_standard_plot_style("Фактическая и предсказанная ATS-оценка", ACTUAL_SCORE_LABEL, PREDICTED_SCORE_LABEL)
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=sampled_predictions,
        x="predicted_score",
        y="residual",
        alpha=0.35,
        s=25,
    )
    plt.axhline(0, color="darkred", linewidth=2)
    apply_standard_plot_style("Остатки модели и предсказанная ATS-оценка", PREDICTED_SCORE_LABEL, RESIDUAL_LABEL)
    plt.tight_layout()
    plt.savefig(output_dir / "residuals_vs_predicted.png", dpi=220)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.histplot(prediction_df["residual"], bins=40, kde=True)
    apply_standard_plot_style("Распределение остатков модели", RESIDUAL_LABEL, "Частота")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_distribution.png", dpi=220)
    plt.close()

    top_importances = importance_df.head(TOP_FEATURES_TO_SHOW).iloc[::-1]
    plot_df = top_importances.copy()
    plot_df["feature"] = plot_df["feature"].map(feature_label)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=plot_df, x="importance", y="feature", orient="h", color="#4C78A8")
    apply_standard_plot_style("Наиболее важные признаки модели", FEATURE_IMPORTANCE_LABEL, "Признак")
    plt.tight_layout()
    plt.savefig(output_dir / "top_feature_importances.png", dpi=220)
    plt.close()


def write_summary(df, feature_columns, cv_results, holdout_metrics, importance_df, output_dir):
    best_cv_row = cv_results.iloc[0]
    holdout_row = holdout_metrics.iloc[0]

    lines = [
        "ML-моделирование ATS-оценки",
        "",
        f"Количество наблюдений: {len(df)}",
        f"Количество числовых признаков: {len(feature_columns)}",
        "",
        "Лучшая модель по кросс-валидации:",
        (
            f"{best_cv_row['model']} "
            f"(CV R^2={best_cv_row['cv_r2_mean']:.4f} +/- {best_cv_row['cv_r2_std']:.4f}, "
            f"CV RMSE={best_cv_row['cv_rmse_mean']:.4f} +/- {best_cv_row['cv_rmse_std']:.4f})."
        ),
        "",
        "Качество лучшей модели на holdout-выборке:",
        (
            f"R^2={holdout_row['r2_test']:.4f}, "
            f"RMSE={holdout_row['rmse_test']:.4f}, "
            f"MAE={holdout_row['mae_test']:.4f}."
        ),
        "",
        "Наиболее важные признаки:",
    ]

    for _, row in importance_df.head(8).iterrows():
        lines.append(f"- {feature_label(row['feature'])}: {row['importance']:.4f}")

    lines.extend(
        [
            "",
            "Подробные результаты см. в папке ml_outputs.",
        ]
    )

    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Чтение данных из {input_path}...")
    df, feature_columns = load_data(input_path, sample_rows=args.sample_rows)
    x = df[feature_columns]
    y = df[TARGET_COLUMN]

    print("Построение набора моделей...")
    models = build_models()

    print("Кросс-валидация моделей...")
    cv_results = run_cross_validation(x, y, models, args.cv_folds, output_dir)

    best_model_name = cv_results.iloc[0]["model"]
    print(f"Обучение лучшей модели на holdout-выборке: {best_model_name}...")
    best_model, holdout_metrics, prediction_df, importance_df = fit_best_model(
        x,
        y,
        feature_columns,
        models,
        best_model_name,
        output_dir,
    )

    print("Сохранение графиков...")
    save_plots(prediction_df, importance_df, output_dir)

    print("Формирование summary...")
    write_summary(df, feature_columns, cv_results, holdout_metrics, importance_df, output_dir)

    print("Готово. Результаты сохранены в папке ml_outputs.")
    print(cv_results.to_string(index=False))


if __name__ == "__main__":
    main()
