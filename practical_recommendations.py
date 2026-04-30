from pathlib import Path
import argparse

import pandas as pd
from feature_labels import feature_group_label, feature_label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Формирование практических выводов по проекту ATS-оценки."
    )
    parser.add_argument("--output", default="practical_recommendations.txt", help="Куда сохранить рекомендации.")
    parser.add_argument("--analysis-dir", default="analysis_outputs", help="Папка со статистическим анализом.")
    parser.add_argument("--phrase-dir", default="phrase_analysis_outputs", help="Папка с анализом словосочетаний и тональности.")
    parser.add_argument("--ablation-dir", default="ablation_outputs", help="Папка с абляционным анализом.")
    parser.add_argument("--multicollinearity-dir", default="multicollinearity_outputs", help="Папка с мультиколлинеарностью.")
    parser.add_argument("--error-dir", default="model_error_outputs", help="Папка с анализом ошибок модели.")
    return parser.parse_args()


def safe_read_csv(path):
    file_path = Path(path)
    if file_path.exists():
        return pd.read_csv(file_path)
    return pd.DataFrame()


def build_recommendations(args):
    correlations = safe_read_csv(Path(args.analysis_dir) / "correlations_with_score.csv")
    rf_importance = safe_read_csv(Path(args.analysis_dir) / "random_forest_feature_importance.csv")
    sentiment_corr = safe_read_csv(Path(args.phrase_dir) / "sentiment_correlations.csv")
    resume_ngrams = safe_read_csv(Path(args.phrase_dir) / "overall_resume_ngrams.csv")
    ablation = safe_read_csv(Path(args.ablation_dir) / "ablation_results.csv")
    high_corr_pairs = safe_read_csv(Path(args.multicollinearity_dir) / "high_correlation_pairs.csv")
    error_groups = safe_read_csv(Path(args.error_dir) / "error_by_vacancy_group.csv")

    lines = [
        "Практические выводы и рекомендации",
        "",
        "1. Что чаще связано с более высокой ATS-оценкой",
    ]

    if not correlations.empty:
        top_positive = correlations.sort_values(by="pearson_r", ascending=False).head(5)
        for _, row in top_positive.iterrows():
            lines.append(
                f"- Усиление признака «{feature_label(row['feature'])}» связано с ростом ATS-оценки (коэффициент Пирсона = {row['pearson_r']:.4f})."
            )

    lines.extend(["", "2. Что чаще связано с более низкой ATS-оценкой"])
    if not correlations.empty:
        top_negative = correlations.sort_values(by="pearson_r", ascending=True).head(5)
        for _, row in top_negative.iterrows():
            lines.append(
                f"- Рост признака «{feature_label(row['feature'])}» связан со снижением ATS-оценки (коэффициент Пирсона = {row['pearson_r']:.4f})."
            )

    lines.extend(["", "3. Наиболее важные признаки по нелинейной модели"])
    if not rf_importance.empty:
        for _, row in rf_importance.head(8).iterrows():
            lines.append(f"- {feature_label(row['feature'])}: importance={row['importance']:.4f}.")

    lines.extend(["", "4. Полезные и рискованные словосочетания"])
    if not resume_ngrams.empty:
        positive_phrases = resume_ngrams[resume_ngrams["association"] == "higher_score"].head(8)
        negative_phrases = resume_ngrams[resume_ngrams["association"] == "lower_score"].head(8)
        lines.append("Полезные формулировки в резюме:")
        for _, row in positive_phrases.iterrows():
            lines.append(f"- {row['ngram']}")
        lines.append("Формулировки, чаще встречающиеся у низкой ATS-оценки:")
        for _, row in negative_phrases.iterrows():
            lines.append(f"- {row['ngram']}")

    lines.extend(["", "5. Тональность"])
    if not sentiment_corr.empty:
        for _, row in sentiment_corr.head(4).iterrows():
            direction = "положительно" if row["pearson_r"] > 0 else "отрицательно"
            lines.append(
                f"- Признак «{feature_label(row['feature'])}» {direction} связан с ATS-оценкой (коэффициент Пирсона = {row['pearson_r']:.4f})."
            )

    lines.extend(["", "6. Что показал абляционный анализ"])
    if not ablation.empty:
        for model_name in ablation["model"].unique():
            model_df = ablation[ablation["model"] == model_name]
            baseline = model_df[model_df["scenario"] == "all_features"]
            without = model_df[model_df["scenario"] == "without_group"].copy()
            if not baseline.empty and not without.empty:
                baseline_r2 = baseline.iloc[0]["cv_r2_mean"]
                without["r2_drop"] = baseline_r2 - without["cv_r2_mean"]
                strongest_drop = without.sort_values("r2_drop", ascending=False).iloc[0]
                lines.append(
                    f"- Для модели {model_name} сильнее всего просаживает качество удаление группы «{feature_group_label(strongest_drop['feature_group'])}» (ΔR^2={strongest_drop['r2_drop']:.4f})."
                )

    lines.extend(["", "7. Осторожно с мультиколлинеарностью"])
    if not high_corr_pairs.empty:
        for _, row in high_corr_pairs.head(6).iterrows():
            lines.append(
                f"- Пара «{feature_label(row['feature_1'])}» / «{feature_label(row['feature_2'])}» имеет высокую корреляцию r={row['correlation']:.4f}; "
                f"при упрощении модели первым кандидатом на удаление является «{feature_label(row['recommended_drop'])}»."
            )

    lines.extend(["", "8. Где модель ошибается сильнее"])
    if not error_groups.empty:
        for _, row in error_groups.head(6).iterrows():
            lines.append(
                f"- В группе вакансий {row['vacancy_group']} средняя абсолютная ошибка равна {row['mean_absolute_error']:.4f}."
            )

    lines.extend(
        [
            "",
            "9. Практические рекомендации по улучшению резюме",
            "- Делать текст резюме структурированным и не перегружать его слишком длинными предложениями.",
            "- Явно отражать релевантные навыки и совпадения с вакансией, но не ограничиваться механическим повторением слов из описания.",
            "- Использовать конкретные достижения, сертификаты и результаты, а не только общие формулировки.",
            "- Следить за стилем изложения: позитивная, уверенная и содержательная подача чаще связана с более высокой ATS-оценкой.",
            "- При адаптации резюме под вакансию ориентироваться не только на отдельные ключевые слова, но и на общий семантический контекст.",
        ]
    )

    return "\n".join(lines)


def main():
    args = parse_args()
    result_text = build_recommendations(args)
    output_path = Path(args.output)
    output_path.write_text(result_text, encoding="utf-8")
    print(f"Готово. Практические рекомендации сохранены в {output_path}.")


if __name__ == "__main__":
    main()
