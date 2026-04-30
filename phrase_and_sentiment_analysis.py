from pathlib import Path
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

from feature_labels import feature_label
from main import calculate_sentiment_features, extract_alpha_tokens
from plot_style import (
    ATS_SCORE_LABEL,
    PEARSON_CORRELATION_LABEL,
    PRESENCE_RATE_DIFF_LABEL,
    NGRAM_LABEL,
    VACANCY_GROUP_LABEL,
    apply_standard_plot_style,
)


DEFAULT_DATASET_NAME = "VaishnaviGude/ats-resume-dataset-1lakh"
DEFAULT_OUTPUT_DIR = "phrase_analysis_outputs"
TARGET_COLUMN = "score"
ROLE_PATTERNS = [
    re.compile(r"(?:job title|title|position|role)\s*:\s*([^\n,.;]+)", re.IGNORECASE),
    re.compile(r"(?:hiring|looking for|seeking|need(?:ing)?)\s+(?:an?|the)?\s*([a-zA-Z][a-zA-Z\s/-]{3,60})", re.IGNORECASE),
]
ROLE_STOP_WORDS = {
    "a", "an", "and", "for", "the", "with", "to", "of", "in", "on", "at",
    "we", "are", "is", "our", "join", "team", "looking", "hiring", "seeking",
}
PLOT_SAMPLE_SIZE = 5000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Анализ словосочетаний и тональности для ATS-оценки."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME, help="Имя датасета Hugging Face.")
    parser.add_argument("--split", default="train", help="Сплит датасета.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Папка для результатов.")
    parser.add_argument("--sample-rows", type=int, default=0, help="Если > 0, анализировать только первые N строк.")
    parser.add_argument("--ngram-min", type=int, default=2, help="Минимальная длина n-граммы.")
    parser.add_argument("--ngram-max", type=int, default=3, help="Максимальная длина n-граммы.")
    parser.add_argument("--min-df", type=int, default=30, help="Минимальная частота фразы в документах.")
    parser.add_argument("--max-features", type=int, default=5000, help="Максимум n-грамм в анализе.")
    parser.add_argument("--top-k", type=int, default=20, help="Сколько топ-фраз сохранять в summary.")
    parser.add_argument("--min-group-size", type=int, default=500, help="Минимальный размер группы вакансий.")
    parser.add_argument("--max-groups", type=int, default=10, help="Максимум групп вакансий для отдельного анализа.")
    return parser.parse_args()


def normalize_text_series(series):
    return series.fillna("").astype(str)


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def load_raw_data(dataset_name, split_name, sample_rows):
    dataset = load_dataset(dataset_name, split=split_name)
    df = dataset.to_pandas()
    if sample_rows > 0:
        df = df.head(sample_rows).copy()
    required_columns = {"resume_text", "jd_text", TARGET_COLUMN}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"В датасете отсутствуют обязательные колонки: {sorted(missing)}")
    return df


def normalize_role_label(label):
    tokens = [token for token in extract_alpha_tokens(label) if token not in ROLE_STOP_WORDS]
    if not tokens:
        return "other"
    return " ".join(tokens[:4])


def extract_vacancy_group(job_text):
    if not isinstance(job_text, str) or not job_text.strip():
        return "other"

    for pattern in ROLE_PATTERNS:
        match = pattern.search(job_text)
        if match:
            return normalize_role_label(match.group(1))

    first_nonempty_line = next(
        (line.strip() for line in job_text.splitlines() if line.strip()),
        "",
    )
    if first_nonempty_line:
        return normalize_role_label(first_nonempty_line)
    return "other"


def add_sentiment_columns(df):
    def score_text(text):
        features = calculate_sentiment_features(extract_alpha_tokens(text))
        return pd.Series(features)

    resume_sentiment = normalize_text_series(df["resume_text"]).apply(score_text).add_prefix("resume_")
    job_sentiment = normalize_text_series(df["jd_text"]).apply(score_text).add_prefix("job_")
    result = pd.concat([df, resume_sentiment, job_sentiment], axis=1)
    result["sentiment_difference"] = (
        result["resume_sentiment_score"] - result["job_sentiment_score"]
    )
    result["sentiment_gap"] = result["sentiment_difference"].abs()
    return result


def analyze_sentiment(df, output_dir):
    sentiment_columns = [
        "resume_sentiment_score",
        "job_sentiment_score",
        "resume_positive_ratio",
        "resume_negative_ratio",
        "job_positive_ratio",
        "job_negative_ratio",
        "sentiment_difference",
        "sentiment_gap",
    ]
    rows = []
    for column in sentiment_columns:
        pearson_r, pearson_p = stats.pearsonr(df[column], df[TARGET_COLUMN])
        spearman_rho, spearman_p = stats.spearmanr(df[column], df[TARGET_COLUMN])
        rows.append(
            {
                "feature": column,
                "pearson_r": pearson_r,
                "pearson_p_value": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p_value": spearman_p,
            }
        )

    sentiment_df = pd.DataFrame(rows).sort_values(
        by="pearson_r", key=lambda series: series.abs(), ascending=False
    )
    sentiment_df.to_csv(output_dir / "sentiment_correlations.csv", index=False, encoding="utf-8-sig")
    return sentiment_df


def prettify_label(label):
    return feature_label(label)


def prepare_vectorizer(args):
    return CountVectorizer(
        stop_words="english",
        binary=True,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=args.min_df,
        max_features=args.max_features,
    )


def analyze_ngram_frame(texts, scores, args):
    vectorizer = prepare_vectorizer(args)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame()
    if matrix.shape[1] == 0:
        return pd.DataFrame()

    high_score = (scores >= scores.median()).astype(int)
    chi2_scores, p_values = chi2(matrix, high_score)

    high_mask = high_score.to_numpy(dtype=bool)
    low_mask = ~high_mask
    high_rates = matrix[high_mask].mean(axis=0).A1
    low_rates = matrix[low_mask].mean(axis=0).A1
    counts = matrix.sum(axis=0).A1
    mean_scores_when_present = matrix.T.dot(scores.to_numpy()) / np.maximum(counts, 1)

    result = pd.DataFrame(
        {
            "ngram": vectorizer.get_feature_names_out(),
            "document_count": counts,
            "high_score_presence_rate": high_rates,
            "low_score_presence_rate": low_rates,
            "presence_rate_difference": high_rates - low_rates,
            "chi2_score": chi2_scores,
            "chi2_p_value": p_values,
            "mean_score_when_present": mean_scores_when_present,
        }
    )
    result["association"] = np.where(
        result["presence_rate_difference"] >= 0,
        "higher_score",
        "lower_score",
    )
    return result.sort_values(
        by=["chi2_score", "document_count"],
        ascending=[False, False],
    )


def save_overall_ngram_analysis(df, text_column, output_name, args, output_dir):
    result = analyze_ngram_frame(normalize_text_series(df[text_column]), df[TARGET_COLUMN], args)
    result.to_csv(output_dir / output_name, index=False, encoding="utf-8-sig")
    return result


def save_group_ngram_analysis(df, args, output_dir):
    group_counts = df["vacancy_group"].value_counts()
    eligible_groups = group_counts[group_counts >= args.min_group_size].head(args.max_groups).index.tolist()
    all_group_rows = []

    for group_name in eligible_groups:
        group_df = df[df["vacancy_group"] == group_name].copy()
        result = analyze_ngram_frame(
            normalize_text_series(group_df["resume_text"]),
            group_df[TARGET_COLUMN],
            args,
        )
        if result.empty:
            continue
        top_positive = result[result["presence_rate_difference"] > 0].head(args.top_k).copy()
        top_negative = result[result["presence_rate_difference"] < 0].head(args.top_k).copy()
        grouped = pd.concat([top_positive, top_negative], ignore_index=True)
        grouped.insert(0, "vacancy_group", group_name)
        grouped.insert(1, "group_size", len(group_df))
        all_group_rows.append(grouped)

    if all_group_rows:
        combined = pd.concat(all_group_rows, ignore_index=True)
    else:
        combined = pd.DataFrame(
            columns=[
                "vacancy_group",
                "group_size",
                "ngram",
                "document_count",
                "high_score_presence_rate",
                "low_score_presence_rate",
                "presence_rate_difference",
                "chi2_score",
                "chi2_p_value",
                "mean_score_when_present",
                "association",
            ]
        )

    combined.to_csv(output_dir / "vacancy_group_resume_ngrams.csv", index=False, encoding="utf-8-sig")
    return combined, group_counts


def plot_sentiment_results(df, sentiment_df, output_dir):
    sns.set_theme(style="whitegrid")

    top_sentiment = sentiment_df.copy()
    top_sentiment["feature"] = top_sentiment["feature"].map(prettify_label)

    plt.figure(figsize=(9, 6))
    sns.barplot(
        data=top_sentiment,
        x="pearson_r",
        y="feature",
        orient="h",
        color="#4C78A8",
    )
    plt.axvline(0, color="black", linewidth=1)
    apply_standard_plot_style("Связь признаков тональности с ATS-оценкой", PEARSON_CORRELATION_LABEL, "Признак")
    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_correlations.png", dpi=220)
    plt.close()

    sampled_df = df.sample(min(len(df), PLOT_SAMPLE_SIZE), random_state=42)
    for feature in ["resume_sentiment_score", "job_sentiment_score", "sentiment_gap"]:
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=sampled_df,
            x=feature,
            y=TARGET_COLUMN,
            scatter_kws={"alpha": 0.25, "s": 18},
            line_kws={"color": "darkred"},
        )
        apply_standard_plot_style(f"{prettify_label(feature)} и ATS-оценка", prettify_label(feature), ATS_SCORE_LABEL)
        plt.tight_layout()
        plt.savefig(output_dir / f"{feature}_vs_score.png", dpi=220)
        plt.close()


def plot_ngram_barplot(df, title, filename, output_dir, color):
    if df.empty:
        return

    plot_df = df.copy().head(15).iloc[::-1]
    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=plot_df,
        x="presence_rate_difference",
        y="ngram",
        orient="h",
        color=color,
    )
    plt.axvline(0, color="black", linewidth=1)
    apply_standard_plot_style(title, PRESENCE_RATE_DIFF_LABEL, NGRAM_LABEL)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=220)
    plt.close()


def plot_group_ngram_results(vacancy_ngrams, output_dir):
    if vacancy_ngrams.empty:
        return

    positive_df = (
        vacancy_ngrams[vacancy_ngrams["association"] == "higher_score"]
        .sort_values(["vacancy_group", "chi2_score"], ascending=[True, False])
        .groupby("vacancy_group", as_index=False)
        .head(1)
        .copy()
    )
    negative_df = (
        vacancy_ngrams[vacancy_ngrams["association"] == "lower_score"]
        .sort_values(["vacancy_group", "chi2_score"], ascending=[True, False])
        .groupby("vacancy_group", as_index=False)
        .head(1)
        .copy()
    )

    if not positive_df.empty:
        positive_df["group_phrase"] = positive_df["vacancy_group"] + " | " + positive_df["ngram"]
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=positive_df.sort_values("presence_rate_difference").iloc[::-1],
            x="presence_rate_difference",
            y="group_phrase",
            orient="h",
            color="#54A24B",
        )
        plt.axvline(0, color="black", linewidth=1)
        apply_standard_plot_style(
            "Топ положительно связанных n-грамм по группам вакансий",
            PRESENCE_RATE_DIFF_LABEL,
            f"{VACANCY_GROUP_LABEL} и n-грамма",
        )
        plt.tight_layout()
        plt.savefig(output_dir / "vacancy_group_top_positive_ngrams.png", dpi=220)
        plt.close()

    if not negative_df.empty:
        negative_df["group_phrase"] = negative_df["vacancy_group"] + " | " + negative_df["ngram"]
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=negative_df.sort_values("presence_rate_difference"),
            x="presence_rate_difference",
            y="group_phrase",
            orient="h",
            color="#E45756",
        )
        plt.axvline(0, color="black", linewidth=1)
        apply_standard_plot_style(
            "Топ отрицательно связанных n-грамм по группам вакансий",
            PRESENCE_RATE_DIFF_LABEL,
            f"{VACANCY_GROUP_LABEL} и n-грамма",
        )
        plt.tight_layout()
        plt.savefig(output_dir / "vacancy_group_top_negative_ngrams.png", dpi=220)
        plt.close()


def write_summary(sentiment_df, resume_ngrams, job_ngrams, vacancy_ngrams, group_counts, args, output_dir):
    lines = [
        "Анализ словосочетаний и тональности",
        "",
        "Тональность:",
    ]

    for _, row in sentiment_df.head(5).iterrows():
        lines.append(
            f"- {feature_label(row['feature'])}: Pearson r={row['pearson_r']:.4f}, Spearman rho={row['spearman_rho']:.4f}"
        )

    lines.extend(["", "Словосочетания резюме, связанные с более высокой ATS-оценкой:"])
    for _, row in resume_ngrams[resume_ngrams["association"] == "higher_score"].head(args.top_k).iterrows():
        lines.append(
            f"- {row['ngram']}: diff={row['presence_rate_difference']:.4f}, chi2={row['chi2_score']:.2f}"
        )

    lines.extend(["", "Словосочетания резюме, связанные с более низкой ATS-оценкой:"])
    for _, row in resume_ngrams[resume_ngrams["association"] == "lower_score"].head(args.top_k).iterrows():
        lines.append(
            f"- {row['ngram']}: diff={row['presence_rate_difference']:.4f}, chi2={row['chi2_score']:.2f}"
        )

    lines.extend(["", "Словосочетания вакансий, связанные с более высокой ATS-оценкой:"])
    for _, row in job_ngrams[job_ngrams["association"] == "higher_score"].head(args.top_k).iterrows():
        lines.append(
            f"- {row['ngram']}: diff={row['presence_rate_difference']:.4f}, chi2={row['chi2_score']:.2f}"
        )

    if not vacancy_ngrams.empty:
        lines.extend(["", "Группы вакансий для отдельного анализа:"])
        for group_name, group_size in group_counts[group_counts >= args.min_group_size].head(args.max_groups).items():
            lines.append(f"- {group_name}: {int(group_size)} наблюдений")

    lines.extend(
        [
            "",
            "Подробные результаты сохранены в CSV-файлах и графиках папки phrase_analysis_outputs.",
        ]
    )

    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка датасета {args.dataset} ({args.split})...")
    df = load_raw_data(args.dataset, args.split, args.sample_rows)
    print(f"Загружено строк: {len(df)}")

    print("Расчет тональности...")
    df = add_sentiment_columns(df)
    df["vacancy_group"] = normalize_text_series(df["jd_text"]).apply(extract_vacancy_group)
    sentiment_df = analyze_sentiment(df, output_dir)

    print("Анализ n-грамм резюме по всему датасету...")
    resume_ngrams = save_overall_ngram_analysis(
        df,
        text_column="resume_text",
        output_name="overall_resume_ngrams.csv",
        args=args,
        output_dir=output_dir,
    )

    print("Анализ n-грамм вакансий по всему датасету...")
    job_ngrams = save_overall_ngram_analysis(
        df,
        text_column="jd_text",
        output_name="overall_job_ngrams.csv",
        args=args,
        output_dir=output_dir,
    )

    print("Анализ n-грамм внутри групп вакансий...")
    vacancy_ngrams, group_counts = save_group_ngram_analysis(df, args, output_dir)

    print("Сохранение графиков...")
    plot_sentiment_results(df, sentiment_df, output_dir)
    plot_ngram_barplot(
        resume_ngrams[resume_ngrams["association"] == "higher_score"],
        "Словосочетания резюме, связанные с более высокой ATS-оценкой",
        "resume_ngrams_higher_score.png",
        output_dir,
        "#54A24B",
    )
    plot_ngram_barplot(
        resume_ngrams[resume_ngrams["association"] == "lower_score"],
        "Словосочетания резюме, связанные с более низкой ATS-оценкой",
        "resume_ngrams_lower_score.png",
        output_dir,
        "#E45756",
    )
    plot_ngram_barplot(
        job_ngrams[job_ngrams["association"] == "higher_score"],
        "Словосочетания вакансий, связанные с более высокой ATS-оценкой",
        "job_ngrams_higher_score.png",
        output_dir,
        "#4C78A8",
    )
    plot_ngram_barplot(
        job_ngrams[job_ngrams["association"] == "lower_score"],
        "Словосочетания вакансий, связанные с более низкой ATS-оценкой",
        "job_ngrams_lower_score.png",
        output_dir,
        "#F58518",
    )
    plot_group_ngram_results(vacancy_ngrams, output_dir)

    print("Формирование summary...")
    write_summary(sentiment_df, resume_ngrams, job_ngrams, vacancy_ngrams, group_counts, args, output_dir)

    print(f"Готово. Результаты сохранены в {output_dir}.")
    print(sentiment_df.to_string(index=False))


if __name__ == "__main__":
    main()
