import os
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+|\n+")
STOP_WORDS = set(ENGLISH_STOP_WORDS)
MAX_FEATURES = 20000
SENTIMENT_LEXICON = {
    "achieve": 1.8, "achievement": 1.9, "adaptable": 1.2, "advanced": 1.2, "agile": 1.1,
    "best": 1.9, "boost": 1.5, "capable": 1.4, "certified": 1.4, "collaborative": 1.3,
    "creative": 1.3, "deliver": 1.2, "driven": 1.5, "efficient": 1.6, "enhance": 1.5,
    "excellent": 2.1, "expert": 1.9, "flexible": 1.0, "growth": 1.1, "improve": 1.5,
    "innovative": 1.8, "lead": 1.2, "leadership": 1.5, "optimize": 1.5, "organized": 1.2,
    "passion": 1.4, "positive": 1.3, "proactive": 1.6, "proficient": 1.6, "reliable": 1.4,
    "responsible": 1.2, "robust": 1.4, "skill": 1.0, "skilled": 1.5, "solid": 1.2,
    "strong": 1.4, "success": 1.8, "successful": 1.8, "support": 1.0, "talent": 1.5,
    "talented": 1.7, "valuable": 1.6, "win": 1.7,
    "bad": -1.8, "challenging": -0.8, "conflict": -1.7, "critical": -1.2, "delay": -1.6,
    "difficult": -1.5, "error": -1.8, "fail": -2.2, "failure": -2.3, "hard": -0.9,
    "issue": -1.4, "lack": -1.8, "lacking": -1.8, "limited": -1.2, "loss": -1.9,
    "low": -1.0, "missing": -1.5, "negative": -1.7, "poor": -1.9, "problem": -1.7,
    "risk": -1.3, "slow": -1.2, "stress": -1.5, "struggle": -1.8, "weak": -1.7, "wrong": -1.8,
}
NEGATION_WORDS = {"no", "not", "never", "none", "without", "hardly", "rarely"}
BOOSTER_WORDS = {
    "very": 1.35, "highly": 1.4, "extremely": 1.6, "strongly": 1.35,
    "significantly": 1.3, "deeply": 1.25, "particularly": 1.2,
}
DAMPENER_WORDS = {
    "slightly": 0.7, "somewhat": 0.8, "partially": 0.85, "moderately": 0.85,
}
STAT_COLUMNS = [
    "resume_word_count",
    "job_word_count",
    "resume_char_count",
    "job_char_count",
    "resume_sentence_count",
    "job_sentence_count",
    "resume_avg_sentence_length",
    "job_avg_sentence_length",
    "resume_avg_word_length",
    "job_avg_word_length",
    "resume_lexical_diversity",
    "job_lexical_diversity",
    "resume_numeric_ratio",
    "job_numeric_ratio",
    "resume_flesch_reading_ease",
    "job_flesch_reading_ease",
    "resume_sentiment_score",
    "job_sentiment_score",
    "resume_positive_ratio",
    "resume_negative_ratio",
    "job_positive_ratio",
    "job_negative_ratio",
    "sentiment_difference",
    "sentiment_gap",
    "shared_token_count",
    "overlap_resume_ratio",
    "overlap_job_ratio",
    "word_count_ratio",
    "char_count_ratio",
    "sentence_count_ratio",
    "word_count_difference",
    "char_count_difference",
    "lexical_diversity_difference",
    "readability_difference",
    "jaccard_similarity",
    "cosine_similarity",
    "score",
]


def normalize_text_series(series):
    """Преобразует колонку к строкам и заменяет пропуски на пустую строку."""
    return series.fillna("").astype(str)


def safe_word_count(text):
    """Считает слова только в непустом строковом значении."""
    if not isinstance(text, str):
        return 0
    stripped = text.strip()
    return len(stripped.split()) if stripped else 0


def extract_alpha_tokens(text):
    """Извлекает только буквенные токены в нижнем регистре."""
    if not isinstance(text, str):
        return []
    return TOKEN_PATTERN.findall(text.lower())


def clean_and_tokenize(text):
    """Быстрая токенизация без NLTK, чтобы не зависеть от загрузки ресурсов."""
    tokens = extract_alpha_tokens(text)
    return [token for token in tokens if token not in STOP_WORDS]


def split_sentences(text):
    """Разбивает текст на предложения по базовым разделителям."""
    if not isinstance(text, str):
        return []
    return [part.strip() for part in SENTENCE_SPLIT_PATTERN.split(text) if part.strip()]


@lru_cache(maxsize=50000)
def count_syllables(word):
    """Грубая оценка количества слогов для readability-метрик."""
    word = word.lower().strip()
    if not word:
        return 0

    vowels = "aeiouy"
    syllables = 0
    prev_is_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            syllables += 1
        prev_is_vowel = is_vowel

    if word.endswith("e") and syllables > 1:
        syllables -= 1

    return max(syllables, 1)


def safe_ratio(numerator, denominator):
    """Безопасно делит числа, чтобы не получать деление на ноль."""
    return numerator / denominator if denominator else 0.0


def calculate_flesch_reading_ease(alpha_tokens, sentence_count):
    """Считает индекс удобочитаемости Flesch Reading Ease."""
    word_count = len(alpha_tokens)
    if word_count == 0 or sentence_count == 0:
        return 0.0

    syllable_count = sum(count_syllables(token) for token in alpha_tokens)
    return 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (
        syllable_count / word_count
    )


def calculate_sentiment_features(alpha_tokens):
    """Оценивает тональность текста с учётом отрицаний и усилителей."""
    token_count = len(alpha_tokens)
    if token_count == 0:
        return {
            "sentiment_score": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
        }

    weighted_positive = 0.0
    weighted_negative = 0.0
    compound_sum = 0.0

    for index, token in enumerate(alpha_tokens):
        if token not in SENTIMENT_LEXICON:
            continue

        score = SENTIMENT_LEXICON[token]
        previous_window = alpha_tokens[max(0, index - 2):index]

        if any(prev_token in NEGATION_WORDS for prev_token in previous_window):
            score *= -0.85

        booster = 1.0
        for prev_token in previous_window:
            if prev_token in BOOSTER_WORDS:
                booster *= BOOSTER_WORDS[prev_token]
            elif prev_token in DAMPENER_WORDS:
                booster *= DAMPENER_WORDS[prev_token]

        score *= booster
        compound_sum += score

        if score > 0:
            weighted_positive += score
        elif score < 0:
            weighted_negative += abs(score)

    normalization = np.sqrt(compound_sum ** 2 + 15.0)
    normalized_score = safe_ratio(compound_sum, normalization)

    return {
        "sentiment_score": normalized_score,
        "positive_ratio": safe_ratio(weighted_positive, token_count),
        "negative_ratio": safe_ratio(weighted_negative, token_count),
    }


def extract_text_features(text):
    """Собирает расширенные текстовые признаки для одного текста."""
    text = text if isinstance(text, str) else ""
    alpha_tokens = extract_alpha_tokens(text)
    filtered_tokens = [token for token in alpha_tokens if token not in STOP_WORDS]
    word_count = safe_word_count(text)
    char_count = len(text)
    sentence_count = len(split_sentences(text))
    line_count = len([line for line in text.splitlines() if line.strip()])
    numeric_token_count = len(NUMBER_PATTERN.findall(text))
    avg_word_length = safe_ratio(sum(len(token) for token in alpha_tokens), len(alpha_tokens))
    lexical_diversity = safe_ratio(len(set(alpha_tokens)), len(alpha_tokens))
    sentiment_features = calculate_sentiment_features(alpha_tokens)

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "line_count": line_count,
        "avg_sentence_length": safe_ratio(word_count, sentence_count),
        "avg_word_length": avg_word_length,
        "lexical_diversity": lexical_diversity,
        "numeric_ratio": safe_ratio(numeric_token_count, word_count),
        "flesch_reading_ease": calculate_flesch_reading_ease(alpha_tokens, sentence_count),
        "sentiment_score": sentiment_features["sentiment_score"],
        "positive_ratio": sentiment_features["positive_ratio"],
        "negative_ratio": sentiment_features["negative_ratio"],
        "tokens": filtered_tokens,
    }


def calculate_jaccard(tokens1, tokens2):
    """Считает индекс Жаккара между двумя наборами токенов."""
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def calculate_cosine_similarities(resume_texts, job_texts):
    """
    Считает cosine similarity пакетно:
    vectorizer обучается один раз на всём корпусе, а затем
    для каждой пары берётся скалярное произведение строк TF-IDF.
    """
    combined_texts = pd.concat([resume_texts, job_texts], ignore_index=True)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES)
    tfidf_matrix = vectorizer.fit_transform(combined_texts)

    split_index = len(resume_texts)
    resume_matrix = tfidf_matrix[:split_index]
    job_matrix = tfidf_matrix[split_index:]

    return resume_matrix.multiply(job_matrix).sum(axis=1).A1


def calculate_pair_features(df):
    """Считает признаки согласованности между резюме и вакансией."""
    resume_token_sets = df["resume_tokens"].apply(set)
    job_token_sets = df["job_tokens"].apply(set)
    shared_token_counts = [
        len(resume_set & job_set)
        for resume_set, job_set in zip(resume_token_sets, job_token_sets)
    ]

    df["shared_token_count"] = shared_token_counts
    df["overlap_resume_ratio"] = [
        safe_ratio(shared_count, len(resume_set))
        for shared_count, resume_set in zip(shared_token_counts, resume_token_sets)
    ]
    df["overlap_job_ratio"] = [
        safe_ratio(shared_count, len(job_set))
        for shared_count, job_set in zip(shared_token_counts, job_token_sets)
    ]
    df["word_count_ratio"] = [
        safe_ratio(resume_count, job_count)
        for resume_count, job_count in zip(df["resume_word_count"], df["job_word_count"])
    ]
    df["char_count_ratio"] = [
        safe_ratio(resume_count, job_count)
        for resume_count, job_count in zip(df["resume_char_count"], df["job_char_count"])
    ]
    df["sentence_count_ratio"] = [
        safe_ratio(resume_count, job_count)
        for resume_count, job_count in zip(
            df["resume_sentence_count"], df["job_sentence_count"]
        )
    ]
    df["word_count_difference"] = df["resume_word_count"] - df["job_word_count"]
    df["char_count_difference"] = df["resume_char_count"] - df["job_char_count"]
    df["lexical_diversity_difference"] = (
        df["resume_lexical_diversity"] - df["job_lexical_diversity"]
    )
    df["readability_difference"] = (
        df["resume_flesch_reading_ease"] - df["job_flesch_reading_ease"]
    )
    df["sentiment_difference"] = (
        df["resume_sentiment_score"] - df["job_sentiment_score"]
    )
    df["sentiment_gap"] = df["sentiment_difference"].abs()
    return df


def extract_features(df):
    """Добавляет признаки для дальнейшего анализа."""
    df = df.copy()

    print("Извлечение признаков: расширенные метрики резюме...", flush=True)
    resume_feature_rows = [
        extract_text_features(text) for text in normalize_text_series(df["resume_text"])
    ]
    resume_features = pd.DataFrame(resume_feature_rows, index=df.index).add_prefix("resume_")

    print("Извлечение признаков: расширенные метрики вакансии...", flush=True)
    job_feature_rows = [
        extract_text_features(text) for text in normalize_text_series(df["jd_text"])
    ]
    job_features = pd.DataFrame(job_feature_rows, index=df.index).add_prefix("job_")

    df = pd.concat([df, resume_features, job_features], axis=1)

    print("Извлечение признаков: индекс Жаккара...", flush=True)
    df["jaccard_similarity"] = [
        calculate_jaccard(resume_tokens, job_tokens)
        for resume_tokens, job_tokens in zip(df["resume_tokens"], df["job_tokens"])
    ]

    print("Извлечение признаков: признаки согласованности пары...", flush=True)
    df = calculate_pair_features(df)

    print("Извлечение признаков: cosine similarity (пакетный TF-IDF)...", flush=True)
    resume_texts = normalize_text_series(df["resume_text"])
    job_texts = normalize_text_series(df["jd_text"])
    df["cosine_similarity"] = calculate_cosine_similarities(resume_texts, job_texts)

    return df.drop(columns=["resume_tokens", "job_tokens"])


def main():
    max_rows = int(os.getenv("MAX_ROWS", "0"))
    dataset_name = "VaishnaviGude/ats-resume-dataset-1lakh"
    split_name = f"train[:{max_rows}]" if max_rows > 0 else "train"

    print(f"Загрузка датасета ({split_name})...", flush=True)
    try:
        train_dataset = load_dataset(dataset_name, split=split_name)
        print("Датасет загружен успешно.", flush=True)
    except Exception as exc:
        print(f"Ошибка при загрузке датасета: {exc}")
        raise SystemExit(1)

    df = train_dataset.to_pandas()
    if max_rows > 0:
        print(f"Для отладки использую только первые {len(df)} строк.", flush=True)

    print("Колонки в датасете:", df.columns.tolist())
    print("Пример данных:")
    print(df.head(2))

    df = extract_features(df)

    output_path = "features.csv"
    stats_df = df[STAT_COLUMNS].copy()
    stats_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nРезультат для статистического анализа сохранен в {output_path}")

    print("\nГотово! DataFrame для статистического анализа:")
    print(stats_df.head())


if __name__ == "__main__":
    main()
