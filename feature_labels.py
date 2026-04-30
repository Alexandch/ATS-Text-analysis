FEATURE_LABELS = {
    "resume_word_count": "Число слов в резюме",
    "job_word_count": "Число слов в вакансии",
    "resume_char_count": "Число символов в резюме",
    "job_char_count": "Число символов в вакансии",
    "resume_sentence_count": "Число предложений в резюме",
    "job_sentence_count": "Число предложений в вакансии",
    "resume_avg_sentence_length": "Средняя длина предложения в резюме",
    "job_avg_sentence_length": "Средняя длина предложения в вакансии",
    "resume_avg_word_length": "Средняя длина слова в резюме",
    "job_avg_word_length": "Средняя длина слова в вакансии",
    "resume_lexical_diversity": "Лексическое разнообразие резюме",
    "job_lexical_diversity": "Лексическое разнообразие вакансии",
    "resume_numeric_ratio": "Доля числовых токенов в резюме",
    "job_numeric_ratio": "Доля числовых токенов в вакансии",
    "resume_flesch_reading_ease": "Индекс читаемости резюме",
    "job_flesch_reading_ease": "Индекс читаемости вакансии",
    "resume_sentiment_score": "Тональность резюме",
    "job_sentiment_score": "Тональность вакансии",
    "resume_positive_ratio": "Доля позитивных слов в резюме",
    "resume_negative_ratio": "Доля негативных слов в резюме",
    "job_positive_ratio": "Доля позитивных слов в вакансии",
    "job_negative_ratio": "Доля негативных слов в вакансии",
    "sentiment_difference": "Разница тональности резюме и вакансии",
    "sentiment_gap": "Абсолютная разница тональности",
    "shared_token_count": "Число общих токенов",
    "overlap_resume_ratio": "Доля пересечения токенов в резюме",
    "overlap_job_ratio": "Доля пересечения токенов в вакансии",
    "word_count_ratio": "Отношение числа слов резюме к вакансии",
    "char_count_ratio": "Отношение числа символов резюме к вакансии",
    "sentence_count_ratio": "Отношение числа предложений резюме к вакансии",
    "word_count_difference": "Разность числа слов",
    "char_count_difference": "Разность числа символов",
    "lexical_diversity_difference": "Разность лексического разнообразия",
    "readability_difference": "Разность читаемости",
    "jaccard_similarity": "Коэффициент Жаккара",
    "cosine_similarity": "Косинусное сходство",
    "score": "ATS-оценка",
    "actual_score": "Фактическая ATS-оценка",
    "predicted_score": "Предсказанная ATS-оценка",
    "residual": "Остаток модели",
    "absolute_error": "Абсолютная ошибка",
    "cv_r2_mean": "Средний CV R²",
    "cv_rmse_mean": "Средний CV RMSE",
    "holdout_r2": "R² на holdout",
    "holdout_rmse": "RMSE на holdout",
}

FEATURE_GROUP_LABELS = {
    "structural": "Структурные признаки",
    "lexical": "Лексические признаки",
    "sentiment": "Признаки тональности",
    "pair_alignment": "Признаки согласованности пары",
    "similarity": "Признаки сходства",
    "other": "Прочие признаки",
    "all": "Все признаки",
}

SCENARIO_LABELS = {
    "all_features": "Все признаки",
    "only_group": "Только группа",
    "without_group": "Без группы",
}


def feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name.replace("_", " "))


def feature_group_label(name: str) -> str:
    return FEATURE_GROUP_LABELS.get(name, name.replace("_", " "))


def scenario_label(name: str) -> str:
    return SCENARIO_LABELS.get(name, name.replace("_", " "))
