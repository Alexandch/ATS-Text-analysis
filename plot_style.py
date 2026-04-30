import matplotlib.pyplot as plt
import textwrap


ATS_SCORE_LABEL = "ATS-оценка"
ACTUAL_SCORE_LABEL = "Фактическая ATS-оценка"
PREDICTED_SCORE_LABEL = "Предсказанная ATS-оценка"
RESIDUAL_LABEL = "Остаток модели"
ABSOLUTE_ERROR_LABEL = "Абсолютная ошибка"
PEARSON_CORRELATION_LABEL = "Коэффициент корреляции Пирсона"
FEATURE_IMPORTANCE_LABEL = "Важность признака"
VIF_LABEL = "Коэффициент VIF"
CV_R2_LABEL = "Средний R² по кросс-валидации"
PRESENCE_RATE_DIFF_LABEL = "Разница доли встречаемости"
FEATURE_LABEL = "Признак"
NGRAM_LABEL = "N-грамма"
VACANCY_GROUP_LABEL = "Группа вакансий"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def apply_standard_plot_style(title, xlabel=None, ylabel=None):
    plt.title(title, fontsize=14, pad=12)
    if xlabel:
        plt.xlabel(xlabel, fontsize=11)
    if ylabel:
        plt.ylabel(ylabel, fontsize=11)


def wrap_label(label, width=18):
    return textwrap.fill(str(label), width=width, break_long_words=False)
