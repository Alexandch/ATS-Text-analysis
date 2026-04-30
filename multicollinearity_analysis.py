from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from feature_labels import feature_label
from ml_modeling import load_data
from plot_style import PEARSON_CORRELATION_LABEL, VIF_LABEL, apply_standard_plot_style, wrap_label


TARGET_COLUMN = "score"
TOP_FEATURES_HEATMAP = 15


def parse_args():
    parser = argparse.ArgumentParser(
        description="Проверка мультиколлинеарности признаков для ATS-оценки."
    )
    parser.add_argument("--input", default="features.csv", help="Путь к CSV с признаками.")
    parser.add_argument("--output-dir", default="multicollinearity_outputs", help="Папка для результатов.")
    parser.add_argument("--sample-rows", type=int, default=0, help="Если > 0, использовать только первые N строк.")
    parser.add_argument("--corr-threshold", type=float, default=0.85, help="Порог сильной корреляции.")
    parser.add_argument("--top-vif", type=int, default=20, help="Сколько признаков с максимальным VIF выводить в summary.")
    return parser.parse_args()


def compute_vif(df, feature_columns):
    x = df[feature_columns].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    rows = []
    for idx, feature in enumerate(feature_columns):
        y = x_scaled[:, idx]
        other_indexes = [i for i in range(len(feature_columns)) if i != idx]
        other_x = x_scaled[:, other_indexes]

        model = LinearRegression()
        model.fit(other_x, y)
        r2 = model.score(other_x, y)
        vif = np.inf if r2 >= 0.999999 else 1.0 / max(1.0 - r2, 1e-12)
        rows.append({"feature": feature, "vif": vif, "r2_from_others": r2})

    return pd.DataFrame(rows).sort_values(by="vif", ascending=False)


def compute_high_correlation_pairs(df, feature_columns, threshold):
    corr_matrix = df[feature_columns + [TARGET_COLUMN]].corr(numeric_only=True)
    pairs = []
    for i, left in enumerate(feature_columns):
        for right in feature_columns[i + 1:]:
            corr_value = corr_matrix.loc[left, right]
            if abs(corr_value) >= threshold:
                target_left = abs(corr_matrix.loc[left, TARGET_COLUMN])
                target_right = abs(corr_matrix.loc[right, TARGET_COLUMN])
                recommended_drop = right if target_left >= target_right else left
                pairs.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "correlation": corr_value,
                        "abs_correlation": abs(corr_value),
                        "abs_corr_with_score_feature_1": target_left,
                        "abs_corr_with_score_feature_2": target_right,
                        "recommended_drop": recommended_drop,
                    }
                )

    return pd.DataFrame(pairs).sort_values(by="abs_correlation", ascending=False), corr_matrix


def save_plots(corr_matrix, vif_df, output_dir):
    sns.set_theme(style="whitegrid")

    top_score_features = (
        corr_matrix[TARGET_COLUMN]
        .drop(labels=[TARGET_COLUMN])
        .sort_values(key=lambda series: series.abs(), ascending=False)
        .head(TOP_FEATURES_HEATMAP)
        .index.tolist()
    )
    heatmap_features = top_score_features + [TARGET_COLUMN]
    heatmap_df = corr_matrix.loc[heatmap_features, heatmap_features]
    heatmap_df.to_csv(output_dir / "top_multicollinearity_heatmap_matrix.csv", encoding="utf-8-sig")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8, "label": PEARSON_CORRELATION_LABEL},
    )
    apply_standard_plot_style(
        "Корреляции между наиболее информативными признаками",
        "Признаки",
        "Признаки",
    )
    plt.xticks(
        ticks=np.arange(len(heatmap_df.columns)) + 0.5,
        labels=[wrap_label(feature_label(column), width=18) for column in heatmap_df.columns],
        rotation=45,
        ha="right",
    )
    plt.yticks(
        ticks=np.arange(len(heatmap_df.index)) + 0.5,
        labels=[wrap_label(feature_label(index), width=18) for index in heatmap_df.index],
        rotation=0,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "multicollinearity_heatmap.png", dpi=220)
    plt.close()

    plot_vif = vif_df.head(20).iloc[::-1].copy()
    plot_vif["feature"] = plot_vif["feature"].map(feature_label)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=plot_vif, x="vif", y="feature", orient="h", color="#E45756")
    apply_standard_plot_style("Наибольшие значения коэффициента VIF", VIF_LABEL, "Признак")
    plt.tight_layout()
    plt.savefig(output_dir / "top_vif.png", dpi=220)
    plt.close()


def write_summary(vif_df, pair_df, threshold, output_dir):
    lines = [
        "Проверка мультиколлинеарности",
        "",
        f"Порог сильной корреляции: {threshold:.2f}",
        "",
        "Признаки с наибольшими значениями VIF:",
    ]

    for _, row in vif_df.head(10).iterrows():
        lines.append(f"- {feature_label(row['feature'])}: VIF={row['vif']:.3f}")

    lines.extend(["", "Наиболее сильные коррелирующие пары признаков:"])
    if pair_df.empty:
        lines.append("- Пар с корреляцией выше порога не найдено.")
    else:
        for _, row in pair_df.head(10).iterrows():
            lines.append(
            f"- {feature_label(row['feature_1'])} и {feature_label(row['feature_2'])}: "
                f"r={row['correlation']:.4f}, рекомендовано убрать {feature_label(row['recommended_drop'])}"
            )

    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Чтение данных из {args.input}...")
    df, feature_columns = load_data(args.input, sample_rows=args.sample_rows)
    print(f"Загружено строк: {len(df)}; признаков: {len(feature_columns)}")

    print("Расчет VIF...")
    vif_df = compute_vif(df, feature_columns)
    vif_df.to_csv(output_dir / "vif_values.csv", index=False, encoding="utf-8-sig")

    print("Поиск сильно коррелирующих пар признаков...")
    pair_df, corr_matrix = compute_high_correlation_pairs(df, feature_columns, args.corr_threshold)
    pair_df.to_csv(output_dir / "high_correlation_pairs.csv", index=False, encoding="utf-8-sig")
    corr_matrix.to_csv(output_dir / "full_correlation_matrix.csv", encoding="utf-8-sig")

    print("Сохранение графиков...")
    save_plots(corr_matrix, vif_df, output_dir)

    print("Формирование summary...")
    write_summary(vif_df, pair_df, args.corr_threshold, output_dir)

    print(f"Готово. Результаты сохранены в {output_dir}.")
    print(vif_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
