"""Exploratory summary of Austin home sales before fitting a regression."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "austin_home_sales.csv"
FIG_DIR = Path(__file__).resolve().parent / "figures"

TARGET = "Sale_Price"
PREDICTORS = ["Sq_Footage", "Bedrooms", "Home_Age_Years", "Dist_Downtown_Mi"]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    FIG_DIR.mkdir(exist_ok=True)

    print(f"rows: {len(df):,}    columns: {df.shape[1]}")
    print("\n--- dtypes ---")
    print(df.dtypes.to_string())
    print("\n--- missing values ---")
    print(df.isna().sum().to_string())
    print("\n--- descriptive statistics ---")
    print(df.describe().T.to_string())

    min_price = df[TARGET].min()
    n_at_floor = int((df[TARGET] == min_price).sum())
    print(
        f"\n{n_at_floor} homes ({n_at_floor / len(df):.0%}) share the minimum "
        f"sale price of ${min_price:,.0f}."
    )

    corr = df.corr(numeric_only=True)
    print("\n--- correlation with Sale_Price ---")
    print(corr[TARGET].drop(TARGET).sort_values(ascending=False).round(3).to_string())
    print("\n--- full correlation matrix ---")
    print(corr.round(3).to_string())

    plot_histograms(df)
    plot_scatters(df, min_price)
    plot_price_by_bedrooms(df)
    plot_correlation(corr)
    print(f"\nfigures written to {FIG_DIR}")


def plot_histograms(df: pd.DataFrame) -> None:
    columns = [TARGET, *PREDICTORS]
    fig, axes = plt.subplots(1, len(columns), figsize=(14, 3.2))
    for ax, column in zip(axes, columns):
        ax.hist(df[column], bins=15, color="#3d5a80", edgecolor="white")
        ax.set_title(column, fontsize=10)
        ax.set_ylabel("count" if column == TARGET else "")
    fig.suptitle("Distributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "histograms.png", dpi=150)
    plt.close(fig)


def plot_scatters(df: pd.DataFrame, floor_price: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    at_floor = df[TARGET] == floor_price
    for ax, column in zip(axes.ravel(), PREDICTORS):
        ax.scatter(
            df.loc[~at_floor, column],
            df.loc[~at_floor, TARGET] / 1000,
            s=18,
            alpha=0.75,
            color="#3d5a80",
            label="other sales",
        )
        ax.scatter(
            df.loc[at_floor, column],
            df.loc[at_floor, TARGET] / 1000,
            s=28,
            alpha=0.9,
            color="#e07a5f",
            label=f"${floor_price:,.0f} sales",
        )
        ax.set_xlabel(column)
        ax.set_ylabel("Sale price ($000)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Sale price against each predictor", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scatters.png", dpi=150)
    plt.close(fig)


def plot_price_by_bedrooms(df: pd.DataFrame) -> None:
    counts = df.groupby("Bedrooms")[TARGET]
    labels = [f"{bedrooms}\n(n={n})" for bedrooms, n in counts.size().items()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        [group.to_numpy() / 1000 for _, group in counts],
        tick_labels=labels,
    )
    ax.set_xlabel("Bedrooms")
    ax.set_ylabel("Sale price ($000)")
    ax.set_title("Sale price by number of bedrooms")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_by_bedrooms.png", dpi=150)
    plt.close(fig)


def plot_correlation(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)), corr.columns)
    for row in range(len(corr)):
        for col in range(len(corr)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
