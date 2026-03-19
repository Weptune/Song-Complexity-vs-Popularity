import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from scipy.stats import f_oneway
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from src.load_data import load_dataset
from src.preprocess import preprocess_data
from src.complexity_index import add_complexity_dimensions
from src.clustering import perform_clustering


# ---------- Plot 1: Overall Complexity vs Popularity ----------
def plot_complexity_vs_popularity(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='musical_complexity',
        y='popularity',
        data=df,
        alpha=0.3
    )
    sns.regplot(
        x='musical_complexity',
        y='popularity',
        data=df,
        scatter=False,
        color='red'
    )
    plt.title("Musical Complexity vs Popularity")
    plt.xlabel("Overall Musical Complexity")
    plt.ylabel("Popularity")
    plt.tight_layout()
    plt.savefig("results/plots/complexity_vs_popularity.png")
    plt.close()


# ---------- Plot 2: Correlation by Complexity Dimension ----------
def plot_dimension_correlations(df):
    dimensions = [
        'rhythmic_complexity',
        'dynamic_complexity',
        'structural_complexity'
    ]

    correlations = {
        dim: df[dim].corr(df['popularity'])
        for dim in dimensions
    }

    plt.figure(figsize=(7, 5))
    sns.barplot(
        x=list(correlations.keys()),
        y=list(correlations.values())
    )
    plt.axhline(0, color='black', linewidth=1)
    plt.title("Correlation of Complexity Dimensions with Popularity")
    plt.ylabel("Pearson Correlation")
    plt.tight_layout()
    plt.savefig("results/plots/complexity_dimension_correlation.png")
    plt.close()


# ---------- Plot 3: Popularity by Complexity Level ----------
def plot_popularity_by_complexity_level(df):
    df['complexity_level'] = pd.qcut(
        df['musical_complexity'],
        q=3,
        labels=['Low', 'Medium', 'High']
    )

    plt.figure(figsize=(7, 5))
    sns.boxplot(
        x='complexity_level',
        y='popularity',
        data=df
    )
    plt.title("Popularity Distribution by Complexity Level")
    plt.xlabel("Complexity Level")
    plt.ylabel("Popularity")
    plt.tight_layout()
    plt.savefig("results/plots/popularity_by_complexity_level.png")
    plt.close()


# ---------- Plot 4: Clusters in Complexity Space ----------
def plot_complexity_space_clusters(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x='rhythmic_complexity',
        y='structural_complexity',
        hue='cluster',
        data=df,
        alpha=0.4
    )
    plt.title("Song Clusters in Complexity Space")
    plt.xlabel("Rhythmic Complexity")
    plt.ylabel("Structural Complexity")
    plt.tight_layout()
    plt.savefig("results/plots/complexity_space_clusters.png")
    plt.close()


# ---------- Plot 5: Each Metric vs Popularity ----------
def plot_metric_vs_popularity(df):
    metrics = [
        'danceability',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo',
        'duration_ms'
    ]

    for metric in metrics:
        plt.figure(figsize=(7, 5))

        sns.scatterplot(
            x=metric,
            y='popularity',
            data=df,
            alpha=0.25
        )

        sns.regplot(
            x=metric,
            y='popularity',
            data=df,
            scatter=False,
            color='red'
        )

        plt.title(f"{metric.replace('_', ' ').title()} vs Popularity")
        plt.xlabel(metric.replace('_', ' ').title())
        plt.ylabel("Popularity")

        plt.tight_layout()
        plt.savefig(f"results/plots/{metric}_vs_popularity.png")
        plt.close()

def plot_feature_importance():
    importance_df = pd.read_csv(
        "results/models/permutation_feature_importance.csv"
    )

    plt.figure(figsize=(8, 6))
    sns.barplot(
        x='importance',
        y='feature',
        data=importance_df,
        order=importance_df.sort_values(
            'importance',
            ascending=False
        )['feature']
    )

    plt.title("Permutation Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("results/plots/feature_importance.png")
    plt.close()

def plot_cumulative_importance():
    importance_df = pd.read_csv(
        "results/models/permutation_feature_importance.csv"
    )

    importance_df = importance_df.sort_values(
        'importance',
        ascending=False
    )

    importance_df['cumulative'] = (
        importance_df['importance'].cumsum()
        / importance_df['importance'].sum()
    )

    plt.figure(figsize=(7, 5))
    plt.plot(
        range(1, len(importance_df)+1),
        importance_df['cumulative'],
        marker='o'
    )

    plt.xticks(range(1, len(importance_df)+1))
    plt.xlabel("Number of Features")
    plt.ylabel("Cumulative Importance")
    plt.title("Cumulative Feature Contribution")
    plt.tight_layout()
    plt.savefig("results/plots/cumulative_importance.png")
    plt.close()

def plot_extreme_comparison(df):
    threshold_high = df['popularity'].quantile(0.9)
    threshold_low = df['popularity'].quantile(0.1)

    high = df[df['popularity'] >= threshold_high]
    low = df[df['popularity'] <= threshold_low]

    features = [
        'loudness',
        'energy',
        'acousticness',
        'instrumentalness'
    ]

    for feature in features:
        plt.figure(figsize=(6, 4))

        sns.kdeplot(high[feature], label="Top 10%", fill=True)
        sns.kdeplot(low[feature], label="Bottom 10%", fill=True)

        plt.title(f"{feature} Distribution: Top vs Bottom Songs")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/plots/extreme_{feature}.png")
        plt.close()

def plot_energy_structure_heatmap(df):
    df['energy_bin'] = pd.qcut(df['energy'], 4)
    df['structure_bin'] = pd.qcut(df['structural_complexity'], 4)

    pivot = df.pivot_table(
        values='popularity',
        index='energy_bin',
        columns='structure_bin',
        aggfunc='mean'
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")

    plt.title("Mean Popularity: Energy vs Structural Complexity")
    plt.tight_layout()
    plt.savefig("results/plots/energy_structure_heatmap.png")
    plt.close()

def run_regression_model(df):
    features = [
        'energy',
        'loudness',
        'danceability',
        'acousticness',
        'instrumentalness',
        'tempo',
        'valence'
    ]

    X = df[features]
    X = sm.add_constant(X)
    y = df['popularity']

    model = sm.OLS(y, X).fit()

    with open("results/models/regression_summary.txt", "w") as f:
        f.write(model.summary().as_text())

    print("Regression model saved.")

def run_interaction_model(df):
    df['energy_x_structure'] = (
        df['energy'] * df['structural_complexity']
    )

    features = [
        'energy',
        'structural_complexity',
        'energy_x_structure'
    ]

    X = df[features]
    X = sm.add_constant(X)
    y = df['popularity']

    model = sm.OLS(y, X).fit()

    with open("results/models/interaction_model_summary.txt", "w") as f:
        f.write(model.summary().as_text())

    print("Interaction model saved.")

def plot_all_features_vs_popularity(df):
    features = [
        'danceability',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo',
        'duration_ms'
    ]

    fig, axes = plt.subplots(5, 2, figsize=(14, 20))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        sns.regplot(
            x=feature,
            y='popularity',
            data=df,
            scatter_kws={'alpha': 0.2},
            line_kws={'color': 'red'},
            ax=axes[i]
        )
        axes[i].set_title(f"{feature.replace('_',' ').title()} vs Popularity")

    plt.tight_layout()
    plt.savefig("results/plots/all_features_vs_popularity.png")
    plt.close()

# ---------- Statistical Test: ANOVA ----------
def test_complexity_levels(df):
    df['complexity_level'] = pd.qcut(
        df['musical_complexity'],
        q=3,
        labels=['Low', 'Medium', 'High']
    )

    low = df[df['complexity_level'] == 'Low']['popularity']
    med = df[df['complexity_level'] == 'Medium']['popularity']
    high = df[df['complexity_level'] == 'High']['popularity']

    stat, p = f_oneway(low, med, high)

    print("ANOVA F-statistic:", stat)
    print("ANOVA p-value:", p)


# ---------- Feature Importance (Permutation) ----------
def compute_feature_importance(df):
    features = [
        'danceability',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo',
        'duration_ms'
    ]

    X = df[features]
    y = df['popularity']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=5,
        random_state=42
    )

    importance_df = pd.DataFrame({
        'feature': features,
        'importance': result.importances_mean
    }).sort_values(by='importance', ascending=False)

    importance_df.to_csv(
        "results/models/permutation_feature_importance.csv",
        index=False
    )

    print("Feature importance saved.")


# ---------- Table: Top Association Rules ----------
def export_top_rules():
    rules = pd.read_csv("results/rules/association_rules.csv")

    if rules.empty:
        print("No association rules found.")
        return

    summary = rules[
        ['antecedents', 'consequents', 'confidence', 'lift']
    ].sort_values(
        by='lift',
        ascending=False
    ).head(10)

    summary.to_csv(
        "results/rules/top_association_rules.csv",
        index=False
    )

# ---------- FULL RESEARCH ANALYSIS PIPELINE ----------
if __name__ == "__main__":

    print("==============================================")
    print("Running Full Research Analysis Pipeline")
    print("==============================================")

    # -------------------------------------------
    # 1️⃣ Load and Prepare Data
    # -------------------------------------------
    print("Loading and preprocessing dataset...")
    df = load_dataset("data/spotify_audio_features.csv")
    df = preprocess_data(df)
    df = add_complexity_dimensions(df)
    df = perform_clustering(df)

    print("Data preparation complete.")

    # -------------------------------------------
    # 2️⃣ Exploratory Structural Analysis
    # -------------------------------------------
    print("Generating structural plots...")
    plot_complexity_vs_popularity(df)
    plot_dimension_correlations(df)
    plot_popularity_by_complexity_level(df)
    plot_complexity_space_clusters(df)
    plot_metric_vs_popularity(df)
    plot_all_features_vs_popularity(df)

    # -------------------------------------------
    # 3️⃣ Statistical Testing
    # -------------------------------------------
    print("Running ANOVA test...")
    test_complexity_levels(df)

    print("Running regression model...")
    run_regression_model(df)

    print("Running interaction regression model...")
    run_interaction_model(df)

    # -------------------------------------------
    # 4️⃣ Predictive Validation
    # -------------------------------------------
    print("Computing feature importance...")
    compute_feature_importance(df)

    print("Generating feature importance plots...")
    plot_feature_importance()
    plot_cumulative_importance()

    # -------------------------------------------
    # 5️⃣ Regime & Interaction Visualizations
    # -------------------------------------------
    print("Generating extreme comparison plots...")
    plot_extreme_comparison(df)

    print("Generating energy-structure heatmap...")
    plot_energy_structure_heatmap(df)

    # -------------------------------------------
    # 6️⃣ Association Rule Summary
    # -------------------------------------------
    print("Exporting top association rules...")
    export_top_rules()

    print("==============================================")
    print("Full Research Analysis Complete.")
    print("Check results/models, results/plots, and results/rules.")
    print("==============================================")
