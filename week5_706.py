import kagglehub
path = kagglehub.dataset_download("yadiraespinoza/world-happiness-2015-2024")

import pandas as pd
import numpy as np

combined_path = path + "/world_happiness_combined.csv"
happiness = pd.read_csv(combined_path, sep=";")

print(happiness.info())
print(happiness.describe())

def decimal_format(dataframe, column_name):
    dataframe[column_name] = (
        dataframe[column_name].astype(str).str.replace(",", ".").astype(float)
    )
    return dataframe


happiness = decimal_format(happiness, "Happiness score")
happiness = decimal_format(happiness, "GDP per capita")
happiness = decimal_format(happiness, "Social support")
happiness = decimal_format(happiness, "Freedom to make life choices")
happiness = decimal_format(happiness, "Generosity")
happiness = decimal_format(happiness, "Perceptions of corruption")

print(happiness.isna().sum())
missing_regional = happiness[happiness["Regional indicator"].isna()]
print(missing_regional)

def update_country_regions(dataframe):
    country_regions = {}
    for country in dataframe["Country"].unique():
        latest_row = (
            dataframe[dataframe["Country"] == country].sort_values("Year").iloc[-1]
        )
        country_regions[country] = latest_row["Regional indicator"]

    for i, country in enumerate(dataframe["Country"]):
        if country in country_regions:
            dataframe.loc[i, "Regional indicator"] = country_regions[country]

    return dataframe


happines = update_country_regions(happiness)

print(happiness.isna().sum())

def check_duplicates(dataframe):
    return int(dataframe.duplicated().sum())


print(check_duplicates(happiness))

def year_range(dataframe, years):
    return {year: dataframe[dataframe["Year"] == year] for year in years}


years = range(2015, 2025)
years_df = year_range(happiness, years)


def group_by_regions(dataframe):
    regions = []
    for i in dataframe["Regional indicator"]:
        if not (i in regions):
            regions.append(i)
    return {
        region: dataframe[dataframe["Regional indicator"] == region]
        for region in regions
    }


regions_df = group_by_regions(happiness)

summary_year_region = happiness.groupby(["Year", "Regional indicator"])[
    "Happiness score"
].agg(["count", "mean", "std"])
print(summary_year_region)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


def train_tree_model(df, target):
    if df.empty:
        raise ValueError("Input dataframe is empty")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")

    X = df.drop(columns=[target])
    y = df[target]
    if X.empty or y.empty:
        raise ValueError("Features or target are empty")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    tree_model.fit(X_train, y_train)
    return tree_model, X_test, y_test


tree_model, X_test, y_test = train_tree_model(
    happiness[
        [
            "GDP per capita",
            "Social support",
            "Healthy life expectancy",
            "Freedom to make life choices",
            "Generosity",
            "Perceptions of corruption",
            "Happiness score",
        ]
    ],
    "Happiness score",
)

y_pred = tree_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Decision Tree Performance")
print("R² score:", r2)
print("RMSE:", rmse)

tree_simplified, X_test_simplified, y_test_simplified = train_tree_model(
    happiness[["GDP per capita", "Social support", "Happiness score"]],
    "Happiness score",
)


# X_simplified = happiness[["GDP per capita", "Social support"]]

# X_train_simplified, X_test_simplified, y_train_simplified, y_test_simplified = (
#    train_test_split(X_simplified, y, test_size=0.2)
# )

# tree_simplified = DecisionTreeRegressor(max_depth=5)
# tree_simplified.fit(X_train_simplified, y_train_simplified)

y_pred_simplified = tree_simplified.predict(X_test_simplified)

print("R² (2 features):", r2_score(y_test_simplified, y_pred_simplified))
print(
    "RMSE (2 features):",
    np.sqrt(mean_squared_error(y_test_simplified, y_pred_simplified)),
)

tree_life_exp, X_test_life_exp, y_test_life_exp = train_tree_model(
    happiness[
        [
            "GDP per capita",
            "Social support",
            "Freedom to make life choices",
            "Generosity",
            "Perceptions of corruption",
            "Healthy life expectancy",
        ]
    ],
    "Healthy life expectancy",
)

# y_life_exp = happiness["Healthy life expectancy"]
# X_life_exp = happiness[["GDP per capita", "Social support", "Freedom to make life choices", "Generosity", "Perceptions of corruption", "Healthy life expectancy"]]

# X_train_life_exp, X_test_life_exp, y_train_life_exp, y_test_life_exp = train_test_split(
#     X_life_exp, y_life_exp, test_size=0.2
# )

# tree_life_exp = DecisionTreeRegressor(max_depth=5)
# tree_life_exp.fit(X_train_life_exp, y_train_life_exp)

y_pred_life_exp = tree_life_exp.predict(X_test_life_exp)

print("R² (predicting life expectancy):", r2_score(y_test_life_exp, y_pred_life_exp))

import matplotlib.pyplot as plt

generosity = happiness["Generosity"]
social_support = happiness["Social support"]
plt.scatter(social_support, generosity, alpha=0.3)
plt.xlabel("Social Support")
plt.ylabel("Generosity")
plt.title("Generosity vs Social Support")

gdp_per_capita = happiness["GDP per capita"]
happiness_score = happiness["Happiness score"]
plt.scatter(gdp_per_capita, happiness_score, alpha=0.3)
plt.xlabel("GDP Per Capita")
plt.ylabel("Happiness Score")
plt.title("Happiness Score vs GDP Per Capita")

plt.scatter(social_support, happiness_score, alpha=0.3)
plt.xlabel("Social Support")
plt.ylabel("Happiness Score")
plt.title("Happiness Score vs Social Support")

plt.scatter(generosity, happiness_score, alpha=0.3)
plt.xlabel("Generosity")
plt.ylabel("Happiness Score")
plt.title("Happiness Score vs Generosity")

