# Week_2_Mini-Assignment_Start_Your_First_Data_Analysis
IDS 706 mini-assignment, including reading in data, analysis, machine learning, and data visualization.

![CI](https://github.com/matthewtfischer/Week_2_Mini-Assignment_Start_Your_First_Data_Analysis/actions/workflows/ci.yml/badge.svg)

## Dependencies
All dependencies are defined in `requirements.txt`:

```
kagglehub
pandas
numpy
scikit-learn
matplotlib
black==25.9.0
isort==5.13.2
flake8==7.1.1
```

## Setup
To build the Docker image, run the following command:
docker run -p 8888:8888 my-notebook-image

To create the container, run the following command:
docker run -p 8888:8888 my-notebook-image

## CI/CD Pipeline
On every commit, the project automatically installs dependencies, formats code with black, lints code with flake8, runs tests, and generates coverage reports.

## Import the Dataset
This notebook analyzes data from the World Happiness Report over the years 2015 to 2024. This dataset includes statistics and survey data from countries all over the world in order to compare what makes a population "happy." In addition to how happy a population is, the report also collects data including GDP per capita, social support, healthy life expectancy, freedom to make life choices, generosity, and perceptions of corruption.

## Inspect the Data
While looking through the data, I noticed that there were 3 entries that were missing in the "Regional indicators" column. Upon further investigation, there were certain countries that had inconsistent classification under this category. As such, I found that if there were any discrepancies in the regional indicators of a country, I replaced all instances of the regional indicator for that country with the most recent instance.

## Basic Filtering and Grouping
To extract meaningful subsets of the data, I made separate dataframes for the data separated by year and by region. I believe that investigating how individual countries and countries worldwide have changed over time and investigating how different regions of the world are similar or differ would be interesting.

## Explore a Machine Learning Algorithm
Using a decision tree machine learning model, I tried to predict the happiness score from the other variables included in the report. Ultimately, my model had an R² score: of 0.702 and an RMSE of 0.628. I also attempted to simplify the model by only using GDP per capita and social support to predict happiness score, which produced an R² score of 0.641 and an RMSE of 0.650. Finally, I tried to predict healthy life expectancy using the other variables, which resulted in an R² score of 0.512 and an RMSE of 5.39.

## Visualization
I plotted happiness scores as a function of GDP per capita, social support, and generosity.
![Happiness Score vs GDP Per Capita](screenshots_706/happ_score_vs_gdp)
![Happiness Score vs Social Support](screenshots_706/happ_score_vs_ss)
![Happiness Score vs Generosity](screenshots_706/happ_score_vs_gen)
Very interestingly, there seems to be a fairly strong positive relationship between GDP per capita and happiness up until GDP per capita is about 9. For GDP per capita at and above this, happiness score seems to be unrelated and ranges from about 4 to about 7.5. Social support had a strong positive correlation with happiness score, seemingly stronger for higher social support scores. Generosity seemed to have a very weak correlation with happiness scores, most data points were populated around the left of the graph (generosity between 0 and 0.5 and happiness scores between 3 and 7). There were also 2 notable clusters, a tail of countries with high happiness scores (between 0.6 and 1) and low-to-moderate happiness scores (between 3.5 and 5.5) and a cloud of countries with moderate generosity scores (0.3 to 0.7, this is higher than most but less than the perviously states tail) and and high happiness scores (between 7 and 8). I also wanted to look at the relation between social support and generosity, which I expected to be highly correlated, but instead, the data are mostly in the bottom left of the graph (high social support and low generosity), with a tail of points of all social support socres and generosity scores between 1 and 5 and some outliers with social support scores between 0.4 and 0.8 and happiness scores higher than 0.8.
![Generosity vs Social Support](screenshots_706/gen_vs_ss)

## Refactoring
In refactoring my code, I renamed many of my variables to be more descriptive, including renaming path to dataset_path, happiness to happiness_df, years_df to yearly_dfs, regions_df to regional_dfs, tree_model to happiness_tree_model, y_pred to y_pred_happiness, tree_simplified to happiness_tree_simplified, and tree_life_exp to life_exp_tree_model. Also, much of my code was already categorized into functions, so I added load_happiness_data to load in the data set that is being used and assign it to a variable as a dataframe.
