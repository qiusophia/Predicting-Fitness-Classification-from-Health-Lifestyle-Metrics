# Predicting Fitness from Health & Lifestyle Data

Can you tell if someone's physically fit just by looking at their basic health numbers and daily habits? That's the question I wanted to answer here.

I used a synthetic Kaggle dataset of 2,000 people to build and compare seven classification models in R. The goal was to predict whether someone is **Fit** or **Not Fit**.

## The data

[Fitness Classification Dataset (Synthetic)](https://www.kaggle.com/datasets/muhammedderric/fitness-classification-dataset-synthetic) by Mohammed Darrige on Kaggle.

It has 10 predictors:

- **Physical measurements:** age, height, weight, resting heart rate, blood pressure
- **Habits:** sleep hours, nutrition quality, activity index, smoking status
- **Demographics:** gender

Before modeling, I cleaned up two things. I dropped about 160 rows with missing sleep values, and I fixed the smoking column, which mixed `1`, `"1"`, `"yes"`, `0`, `"0"` and `"no"`. That left 1,840 rows.

## What I did

1. Cleaned the data and explored it with EDA (distributions, boxplots, a correlation matrix, and class balance)
2. Made a stratified 80/20 train/test split
3. Built one shared preprocessing recipe (dummy encoding plus centering and scaling)
4. Tuned models using 10-fold stratified cross-validation, with ROC AUC as the main metric
5. Evaluated the best model on the held-out test set

## Results

| Model | CV ROC AUC |
|---|---|
| **Elastic Net** | **0.863** |
| Logistic Regression | 0.862 |
| LDA | 0.862 |
| QDA | 0.862 |
| Boosted Tree | 0.853 |
| Random Forest | 0.845 |
| Decision Tree | 0.795 |

The fancy models didn't win. Random forest hit 0.976 ROC AUC on the training data, but only 0.845 under cross-validation. That's classic overfitting.

The elastic net (penalty = 0.0022, mixture = 0.667) was the most consistent model. On the test set, it scored:

- **Test ROC AUC:** 0.91
- **Test accuracy:** about 82%

The strongest predictors were **activity level, nutrition quality, and smoking status**. Height and weight barely mattered. So the model mostly confirmed what we already know: move more, eat better, don't smoke.

## Tools

R with `tidymodels`, `discrim`, `vip`, `finalfit`, `ggcorrplot`, `kableExtra`, and the `tidyverse`

## Repo contents

- `231final.html` is the full report, with code, plots, and write-up
- `fitness_dataset.csv` is the dataset
- `fitness_*.rda` files are the saved model results. Tuning takes about 20 minutes, so these let you skip that step.

## Caveats

This data is synthetic. The predictors are almost completely uncorrelated (height and weight don't even move together), which you'd never see in real health data. A good next step would be trying this approach on real, messier data, where the tree-based models might actually earn their keep.
