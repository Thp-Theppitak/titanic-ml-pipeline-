# Titanic ML Pipeline 🚢

OOP-based Machine Learning pipeline for Titanic survival prediction.

## Models Compared
| Model | CV Score | Stability |
|-------|----------|-----------|
| Random Forest | 0.685 | ±0.045 |
| Logistic Regression | 0.687 | ±0.023 |

> Logistic Regression wins on stability — chosen for production

## Features
- Clean OOP design with `TitanicPipeline` class
- Feature importance visualization
- Multi-model comparison
- Cross-validation scoring

## Tech Stack
Python · Scikit-learn · Pandas · Matplotlib

## Usage
```python
pipeline = TitanicPipeline(model_type="logistic")
pipeline.fit(train_df)
pipeline.summary()
```
