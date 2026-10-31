# Restaurant Rating Prediction

Predict restaurant ratings with regression models using a Zomato CSV dataset.
The project is designed for common Kaggle schemas, including columns such as
`rate`, `aggregate_rating` (or `Aggregate rating`), `rate (out of 5)`,
`num of ratings`, `avg cost (two people)`, `cuisines`, and `location`.

## Setup

1. Put the downloaded dataset at `data/zomato.csv`.
2. Create and activate a virtual environment.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

## Train and evaluate

```bash
python src/restaurant_rating_prediction.py
```

The script prints a compact data-quality/EDA summary, compares a linear
regression baseline with a random forest regressor on a held-out test set, and
saves the best model and evaluation metrics under `artifacts/`.

## Run the frontend

After training the model, launch the interactive prediction app. The app uses
the trained model and a compact feature metadata file; it does not need the
original CSV at runtime:

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. The result includes a clearly
disclosed presentation adjustment of up to +0.30,
capped at 5.00; it also displays the unadjusted model prediction.

## Deploy on Streamlit Community Cloud

1. Train the model locally so `artifacts/best_model.joblib`,
   `artifacts/metrics.json`, and `artifacts/feature_metadata.json` exist.
2. Push the project to a GitHub repository. The CSV in `data/` is ignored;
   the model, metrics, and feature metadata are included for inference.
3. In Streamlit Community Cloud, choose **Create app**, select the repository,
   branch, and `app.py` as the entry point, then deploy. Keep the pinned
   dependencies in `requirements.txt` to match the serialized model.

The model file is about 59 MB. The training dataset is not required on the
deployed app and remains excluded from Git.

To use a different CSV path:

```bash
python src/restaurant_rating_prediction.py --data path/to/your.csv
```

Target rating columns are detected in this order: `rating`, `rate`,
`aggregate_rating`, `Aggregate rating`, `rate (out of 5)`. Rating values such
as `4.1/5`, `4.1`, `NEW`, and `-` are handled; invalid or missing targets are
excluded. Target-like columns are
removed from the input features to avoid leakage. Obvious identifiers and
free-text review/menu fields are also excluded. When both `area` and
`local address` are present, `local address` is excluded to avoid redundant
location inputs. Cuisine combinations are split into individual cuisine
features; the frontend lets you select multiple cuisines. The location selector
uses the areas represented in the dataset. The dataset must have at least five
rows with valid ratings and at least one usable feature.

## Project layout

```text
data/                     Place zomato.csv here (not committed)
app.py                    Streamlit prediction frontend
src/restaurant_rating_prediction.py
tests/test_restaurant_rating_prediction.py
artifacts/                Trained model, metrics, and feature metadata
```

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```
