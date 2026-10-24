# Winnipeg Property Value Predictor

This project predicts the assessed value of detached houses in Winnipeg. I used the open data from the City of Winnipeg.

My best model is a tuned gradient boosting model. On houses it never saw, the average error is about $25,500. This is about 6% of the median house value ($411,000). If I just guess the median price for every house, the average error is about $133,900.

> **Note:** I predict the *assessed value*, not the sale price. The city uses the assessed value for property tax. It can be different from the price a house sells for.

## Results

I measured all numbers on 35,531 test houses. The models did not see these houses during training.

| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| Median-price guess (baseline) | $133,916 | n/a | n/a |
| Linear regression | $32,301 | $68,130 | 0.891 |
| Random Forest | $27,917 | $59,847 | 0.916 |
| Gradient boosting (default settings) | $28,499 | $58,454 | 0.920 |
| **Gradient boosting (tuned)** | **$25,518** | **$54,541** | **0.930** |

MAE is the average error in dollars. RMSE is like MAE, but big mistakes count more. R² shows how much of the change in price the model explains (1.0 is perfect).

**Where the model works well and where it does not** (tuned model, houses grouped by actual price):

| Price band | Median error | Test houses |
|---|---|---|
| Under 300K | 7.7% | 6,801 |
| 300K to 500K | 3.3% | 17,889 |
| 500K to 800K | 2.6% | 9,524 |
| 800K to 1.5M | 7.4% | 1,190 |
| Over 1.5M | 9.9% | 127 |

The model is best for mid-priced houses (about 77% of the test houses). It is worst for the cheapest and the most expensive houses.

## Data

- **Source:** City of Winnipeg Open Data Portal (data.winnipeg.ca), dataset "Assessment Parcels". The licence is Open Government Licence - Winnipeg.
- **Version I used:** I downloaded it on 2026-10-10. It has 245,487 rows and 71 columns. The city updates the data every day, so a new download will be a little different.
- **The data is not in this repository.** The raw file is about 267 MB, and this is too big for GitHub. To run the project, download the CSV from the portal and save it as `data/Assessment_Parcels_20261010.csv`. You can also change the file name in the first cell of `01_cleaning.ipynb`.

## Project structure

```
notebooks/
  01_cleaning.ipynb   Load the raw data, explore it, and clean it
  02_eda.ipynb        Charts and what I found in the cleaned data
  03_modeling.ipynb   Features, models, tuning, and error analysis
data/                 Not in the repository (see Data above)
requirements.txt
```

## What I did

### 1. Cleaning (`01_cleaning.ipynb`)

The raw file has 245,487 properties of all types: houses, condos, commercial, parks. First I explored the data and wrote down each cleaning decision with the reason. Then I did the cleaning.

- **Scope:** I kept only detached single dwellings (177,823 properties). I wanted the model to learn from one type of property.
- **Data leakage:** The columns `Assessed Value 1` to `5` have the same value as the target. The model could just copy the answer, so I dropped them and the other assessment detail columns.
- **Formatting:** The value and area columns were text with `$` and `,` inside. I changed them to numbers. I changed the Yes/No columns to 1/0.
- **Rows I removed (169 in total):**
  - 56 land-only parcels with no house
  - 10 other properties with zero dwelling units and strange values
  - 30 houses with 2 or fewer rooms
  - 7 houses with a non-residential property class
  - 33 houses with an assessed value below $50,000 (these looked like special assessments)
  - 33 houses with a missing basement, rooms, or zoning value
- **Columns:** I reduced 71 columns to 35. The dropped columns were empty, always the same value, a copy of the target, or only an ID. I changed the text column `Property Influences` into 12 flags and one traffic level. I grouped 37 zoning codes into 8 groups.
- **Missing frontage:** Houses with no water frontage have a median lot of about 102,000 sq ft. A normal house lot is about 5,400 sq ft. I think these are rural lots without a city connection, so I filled the missing values with 0.
- **Checks:** I counted every drop rule before and after I used it. The final table has no missing values.

The final table has 177,654 houses and 35 columns.

### 2. Exploration (`02_eda.ipynb`)

- The assessed values are very right-skewed. On a log scale they look close to a bell shape. So the model predicts the log of the value.
- Living area has the strongest link with value (correlation 0.81). Year built is next (0.59), then attached garage (0.58).
- Location matters a lot. The median value is about $140,000 in the cheapest big neighbourhoods and about $1.05 million in the most expensive. This is about 7 times more.
- Houses with an attached garage, a pool, or a fireplace have much higher median values. But these houses are also bigger and in more expensive areas, so the gap is not only from the feature.

### 3. Modeling (`03_modeling.ipynb`)

- **Split:** I used a random 80/20 split: 142,123 houses for training and 35,531 for testing.
- **Features:** I used the log of the value, the log of living area, and the log of land area. I used one-hot encoding for neighbourhood (203 values), market region, building type, and zoning. This gives 258 columns. I fitted the encoder on the training data only.
- **Models:** a median-price baseline, linear regression, Random Forest, and gradient boosting.
- **Tuning:** I used a randomized search with 3-fold cross-validation on the training data only (6 combinations). I used the test set one time, for the final result.
- **Overfitting check:** The tuned model has a training error of $23,324 and a test error of $25,518. They are close, so overfitting is small.

### A bug I found

My first linear regression had a test RMSE of about $70,000 but a training RMSE of about $159,000. This is the wrong way around. The problem was a few very large houses. The model predicted the log of the price, but I gave it the raw living area. One house with 14,348 sq ft had an actual value of $8.98 million, but the model predicted $61.6 million. When I used the log of living area too, the problem was fixed. It shows why I should always compare the training and test errors.

## Limitations

- The target is the assessed value, not a sale price.
- The data has no information about the condition of the house. Some cheap inner-city houses have low assessed values that repeat (for example, many are exactly $30,000 before I removed them). The data cannot explain this.
- Each neighbourhood has only one location point. So the model cannot see things inside a neighbourhood, for example being near the river.
- The model is less accurate for very cheap and very expensive houses. Only 127 test houses are over $1.5 million, so that result is a rough number.
- My hyperparameter search tried only 6 combinations. A bigger search could give a better result.
- All results come from one random split of one version of the data.

## How to run

1. Clone the repository and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Download the Assessment Parcels CSV (see Data above) into `data/`.
3. Start Jupyter (`jupyter lab`) and run the notebooks in order: `01_cleaning`, `02_eda`, `03_modeling`. Notebook 01 saves the cleaned table, and notebooks 02 and 03 load it.

## Use of AI

I used Claude (made by Anthropic) as a tutor for this project. It explained ideas, told me what to check next, and gave hints and feedback on my code and notes. I wrote the notebook code myself, ran all the checks on the data, and decided which rows and columns to remove and why. I edited the notes and this README so they match what I did. 

## Author

Muhammad Maaz Khan
