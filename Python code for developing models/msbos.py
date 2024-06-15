import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

from main.DataProcessor import DataProcessor
from main.ParamGridSearchCV import ParamGridSearchCV

one_hot_var_list = None
numeric_var_list = [
  "ami",
  "rend",
  "canc",
  "aPTT",
  "Hb",
  "Plt",
  "PT_inr",
  "coumarin_deriatives",
  "use_quan",
  "msbos"
]
msbos_var = "msbos"
outcome_var = "use_quan"


# =======================================================================
# Current practice (msbos) cv
# Result
# 1. output/msbos/gridsearch_cv_results
# 2. output/msbos/gridsearch_cv_summary.csv
# =======================================================================
CV_N_SPLITS = 5
CV_N_REPEATS = 10

cv = RepeatedKFold(
    n_splits=CV_N_SPLITS,
    n_repeats=CV_N_REPEATS,
    random_state=0
)

param_gridsearch_cv = ParamGridSearchCV(
    one_hot_var_list=one_hot_var_list,
    numeric_var_list=numeric_var_list,
    msbos_var=msbos_var,
    outcome_var=outcome_var,
    cv=cv
)

param_gridsearch_cv.conduct_msbos_cv()


# =======================================================================
# MSBOS evaluation with test dataset
# Result:
# 1. output/msbos/evaluation.csv
# 2. output/msbos/test_data_msbos_values.csv
# =======================================================================
_, x_test, _, y_test = DataProcessor().make_msbos_x_y_datasets(
    msbos_var=msbos_var,
    outcome_var=outcome_var
)

mse = mean_squared_error(y_test.values, x_test.values)
r2 = r2_score(y_test.values, x_test.values)
summative_predicted_rbc_pack = int(np.sum(x_test))

evaluation_df = pd.DataFrame(
  {
    "MSE": [mse],
    "R2": [r2],
    "Summative predicted RBC (pack)": [summative_predicted_rbc_pack]
  }
)
evaluation_df.index = ["MSBOS"]

evaluation_df.to_csv("output/msbos/evaluation.csv")

prediction_df = pd.DataFrame(
  {
    "msbos_vals": x_test,
    "true_vals": y_test
  }
)

prediction_df.to_excel("output/msbos/final_model_test_data_prediction_values.xlsx", index=False)

print("\n================================================================================================")
print("Current practice (msbos) model evaluation")
print("================================================================================================")
print(f"MSBOS MSE: {mse :.3f}")
print(f"MSBOS r2: {r2 :.3f}")
print(f"MSBOS Summative predicted RBC: {summative_predicted_rbc_pack :.3f}")
