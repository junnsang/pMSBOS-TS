import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RepeatedKFold

from main.DataProcessor import DataProcessor
from main.ModelTrainer import ModelTrainer
from main.ParamGridSearchCV import ParamGridSearchCV
from main.utils import get_adjusted_r2, adjust_pred_value


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
# ANN parameter gridsearch cv
# Result
# 1. output/ann/gridsearch_cv_results
# 2. output/ann/gridsearch_cv_summary.csv
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

grid_params = {
    "hidden_layer_sizes": [
        [100],
        [100, 100],
        [100, 50, 25],
        [100, 100, 50],
        [100, 50],
        [50],
        [50, 50],
        [50, 25, 13],
        [50, 50, 25],
        [50, 25]
    ]
}

param_gridsearch_cv.conduct_ann_cv(grid_params=grid_params, max_iter=200)


# =======================================================================
# ANN model training with best params
# Result:
# 1. output/ann/trained_ann_model.joblib
# =======================================================================
X_train, X_test, y_train, y_test = DataProcessor().make_modeling_X_y_datasets(
    one_hot_var_list=one_hot_var_list,
    numeric_var_list=numeric_var_list,
    outcome_var=outcome_var
)

param_search_results = pd.read_csv("output/ann/gridsearch_cv_summary.csv")
best_param = \
    param_search_results\
    .sort_values(by="ann_mse_mean")\
    .head(1)[["param"]]

ann_model = (
  ModelTrainer(X_train=X_train.values, y_train=y_train.values)
  .train_ann(
      param=eval(best_param["param"][0]),
      max_iter=200,
      save_model_name="trained_ann_model"
  )
)


# =======================================================================
# ANN model evaluation with test dataset
# Result:
# 1. output/ann/evaluation.csv
# 2. output/ann/final_model_test_data_prediction_values.csv
# =======================================================================
y_pred = ann_model.predict(X_test.values)

mse = mean_squared_error(y_test.values, adjust_pred_value(y_pred))
adj_r2 = get_adjusted_r2(y_test.values, adjust_pred_value(y_pred), X_test.values.shape[1])
summative_predicted_rbc_pack = int(np.sum(adjust_pred_value(y_pred)))

evaluation_df = pd.DataFrame(
  {
    "MSE": [mse],
    "Adjusted R2": [adj_r2],
    "Summative predicted RBC (pack)": [summative_predicted_rbc_pack]
  }
)
evaluation_df.index = ["ANN model"]

evaluation_df.to_csv("output/ann/evaluation.csv")

prediction_df = pd.DataFrame(
  {
    "ann_pred_vals": y_pred,
    "adjusted_ann_pred_vals": adjust_pred_value(y_pred),
    "true_vals": y_test.values
  }
)

prediction_df.to_excel("output/ann/final_model_test_data_prediction_values.xlsx", index=False)

print("\n================================================================================================")
print("ANN model evaluation")
print("================================================================================================")
print(f"ANN MSE: {mse :.3f}")
print(f"ANN Adj r2: {adj_r2 :.3f}")
print(f"ANN Summative predicted RBC: {summative_predicted_rbc_pack :.3f}")
