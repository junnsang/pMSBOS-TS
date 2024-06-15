import numpy as np
import pandas as pd

from datetime import datetime
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, ParameterGrid, train_test_split
from typing import List

from main.DataProcessor import DataProcessor
from main.ModelTrainer import ModelTrainer
from main.utils import adjust_pred_value, get_adjusted_r2, get_95_conf_interval


class ParamGridSearchCV:
  """
  각 모델에 대해 지정한 파라미터 후보들에 대해 파라미터 조합 그리드서치를 진행하고,
  결과를 output 폴더에 저장합니다.
  성능 평가 지표로 mse, adjusted r square, r square를 활용합니다.
  """

  def __init__(
      self,
      one_hot_var_list: List[str],
      numeric_var_list: List[str],
      msbos_var: str,
      outcome_var: str,
      cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
  ) -> None:
    self.one_hot_var_list = one_hot_var_list
    self.numeric_var_list = numeric_var_list
    self.msbos_var = msbos_var
    self.outcome_var = outcome_var
    self.cv = cv

    self.X_trainval, self.X_test, self.y_trainval, self.y_test = self._get_np_X_y_datasets()
    self.ouput_dir = Path(__file__).parent.parent.joinpath("output")

  def _get_np_X_y_datasets(self):
    X_trainval, X_test, y_trainval, y_test = DataProcessor().make_modeling_X_y_datasets(
        one_hot_var_list=self.one_hot_var_list,
        numeric_var_list=self.numeric_var_list,
        outcome_var=self.outcome_var
    )
    return (X_trainval.values, X_test.values, y_trainval.values, y_test.values)

  def conduct_msbos_cv(self) -> None:
    """
    msbos의 cv 수행 성능을 구합니다.
    """
    x_trainval, x_test, y_trainval, y_test = DataProcessor().make_msbos_x_y_datasets(
        msbos_var=self.msbos_var,
        outcome_var=self.outcome_var
    )

    start_time = datetime.now()
    print("\n================================================================================================")
    print(f"[{start_time}] Start current practice (msbos) cv")
    print("================================================================================================")

    mse_evals = []
    r2_evals = []

    summ_rbc_vals = []

    cv_round = 0

    for trainval_index, test_index in self.cv.split(x_trainval.values, y_trainval.values):
      cv_round += 1
      _, x_test_in = x_trainval.values[trainval_index], x_trainval.values[test_index]
      _, y_test_in = y_trainval.values[trainval_index], y_trainval.values[test_index]

      summ_rbc_vals.append(int(np.sum(x_test_in)))

      cv_df = pd.DataFrame({"msbos_vals": x_test_in, "true_vals": y_test_in})
      cv_df.to_csv(
          str(self.ouput_dir.joinpath(
              "msbos",
              "gridsearch_cv_results",
              f"cv_{cv_round}.csv"
          )),
          index=False
      )

      mse = mean_squared_error(y_test_in, x_test_in)
      r2 = r2_score(y_test_in, x_test_in)

      mse_evals.append(mse)
      r2_evals.append(r2)

    mse_mean, mse_95_ci_lower, mse_95_ci_upper = get_95_conf_interval(mse_evals)
    r2_mean, r2_95_ci_lower, r2_95_ci_upper = get_95_conf_interval(r2_evals)
    rbc_mean, rbc_95_ci_lower, rbc_95_ci_upper = get_95_conf_interval(summ_rbc_vals)

    print(f"MSBOS MSE (cv mean [95%CI]): {mse_mean :.3f} [{mse_95_ci_lower :.3f}-{mse_95_ci_upper: .3f}]")
    print(f"MSBOS r2 (cv mean [95%CI]): {r2_mean :.3f} [{r2_95_ci_lower :.3f}-{r2_95_ci_upper: .3f}]")
    print(f"MSBOS Summative predicted RBC (cv mean [95%CI]): {rbc_mean :.3f} [{rbc_95_ci_lower :.3f}-{rbc_95_ci_upper: .3f}]")
    print(f"Cumulative time: {(datetime.now() - start_time).seconds / 60 :.3f} minutes\n")

    result_df = pd.DataFrame({
        "msbos_mse_cv_results": [mse_evals],
        "msbos_r2_results": [r2_evals],
        "msbos_mse_mean": [mse_mean],
        "msbos_mse_95_ci_lower": [mse_95_ci_lower],
        "msbos_mse_95_ci_upper": [mse_95_ci_upper],
        "msbos_r2_mean": [r2_mean],
        "msbos_r2_95_ci_lower": [r2_95_ci_lower],
        "msbos_r2_95_ci_upper": [r2_95_ci_upper],
        "msbos_summ_rbc_mean": [rbc_mean],
        "msbos_summ_rbc_95_ci_lower": [rbc_95_ci_lower],
        "msbos_summ_rbc_95_ci_upper": [rbc_95_ci_upper],
    })

    result_df.to_csv(
        str(self.ouput_dir.joinpath(
            "msbos",
            "gridsearch_cv_summary.csv"
        )),
        index=False
    )

  def conduct_xgb_cv(
      self,
      grid_params: dict,
      tree_method: str = "hist",
      valid_size_in_trainval=1 / 8
  ) -> None:
    start_time = datetime.now()
    gridsearch_results = []
    print("\n================================================================================================")
    print(f"[{start_time}] Start XGBoost model parameter search cv")
    print("================================================================================================")

    for param_ind, param in enumerate(ParameterGrid(grid_params)):
      mse_evals = []
      adj_r2_evals = []
      best_ntree_limits = []

      summ_rbc_vals = []

      print()
      print(f"[{param_ind + 1}] param:\n{param}")

      cv_round = 0
      out_path = self.ouput_dir.joinpath(
          "xgboosting",
          "gridsearch_cv_results",
          f"param_{param_ind + 1}"
      )

      for trainval_index, test_index in self.cv.split(self.X_trainval, self.y_trainval):
        cv_round += 1
        X_trainval_in, X_test_in = self.X_trainval[trainval_index], self.X_trainval[test_index]
        y_trainval_in, y_test_in = self.y_trainval[trainval_index], self.y_trainval[test_index]

        X_train_in, X_valid_in, y_train_in, y_valid_in = train_test_split(
            X_trainval_in, y_trainval_in,
            test_size=valid_size_in_trainval,
            random_state=0
        )

        xgb_model = ModelTrainer(X_train_in, y_train_in).train_xgboost(
            param=param,
            tree_method=tree_method,
            eval_set=[(X_valid_in, y_valid_in)],
            n_estimators=10000
        )
        y_pred = adjust_pred_value(xgb_model.predict(X_test_in))

        summ_rbc_vals.append(int(np.sum(y_pred)))

        cv_df = pd.DataFrame({
            "xgb_pred_vals": xgb_model.predict(X_test_in),
            "adjusted_xgb_pred_vals": y_pred,
            "true_vals": y_test_in
        })

        out_path.mkdir(parents=True, exist_ok=True)

        cv_df.to_csv(
            str(out_path.joinpath(f"cv_{cv_round}.csv")),
            index=False
        )

        mse = mean_squared_error(y_test_in, y_pred)
        mse_evals.append(mse)

        adj_r2 = get_adjusted_r2(y_test_in, y_pred, X_test_in.shape[1])
        adj_r2_evals.append(adj_r2)

        best_ntree_limits.append(xgb_model.best_ntree_limit)

      early_stopping_cv_round_mean = round(np.mean(best_ntree_limits))
      mse_mean, mse_95_ci_lower, mse_95_ci_upper = get_95_conf_interval(mse_evals)
      adj_r2_mean, adj_r2_95_ci_lower, adj_r2_95_ci_upper = get_95_conf_interval(adj_r2_evals)
      rbc_mean, rbc_95_ci_lower, rbc_95_ci_upper = get_95_conf_interval(summ_rbc_vals)

      gridsearch_results.append(
        [
          f"param_{param_ind + 1}", param, mse_evals, adj_r2_evals, best_ntree_limits,
          early_stopping_cv_round_mean,
          mse_mean, mse_95_ci_lower, mse_95_ci_upper,
          adj_r2_mean, adj_r2_95_ci_lower, adj_r2_95_ci_upper,
          rbc_mean, rbc_95_ci_lower, rbc_95_ci_upper
        ]
      )

      print(f"XGBoost Early stopping (cv mean): {early_stopping_cv_round_mean :.3f}")
      print(f"XGBoost MSE (cv mean [95%CI]): {mse_mean :.3f} [{mse_95_ci_lower :.3f}-{mse_95_ci_upper: .3f}]")
      print(f"XGBoost Adj r2 (cv mean [95%CI]): {adj_r2_mean :.3f} [{adj_r2_95_ci_lower :.3f}-{adj_r2_95_ci_upper: .3f}]")
      print(f"XGBoost Summative predicted RBC (cv mean [95%CI]): {rbc_mean :.3f} [{rbc_95_ci_lower :.3f}-{rbc_95_ci_upper: .3f}]")
      print(f"Cumulative time: {(datetime.now() - start_time).seconds / 60 :.3f} minutes\n")

    gridsearch_result_df = pd.DataFrame(
        gridsearch_results,
        columns=[
          "param_nm", "param", "xgb_mse_cv_results", "xgb_adj_r2_cv_results", "early_stopping_cv_rounds",
          "early_stopping_cv_round_mean",
          "xgb_mse_mean", "xgb_mse_95_ci_lower", "xgb_mse_95_ci_upper",
          "xgb_adj_r2_mean", "xgb_adj_r2_95_ci_lower", "xgb_adj_r2_95_ci_upper",
          "xgb_summ_rbc_mean", "xgb_summ_rbc_95_ci_lower", "xgb_summ_rbc_95_ci_upper"
        ]
    ).sort_values(by="xgb_mse_mean")

    gridsearch_result_df.to_csv(
        str(self.ouput_dir.joinpath(
            "xgboosting",
            "gridsearch_cv_summary.csv"
        )),
        index=False
    )

  def conduct_gpr_cv(self) -> None:
    start_time = datetime.now()
    print("\n================================================================================================")
    print(f"[{start_time}] Gaussian Process Regression model parameter search cv")
    print("================================================================================================")

    mse_evals = []
    adj_r2_evals = []

    summ_rbc_vals = []

    cv_round = 0

    for trainval_index, test_index in self.cv.split(self.X_trainval, self.y_trainval):
      cv_round += 1

      X_train_in, X_test_in = self.X_trainval[trainval_index], self.X_trainval[test_index]
      y_train_in, y_test_in = self.y_trainval[trainval_index], self.y_trainval[test_index]

      gpr_model = ModelTrainer(X_train_in, y_train_in).train_gaussian_process_regression()
      y_pred = adjust_pred_value(gpr_model.predict(X_test_in))
      y_pred = np.where(y_pred > 15, 15, y_pred)

      summ_rbc_vals.append(int(np.sum(y_pred)))

      cv_df = pd.DataFrame({
          "gpr_pred_vals": gpr_model.predict(X_test_in),
          "adjusted_lr_pred_vals": y_pred,
          "true_vals": y_test_in
      })

      cv_df.to_csv(
          str(self.ouput_dir.joinpath(
              "gaussian_process_regression",
              "gridsearch_cv_results",
              f"cv_{cv_round}.csv"
          )),
          index=False
      )

      mse = mean_squared_error(y_test_in, y_pred)
      mse_evals.append(mse)

      adj_r2 = get_adjusted_r2(y_test_in, y_pred, X_test_in.shape[1])
      adj_r2_evals.append(adj_r2)

    mse_mean, mse_95_ci_lower, mse_95_ci_upper = get_95_conf_interval(mse_evals)
    adj_r2_mean, adj_r2_95_ci_lower, adj_r2_95_ci_upper = get_95_conf_interval(adj_r2_evals)
    rbc_mean, rbc_95_ci_lower, rbc_95_ci_upper = get_95_conf_interval(summ_rbc_vals)
    
    print(f"Gaussian Process Regression MSE (cv mean [95%CI]): {mse_mean :.3f} [{mse_95_ci_lower :.3f}-{mse_95_ci_upper: .3f}]")
    print(f"Gaussian Process Regression Adj r2 (cv mean [95%CI]): {adj_r2_mean :.3f} [{adj_r2_95_ci_lower :.3f}-{adj_r2_95_ci_upper: .3f}]")
    print(f"Gaussian Process Regression Summative predicted RBC (cv mean [95%CI]): {rbc_mean :.3f} [{rbc_95_ci_lower :.3f}-{rbc_95_ci_upper: .3f}]")
    print(f"Cumulative time: {(datetime.now() - start_time).seconds / 60 :.3f} minutes\n")

    result_df = pd.DataFrame({
        "gpr_mse_cv_results": [mse_evals],
        "gpr_adj_r2_results": [adj_r2_evals],
        "gpr_mse_mean": [mse_mean],
        "gpr_mse_95_ci_lower": [mse_95_ci_lower],
        "gpr_mse_95_ci_upper": [mse_95_ci_upper],
        "gpr_adj_r2_mean": [adj_r2_mean],
        "gpr_adj_r2_95_ci_lower": [adj_r2_95_ci_lower],
        "gpr_adj_r2_95_ci_upper": [adj_r2_95_ci_upper],
        "gpr_summ_rbc_mean": [rbc_mean],
        "gpr_summ_rbc_95_ci_lower": [rbc_95_ci_lower],
        "gpr_summ_rbc_95_ci_upper": [rbc_95_ci_upper],
    })

    result_df.to_csv(
        str(self.ouput_dir.joinpath(
            "gaussian_process_regression",
            "gridsearch_cv_summary.csv"
        )),
        index=False
    )

  def conduct_lr_cv(self) -> None:
    start_time = datetime.now()
    print("\n================================================================================================")
    print(f"[{start_time}] Start Linear Regression model parameter search cv")
    print("================================================================================================")

    mse_evals = []
    adj_r2_evals = []

    summ_rbc_vals = []

    cv_round = 0

    for trainval_index, test_index in self.cv.split(self.X_trainval, self.y_trainval):
      cv_round += 1

      X_train_in, X_test_in = self.X_trainval[trainval_index], self.X_trainval[test_index]
      y_train_in, y_test_in = self.y_trainval[trainval_index], self.y_trainval[test_index]

      lr_model = ModelTrainer(X_train_in, y_train_in).train_linear_regression()
      y_pred = adjust_pred_value(lr_model.predict(X_test_in))
      y_pred = np.where(y_pred > 15, 15, y_pred)

      summ_rbc_vals.append(int(np.sum(y_pred)))

      cv_df = pd.DataFrame({
          "lr_pred_vals": lr_model.predict(X_test_in),
          "adjusted_lr_pred_vals": y_pred,
          "true_vals": y_test_in
      })

      cv_df.to_csv(
          str(self.ouput_dir.joinpath(
              "linear_regression",
              "gridsearch_cv_results",
              f"cv_{cv_round}.csv"
          )),
          index=False
      )

      mse = mean_squared_error(y_test_in, y_pred)
      mse_evals.append(mse)

      adj_r2 = get_adjusted_r2(y_test_in, y_pred, X_test_in.shape[1])
      adj_r2_evals.append(adj_r2)

    mse_mean, mse_95_ci_lower, mse_95_ci_upper = get_95_conf_interval(mse_evals)
    adj_r2_mean, adj_r2_95_ci_lower, adj_r2_95_ci_upper = get_95_conf_interval(adj_r2_evals)
    rbc_mean, rbc_95_ci_lower, rbc_95_ci_upper = get_95_conf_interval(summ_rbc_vals)
    
    print(f"Linear Regression MSE (cv mean [95%CI]): {mse_mean :.3f} [{mse_95_ci_lower :.3f}-{mse_95_ci_upper: .3f}]")
    print(f"Linear Regression Adj r2 (cv mean [95%CI]): {adj_r2_mean :.3f} [{adj_r2_95_ci_lower :.3f}-{adj_r2_95_ci_upper: .3f}]")
    print(f"Linear Regression Summative predicted RBC (cv mean [95%CI]): {rbc_mean :.3f} [{rbc_95_ci_lower :.3f}-{rbc_95_ci_upper: .3f}]")
    print(f"Cumulative time: {(datetime.now() - start_time).seconds / 60 :.3f} minutes\n")

    result_df = pd.DataFrame({
        "lr_mse_cv_results": [mse_evals],
        "lr_adj_r2_results": [adj_r2_evals],
        "lr_mse_mean": [mse_mean],
        "lr_mse_95_ci_lower": [mse_95_ci_lower],
        "lr_mse_95_ci_upper": [mse_95_ci_upper],
        "lr_adj_r2_mean": [adj_r2_mean],
        "lr_adj_r2_95_ci_lower": [adj_r2_95_ci_lower],
        "lr_adj_r2_95_ci_upper": [adj_r2_95_ci_upper],
        "lr_summ_rbc_mean": [rbc_mean],
        "lr_summ_rbc_95_ci_lower": [rbc_95_ci_lower],
        "lr_summ_rbc_95_ci_upper": [rbc_95_ci_upper],
    })

    result_df.to_csv(
        str(self.ouput_dir.joinpath(
            "linear_regression",
            "gridsearch_cv_summary.csv"
        )),
        index=False
    )

  def conduct_rf_cv(self, grid_params: dict) -> None:
    start_time = datetime.now()
    gridsearch_results = []
    print("\n================================================================================================")
    print(f"[{start_time}] Start Random Forest model parameter search cv")
    print("================================================================================================")

    for param_ind, param in enumerate(ParameterGrid(grid_params)):
      mse_evals = []
      adj_r2_evals = []

      summ_rbc_vals = []

      print()
      print(f"[{param_ind + 1}] param:\n{param}")

      cv_round = 0
      out_path = self.ouput_dir.joinpath(
          "random_forest",
          "gridsearch_cv_results",
          f"param_{param_ind + 1}"
      )
      for train_index, test_index in self.cv.split(self.X_trainval, self.y_trainval):
        cv_round += 1
        X_train_in, X_test_in = self.X_trainval[train_index], self.X_trainval[test_index]
        y_train_in, y_test_in = self.y_trainval[train_index], self.y_trainval[test_index]

        rf_model = ModelTrainer(X_train_in, y_train_in).train_random_forest(param)
        y_pred = adjust_pred_value(rf_model.predict(X_test_in))

        summ_rbc_vals.append(int(np.sum(y_pred)))

        cv_df = pd.DataFrame({
            "rf_pred_vals": rf_model.predict(X_test_in),
            "adjusted_rf_pred_vals": y_pred,
            "true_vals": y_test_in
        })

        out_path.mkdir(parents=True, exist_ok=True)

        cv_df.to_csv(
            str(out_path.joinpath(f"cv_{cv_round}.csv")),
            index=False
        )

        mse = mean_squared_error(y_test_in, y_pred)
        mse_evals.append(mse)

        adj_r2 = get_adjusted_r2(y_test_in, y_pred, X_test_in.shape[1])
        adj_r2_evals.append(adj_r2)

      mse_mean, mse_95_ci_lower, mse_95_ci_upper = get_95_conf_interval(mse_evals)
      adj_r2_mean, adj_r2_95_ci_lower, adj_r2_95_ci_upper = get_95_conf_interval(adj_r2_evals)
      rbc_mean, rbc_95_ci_lower, rbc_95_ci_upper = get_95_conf_interval(summ_rbc_vals)

      gridsearch_results.append(
        [
          f"param_{param_ind + 1}", param, mse_evals, adj_r2_evals,
          mse_mean, mse_95_ci_lower, mse_95_ci_upper,
          adj_r2_mean, adj_r2_95_ci_lower, adj_r2_95_ci_upper,
          rbc_mean, rbc_95_ci_lower, rbc_95_ci_upper
        ]
      )

      print(f"Random Forest MSE (cv mean [95%CI]): {mse_mean :.3f} [{mse_95_ci_lower :.3f}-{mse_95_ci_upper: .3f}]")
      print(f"Random Forest Adj r2 (cv mean [95%CI]): {adj_r2_mean :.3f} [{adj_r2_95_ci_lower :.3f}-{adj_r2_95_ci_upper: .3f}]")
      print(f"Random Forest Summative predicted RBC (cv mean [95%CI]): {rbc_mean :.3f} [{rbc_95_ci_lower :.3f}-{rbc_95_ci_upper: .3f}]")
      print(f"Cumulative time: {(datetime.now() - start_time).seconds / 60 :.3f} minutes\n")

    gridsearch_result_df = pd.DataFrame(
        gridsearch_results,
        columns=[
          "param_nm", "param", "rf_mse_cv_results", "rf_adj_r2_cv_results",
          "rf_mse_mean", "rf_mse_95_ci_lower", "rf_mse_95_ci_upper",
          "rf_adj_r2_mean", "rf_adj_r2_95_ci_lower", "rf_adj_r2_95_ci_upper",
          "rf_summ_rbc_mean", "rf_summ_rbc_95_ci_lower", "rf_summ_rbc_95_ci_upper"
        ]
    ).sort_values(by="rf_mse_mean")

    gridsearch_result_df.to_csv(
        str(self.ouput_dir.joinpath(
            "random_forest",
            "gridsearch_cv_summary.csv"
        )),
        index=False
    )

  def conduct_ann_cv(
      self,
      grid_params: dict,
      max_iter: int
  ) -> None:
    start_time = datetime.now()
    gridsearch_results = []
    print("\n================================================================================================")
    print(f"[{start_time}] Start ANN model parameter search cv")
    print("================================================================================================")

    for param_ind, param in enumerate(ParameterGrid(grid_params)):
      mse_evals = []
      adj_r2_evals = []

      summ_rbc_vals = []

      print()
      print(f"[{param_ind + 1}] param:\n{param}")

      cv_round = 0
      out_path = self.ouput_dir.joinpath(
          "ann",
          "gridsearch_cv_results",
          f"param_{param_ind + 1}"
      )

      for train_index, test_index in self.cv.split(self.X_trainval, self.y_trainval):
        cv_round += 1
        X_train_in, X_test_in = self.X_trainval[train_index], self.X_trainval[test_index]
        y_train_in, y_test_in = self.y_trainval[train_index], self.y_trainval[test_index]

        ann_model = ModelTrainer(X_train_in, y_train_in).train_ann(param=param, max_iter=max_iter)
        y_pred = adjust_pred_value(ann_model.predict(X_test_in))

        summ_rbc_vals.append(int(np.sum(y_pred)))

        cv_df = pd.DataFrame({
            "ann_pred_vals": ann_model.predict(X_test_in),
            "adjusted_ann_pred_vals": y_pred,
            "true_vals": y_test_in
        })

        out_path.mkdir(parents=True, exist_ok=True)

        cv_df.to_csv(
            str(out_path.joinpath(f"cv_{cv_round}.csv")),
            index=False
        )

        mse = mean_squared_error(y_test_in, y_pred)
        mse_evals.append(mse)

        adj_r2 = get_adjusted_r2(y_test_in, y_pred, X_test_in.shape[1])
        adj_r2_evals.append(adj_r2)

      mse_mean, mse_95_ci_lower, mse_95_ci_upper = get_95_conf_interval(mse_evals)
      adj_r2_mean, adj_r2_95_ci_lower, adj_r2_95_ci_upper = get_95_conf_interval(adj_r2_evals)
      rbc_mean, rbc_95_ci_lower, rbc_95_ci_upper = get_95_conf_interval(summ_rbc_vals)

      gridsearch_results.append(
        [
          f"param_{param_ind + 1}", param, mse_evals, adj_r2_evals,
          mse_mean, mse_95_ci_lower, mse_95_ci_upper,
          adj_r2_mean, adj_r2_95_ci_lower, adj_r2_95_ci_upper,
          rbc_mean, rbc_95_ci_lower, rbc_95_ci_upper
        ]
      )

      print(f"ANN MSE (cv mean [95%CI]): {mse_mean :.3f} [{mse_95_ci_lower :.3f}-{mse_95_ci_upper: .3f}]")
      print(f"ANN Adj r2 (cv mean [95%CI]): {adj_r2_mean :.3f} [{adj_r2_95_ci_lower :.3f}-{adj_r2_95_ci_upper: .3f}]")
      print(f"ANN Summative predicted RBC (cv mean [95%CI]): {rbc_mean :.3f} [{rbc_95_ci_lower :.3f}-{rbc_95_ci_upper: .3f}]")
      print(f"Cumulative time: {(datetime.now() - start_time).seconds / 60 :.3f} minutes\n")

    gridsearch_result_df = pd.DataFrame(
        gridsearch_results,
        columns=[
          "param_nm", "param", "ann_mse_cv_results", "ann_adj_r2_cv_results",
          "ann_mse_mean", "ann_mse_95_ci_lower", "ann_mse_95_ci_upper",
          "ann_adj_r2_mean", "ann_adj_r2_95_ci_lower", "ann_adj_r2_95_ci_upper",
          "ann_summ_rbc_mean", "ann_summ_rbc_95_ci_lower", "ann_summ_rbc_95_ci_upper"
        ]
    ).sort_values(by="ann_mse_mean")

    gridsearch_result_df.to_csv(
        str(self.ouput_dir.joinpath(
            "ann",
            "gridsearch_cv_summary.csv"
        )),
        index=False
    )
