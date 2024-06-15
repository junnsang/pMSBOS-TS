import joblib
import numpy as np
import xgboost

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


class ModelTrainer:
  def __init__(
      self,
      X_train: np.ndarray,
      y_train: np.ndarray
  ) -> None:
    self.X_train = X_train
    self.y_train = y_train
    self.ouput_dir = Path(__file__).parent.parent.joinpath("output")

  def train_xgboost(
      self,
      param: dict,
      tree_method: str,
      eval_set: list = None,
      n_estimators: int = 10000,
      save_model_name: str = None
  ):
    xgb_model = xgboost.XGBRegressor(
        objective="reg:squarederror",
        random_state=0,
        n_estimators=n_estimators,
        tree_method=tree_method,
        **param
    )

    if eval_set:
      xgb_model.fit(
          self.X_train, self.y_train,
          early_stopping_rounds=1000,
          eval_set=eval_set,
          eval_metric="rmse",
          verbose=False
      )
    else:
      xgb_model.fit(
          self.X_train, self.y_train,
          verbose=False
      )

    if save_model_name:
      save_path = str(self.ouput_dir.joinpath("xgboosting", f"{save_model_name}.json"))
      xgb_model.save_model(save_path)

    return xgb_model

  def train_linear_regression(self, save_model_name: str = None):
    lr_model = LinearRegression()
    lr_model.fit(self.X_train, self.y_train)

    if save_model_name:
      save_path = str(self.ouput_dir.joinpath("linear_regression", f"{save_model_name}.joblib"))
      joblib.dump(lr_model, save_path)

    return lr_model

  def train_random_forest(
      self,
      param: dict,
      save_model_name: str = None
  ):
    rf_model = RandomForestRegressor(random_state=0, n_jobs=-1, **param)
    rf_model.fit(self.X_train, self.y_train)

    if save_model_name:
      save_path = str(self.ouput_dir.joinpath("random_forest", f"{save_model_name}.joblib"))
      joblib.dump(rf_model, save_path)

    return rf_model

  def train_ann(
      self,
      param: dict,
      max_iter: int,
      save_model_name: str = None
  ):
    mlp_model = MLPRegressor(random_state=0, max_iter=max_iter, **param)
    mlp_model.fit(self.X_train, self.y_train)

    if save_model_name:
      save_path = str(self.ouput_dir.joinpath("ann", f"{save_model_name}.joblib"))
      joblib.dump(mlp_model, save_path)

    return mlp_model

  def train_gaussian_process_regression(self, save_model_name: str = None):
    kernel = DotProduct() + WhiteKernel()
    gpr_model = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr_model.fit(self.X_train, self.y_train)

    if save_model_name:
      save_path = str(self.ouput_dir.joinpath("gaussian_process_regression", f"{save_model_name}.joblib"))
      joblib.dump(gpr_model, save_path)

    return gpr_model
