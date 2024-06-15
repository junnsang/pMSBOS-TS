import pandas as pd
from pathlib import Path
from typing import List, Union


class DataProcessor:
  def __init__(self) -> None:
      self.data_dir = Path(__file__).parent.parent.joinpath("data")

  def load_raw_datasets(self) -> List[pd.DataFrame]:
    """모델 train과 test에 활용할 raw 데이터셋을 판다스 데이터프레임 타입으로 불러옵니다.

    Returns:
        List[pd.DataFrame]: 모델 train과 test에 활용될 raw 데이터셋
    """
    train_data = pd.read_csv(self.data_dir.joinpath("train_data.csv"))
    test_data = pd.read_csv(self.data_dir.joinpath("test_data.csv"))

    return [train_data, test_data]

  def convert_into_dummy_coded_datasets(
      self,
      df: pd.DataFrame,
      one_hot_var_list: List[str],
      numeric_var_list: List[str]
  ) -> pd.DataFrame:
    """one hot encoding과 float type conversion을 수행합니다.

    Args:
        df (pd.DataFrame): 모델링 데이터셋
        one_hot_vars (List[str]): one hot encoding이 필요한 컬럼 리스트 (3개 이상의 범주형 변수)
        float_vars (List[str]): float로 type conversion이 필요한 컬럼 리스트 (True/False categorical, continuous 변수)

    Returns:
        pd.DataFrame: one hot encoding과 float type conversion이 적용된 모델링 데이터셋
    """
    if one_hot_var_list:
      df[one_hot_var_list] = df[one_hot_var_list].astype(object)
    if numeric_var_list:
      df[numeric_var_list] = df[numeric_var_list].astype(float)
    df = pd.get_dummies(df)

    return df

  def make_modeling_X_y_datasets(
      self,
      one_hot_var_list: List[str],
      numeric_var_list: List[str],
      outcome_var: str
  ) -> List[Union[pd.DataFrame, pd.Series]]:
    """모델링에 활용될 수 있는 형태의 train과 test 데이터셋에서
    [X_train(or X_trainval), X_test, y_train(or y_trainval), y_test] 데이터셋을 생성합니다.
    
    X: input 변수들, y: output 변수

    Returns:
        List[Union[pd.DataFrame, pd.Series]]: 모델링에 활용될 수 있는 train과 test 데이터셋의 X, y 데이터셋
    """
    train_data, test_data = self.load_raw_datasets()

    datasets = pd.concat(
        [
          train_data.assign(train_data_yn=1),
          test_data.assign(train_data_yn=0)
        ]
    )

    dummy_datasets = self.convert_into_dummy_coded_datasets(
        df=datasets,
        one_hot_var_list=one_hot_var_list,
        numeric_var_list=numeric_var_list
    )

    train_data = (
        dummy_datasets
        .query("train_data_yn == 1")
        .drop(columns=["train_data_yn"])
    )
    test_data = (
        dummy_datasets
        .query("train_data_yn == 0")
        .drop(columns=["train_data_yn"])
    )

    X_trainval = train_data.drop(columns=[outcome_var])
    y_trainval = train_data[outcome_var]

    X_test = test_data.drop(columns=[outcome_var])
    y_test = test_data[outcome_var]

    return [X_trainval, X_test, y_trainval, y_test]

  def make_msbos_x_y_datasets(
      self,
      msbos_var: str,
      outcome_var: str
  ) -> List[pd.Series]:
    train_data, test_data = self.load_raw_datasets()

    x_trainval = train_data[msbos_var]
    y_trainval = train_data[outcome_var]

    x_test = test_data[msbos_var]
    y_test = test_data[outcome_var]

    return [x_trainval, x_test, y_trainval, y_test]
