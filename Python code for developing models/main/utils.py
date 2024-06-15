import numpy as np
import scipy.stats as st

from sklearn.metrics import r2_score


def adjust_pred_value(x):
  """
  모델 예측값이 음수일 경우 0, 0 이상일 경우 반올림
  """
  x = np.where(x < 0, 0, x)
  x = np.round(x)

  return x


def get_adjusted_r2(true_vals, predicted_vals, num_of_vals):
  """
  adjusted r2 계산
  """
  adj_r2 = 1 - (1 - r2_score(true_vals, predicted_vals)) * (len(true_vals) - 1) / (len(true_vals) - num_of_vals - 1)

  return adj_r2


def get_95_conf_interval(x):
  """
  95% 신뢰구간 계산
  """
  lower, upper = st.t.interval(
      alpha=0.95,
      df=len(x) - 1,
      loc=np.mean(x),
      scale=st.sem(x)
  )
  return [np.mean(x), lower, upper]
