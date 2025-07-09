import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from typing import Tuple
from .ml_model import LogisticRegressionReranker

class RerankerTrainer:
    def __init__(self, model=None, test_size=0.2, random_state=42):
        self.model = model or LogisticRegressionReranker()
        self.test_size = test_size
        self.random_state = random_state

    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, LogisticRegressionReranker]:
        # Разбиваем на train и val
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Обучаем модель
        self.model.train(X_train, y_train)

        # Предсказываем вероятности на валидации
        y_pred_proba = self.model.predict(X_val)

        # Считаем ROC-AUC
        auc = roc_auc_score(y_val, y_pred_proba)
        print(f"Validation ROC-AUC: {auc:.4f}")

        return auc, self.model
