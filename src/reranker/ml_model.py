import os
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

class BaseRerankModel(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Обучить модель"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> List[float]:
        """Вернуть прогнозы вероятностей или скорингов"""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Сохранить модель на диск"""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Загрузить модель с диска"""
        pass

class LogisticRegressionReranker(BaseRerankModel):
    def __init__(self):
        self.model = LogisticRegression(solver='liblinear')

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> List[float]:
        probs = self.model.predict_proba(X)
        # Вернуть вероятность положительного класса (класс 1)
        return probs[:, 1].tolist()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)
