import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


class LogisticValidator:
    """
    Clase para validar un modelo de Regresión Logística aplicando CV sobre
    diferentes valores de regularización (C) y evaluando ROC-AUC.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "overview",
        genre_column: str = "genre",
        target_genre: str = None,
        valid_genres: Optional[set] = None,
        C_values: List[float] = None,
        cv_folds: int = 5,
        random_state: int = 42,
        max_features: Optional[int] = None,
        ngram_range: Tuple[int, int] = (1, 1)
    ):
        """
        :param df: DataFrame con las columnas de texto y género.
        :param text_column: Nombre de la columna que contiene el texto (overview).
        :param genre_column: Nombre de la columna que contiene el género único.
        :param target_genre: Género que queremos predecir (1). El resto será 0.
                             Si es None, se toma el primer género válido encontrado.
        :param valid_genres: Conjunto de géneros válidos. Si es None, se usa un conjunto por defecto.
        :param C_values: Lista de valores de C (inverso de regularización) para probar en CV.
                         Si es None, se usa [0.01, 0.1, 1, 10, 100].
        :param cv_folds: Número de folds para la validación cruzada (por defecto 5).
        :param random_state: Semilla para reproducibilidad en KFold y LogisticRegression.
        :param max_features: Número máximo de características para el TF-IDF. Si es None, toma todas.
        :param ngram_range: Tupla (min_n, max_n) para n-gramas en TF-IDF.
        """
        self.df = df.reset_index(drop=True)
        self.text_column = text_column
        self.genre_column = genre_column

        # Géneros válidos por defecto
        if valid_genres is None:
            self.valid_genres = {
                "Drama", "Comedy", "Documentary", "Horror", "Thriller", "Western",
                "Action", "Animation", "Science Fiction", "Crime", "Music", "Adventure"
            }
        else:
            self.valid_genres = valid_genres

        # Determinar target_genre si no se provee
        if target_genre is None:
            found = [
                g for g in sorted(self.valid_genres)
                if g in self.df[self.genre_column].unique()
            ]
            if not found:
                raise ValueError("No se encontró ningún género válido en el DataFrame.")
            self.target_genre = found[0]
        else:
            if target_genre not in self.valid_genres:
                raise ValueError(f"target_genre '{target_genre}' no está en valid_genres.")
            self.target_genre = target_genre

        # Etiquetas binarias: 1 si genre == target, 0 en caso contrario
        self.labels = (self.df[self.genre_column] == self.target_genre).astype(int).to_numpy()

        # Parámetros TF-IDF
        self.max_features = max_features
        self.ngram_range = ngram_range

        # Valores de C a evaluar
        self.C_values = C_values if C_values is not None else [0.01, 0.1, 1.0, 10.0, 100.0]
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Precomputar matriz TF-IDF sobre todo el dataset
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range
        )
        self.X_tfidf = self._vectorizer.fit_transform(self.df[self.text_column].astype(str))

    def cross_validate_C(self) -> Dict[float, float]:
        """
        Realiza validación cruzada para cada valor de C en self.C_values, 
        usando KFold en self.cv_folds. Retorna un diccionario: {C: AUC_promedio}.
        """
        results: Dict[float, float] = {}
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for C in self.C_values:
            aucs = []
            for train_idx, val_idx in kf.split(self.X_tfidf):
                X_train = self.X_tfidf[train_idx]
                y_train = self.labels[train_idx]
                X_val = self.X_tfidf[val_idx]
                y_val = self.labels[val_idx]

                model = LogisticRegression(
                    C=C,
                    solver="liblinear",
                    random_state=self.random_state,
                    max_iter=1000
                )
                model.fit(X_train, y_train)
                y_scores = model.predict_proba(X_val)[:, 1]
                # Si y_val es todo 0 o todo 1, roc_auc_score falla; en ese caso, AUC = 0.5
                try:
                    auc = roc_auc_score(y_val, y_scores)
                except ValueError:
                    auc = 0.5
                aucs.append(auc)

            avg_auc = float(np.mean(aucs))
            results[C] = avg_auc
            print(f"C={C}: AUC promedio {self.cv_folds}-fold = {avg_auc:.4f}")

        return results

    def evaluate_final_model(self, best_C: float, cv_folds: int = 10) -> float:
        """
        Con el valor best_C, evalúa el modelo final usando CV de cv_folds pliegos
        retornando el AUC promedio. 
        """
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state + 1)
        aucs = []

        for train_idx, val_idx in kf.split(self.X_tfidf):
            X_train = self.X_tfidf[train_idx]
            y_train = self.labels[train_idx]
            X_val = self.X_tfidf[val_idx]
            y_val = self.labels[val_idx]

            model = LogisticRegression(
                C=best_C,
                solver="liblinear",
                random_state=self.random_state,
                max_iter=1000
            )
            model.fit(X_train, y_train)
            y_scores = model.predict_proba(X_val)[:, 1]
            try:
                auc = roc_auc_score(y_val, y_scores)
            except ValueError:
                auc = 0.5
            aucs.append(auc)

        avg_auc = float(np.mean(aucs))
        print(f"Evaluación final con {cv_folds}-fold CV y C={best_C} → AUC promedio = {avg_auc:.4f}")
        return avg_auc