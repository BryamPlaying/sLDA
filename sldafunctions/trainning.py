import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    RocCurveDisplay,
    roc_auc_score
)


class LogisticTrainer:
    """
    Clase para entrenar un modelo de Regresión Logística sobre todo el dataset,
    interpretar coeficientes (features relevantes), evaluar con métricas y gráficos,
    y predecir sobre texto nuevo. También genera nubes de palabras para cada género
    y permite salvar el mejor modelo entrenado en ./src/models.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "overview",
        genre_column: str = "genre",
        target_genre: str = None,
        valid_genres: Optional[set] = None,
        C: float = 1.0,
        max_features: Optional[int] = None,
        ngram_range: Tuple[int, int] = (1, 2),
        stop_words: str = 'english',
        random_state: int = 42
    ):
        """
        :param df: DataFrame con las columnas de texto y género.
        :param text_column: Nombre de la columna de texto.
        :param genre_column: Nombre de la columna de género (target).
        :param target_genre: Género de interés (1). Si es None, se toma el primer válido.
        :param valid_genres: Conjunto de géneros válidos.
        :param C: Valor de regularización para LogisticRegression.
        :param max_features: Número máximo de características para TF-IDF.
        :param ngram_range: Rango de n-gramas para TF-IDF.
        :param stop_words: Stopwords para TF-IDF vectorizer.
        :param random_state: Semilla para reproducibilidad.
        """
        self.df = df.reset_index(drop=True)
        self.text_column = text_column
        self.genre_column = genre_column

        # Definir géneros válidos
        if valid_genres is None:
            self.valid_genres = {
                "Drama", "Comedy", "Documentary", "Horror", "Thriller", "Western",
                "Action", "Animation", "Science Fiction", "Crime", "Music", "Adventure"
            }
        else:
            self.valid_genres = valid_genres

        # Determinar target_genre si no se pasa
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

        self.C = C
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words
        self.random_state = random_state

        # Etiquetas binarias (1 si genre == target_genre, 0 en caso contrario)
        self.y = (self.df[self.genre_column] == self.target_genre).astype(int).to_numpy()

        # Vectorizador TF-IDF (entrenado sobre TODO el dataset; se reentrena en evaluación)
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words
        )
        self.X_tfidf_full = self.vectorizer.fit_transform(
            self.df[self.text_column].astype(str)
        )

        # Instancia de LogisticRegression (no ajustada aún)
        self.model = LogisticRegression(
            C=self.C,
            solver="liblinear",
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000
        )

    def train_on_full_data(self, save_model: bool = False, model_name: str = "best_model.pkl") -> LogisticRegression:
        """
        Entrena el modelo de Regresión Logística sobre TODO el dataset vectorizado previamente.
        Si save_model=True, salva el modelo (pickle) en ./src/models/{model_name}.
        Devuelve la instancia entrenada.
        """
        # Ajustar el modelo con todos los datos
        self.model.fit(self.X_tfidf_full, self.y)

        # Guardar el modelo en disco si se solicita
        if save_model:
            os.makedirs("./src/models", exist_ok=True)
            modelo_path = os.path.join("./src/models", model_name)
            with open(modelo_path, "wb") as f:
                pickle.dump(self.model, f)

        return self.model

    def interpret_coefficients(self, top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retorna un diccionario {'positivas': [...], 'negativas': [...]}
        con las top_n palabras con coeficientes más positivos (asoc. a target_genre=1)
        y top_n más negativos (asoc. a clase 0).
        """
        if self.model.coef_.shape[0] != 1:
            raise RuntimeError("El modelo no está entrenado de forma binaria en una sola clase.")
        
        coef_array = self.model.coef_[0]
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # Ordenar índices por valor de coeficiente
        sorted_idx = np.argsort(coef_array)
        neg_idx = sorted_idx[:top_n]          # índices de coeficientes más negativos
        pos_idx = sorted_idx[-top_n:][::-1]   # índices de coeficientes más positivos

        negatives = [(feature_names[i], float(coef_array[i])) for i in neg_idx]
        positives = [(feature_names[i], float(coef_array[i])) for i in pos_idx]

        return {"positivas": positives, "negativas": negatives}

    def predict_on_new_texts(self, new_texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Para una lista de nuevos textos (reviews ficticias), vectoriza con el mismo TF-IDF
        y retorna (scores_continuos, probabilidades).
        """
        X_new = self.vectorizer.transform(new_texts)
        scores = self.model.decision_function(X_new)
        probs = self.model.predict_proba(X_new)[:, 1]
        return scores, probs

    def evaluate_model(
        self,
        test_size: float = 0.3,
        random_state: Optional[int] = None,
        save_results: bool = False
    ):
        """
        Separa el dataset en train/test, reentrena el vectorizador y el modelo en train,
        y muestra:
          - Matriz de confusión en test.
          - Curva ROC y AUC en test.

        Si save_results=True, guarda la figura en './src/validate_results/confusion_roc.png'.
        """
        rs = self.random_state if random_state is None else random_state

        # Crear carpeta para resultados si corresponde
        if save_results:
            os.makedirs("./src/validate_results", exist_ok=True)

        # División en train/test
        X = self.df[self.text_column].astype(str)
        y = self.y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=rs
        )

        # Re-ajustar vectorizador únicamente con train
        vect = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words
        )
        X_train_tfidf = vect.fit_transform(X_train)
        X_test_tfidf = vect.transform(X_test)

        # Re-entrenar modelo en train
        model = LogisticRegression(
            C=self.C,
            solver="liblinear",
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000
        )
        model.fit(X_train_tfidf, y_train)

        # Predecir en test
        y_pred = model.predict(X_test_tfidf)
        y_proba = model.predict_proba(X_test_tfidf)[:, 1]

        # Calcular ROC y AUC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)

        # Dibujar matriz de confusión + curva ROC lado a lado
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        disp_cm = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[0, 1]
        )
        disp_cm.plot(ax=axes[0], cmap="Blues", colorbar=False)
        axes[0].set_title("Confusion Matrix\n(Action vs Others)")

        # Curva ROC
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score, estimator_name="LogReg").plot(
            ax=axes[1]
        )
        axes[1].set_title(f"ROC Curve (AUC = {auc_score:.3f})")

        plt.tight_layout()

        # Guardar figura si se solicita
        if save_results:
            fig.savefig("./src/validate_results/confusion_roc.png")

        plt.show()

    def generate_wordclouds(self, save_results: bool = False):
        """
        Genera nubes de palabras (word clouds) para cada género en el DataFrame.
        Usa la columna de texto original y convierte a minúsculas en el momento.
    
        Si save_results=True, guarda cada imagen en './src/wordclouds/<Genero>_wordcloud.png'
        y NO muestra las imágenes en pantalla, pero informa en consola dónde encontrarlas.
        """
        # Crear carpeta para wordclouds si corresponde
        if save_results:
            os.makedirs("./src/wordclouds", exist_ok=True)
    
        all_genres = self.df[self.genre_column].unique()
        stopwords = set(STOPWORDS)
        n_rows = int(np.ceil(len(all_genres) / 2))
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=2,
            figsize=(12, 6 * n_rows)
        )
        axes = axes.flatten()
    
        for i, genre in enumerate(all_genres):
            texts = (
                self.df
                .loc[self.df[self.genre_column] == genre, self.text_column]
                .astype(str)
                .str.lower()
            )
            combined_text = " ".join(texts.tolist())
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                stopwords=stopwords
            ).generate(combined_text)
    
            axes[i].imshow(wordcloud, interpolation="bilinear")
            axes[i].set_title(f"Word Cloud: {genre}", fontsize=14)
            axes[i].axis("off")
    
            # Guardar cada wordcloud si se solicita
            if save_results:
                wc_path = f"./src/wordclouds/{genre}_wordcloud.png"
                wordcloud.to_file(wc_path)
                print(f"Word cloud de '{genre}' guardada en: {wc_path}")
    
        # Ocultar subplots sobrantes (si número de géneros es impar)
        for j in range(len(all_genres), len(axes)):
            axes[j].axis("off")
    
        plt.tight_layout()
    
        if save_results:
            # Si estamos guardando, cerramos la figura para que no se muestre en el notebook
            plt.close(fig)
        else:
            # Si NO estamos guardando, mostramos la figura en pantalla
            plt.show()