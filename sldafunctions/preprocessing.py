import ast
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
import pandas as pd

class MoviePreprocessor:
    """
    Clase para preprocesar un DataFrame de películas con las siguientes etapas:
      1. Eliminar películas con overview vacío o nulo.
      2. Convertir la columna 'genres' (string JSON-like) en lista de nombres de géneros.
      3. Filtrar películas que tengan EXACTAMENTE un solo género válido.
      4. Crear una nueva columna 'genre' que contenga únicamente ese nombre de género.
      5. Conservar únicamente aquellas sinopsis que, al fragmentarse con
         SentenceTransformersTokenTextSplitter, resulten en un único bloque de texto.
    """

    def __init__(
        self,
        genres_column: str = "genres",
        overview_column: str = "overview",
        valid_genres: list = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        :param genres_column: Nombre de la columna que contiene el string JSON-like de géneros 
                              (lista de dicts con claves 'id' y 'name').
        :param overview_column: Nombre de la columna que contiene la sinopsis.
        :param valid_genres: Lista de géneros válidos (si no se pasa, se usa la lista por defecto).
        :param chunk_size: Número de tokens por fragmento para el splitter.
        :param chunk_overlap: Número de tokens de solapamiento entre fragmentos.
        :param model_name: Nombre del modelo SentenceTransformers para el tokenizador.
        """
        if valid_genres is None:
            # Lista por defecto de géneros válidos
            self.valid_genres = {
                "Drama", "Comedy", "Documentary", "Horror", "Thriller", "Western",
                "Action", "Animation", "Science Fiction", "Crime", "Music", "Adventure"
            }
        else:
            self.valid_genres = set(valid_genres)

        self.genres_column = genres_column
        self.overview_column = overview_column

        # Instanciar el splitter basado en SentenceTransformers
        self.splitter = SentenceTransformersTokenTextSplitter(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas las etapas de preprocesamiento al DataFrame:
          1. Filtrado de sinopsis vacías.
          2. Parseo de 'genres' a lista de nombres.
          3. Filtrado por “exactamente un género válido”.
          4. Construcción de columna 'genre' con el nombre del género único.
          5. Filtrado por sinopsis unificada (1 solo bloque de texto).
        Devuelve un nuevo DataFrame con las filas que cumplen todos los criterios,
        y añade la columna 'genre' con el string del género.
        """
        df = df.copy()

        # Eliminar filas con overview vacío o nulo
        df = self._filter_empty_overview(df)

        # Crear columna auxiliar 'genres_list' que contenga la lista de nombres de géneros
        df["genres_list"] = df[self.genres_column].apply(self._parse_genres)

        # Filtrar por “exactamente un género válido”
        df = self._filter_single_valid_genre(df)

        # Crear nueva columna 'genre' con el único género válido
        df["genre"] = df["genres_list"].apply(lambda lst: lst[0])

        # Ya no necesitamos 'genres_list' para el resto de pipeline
        df = df.drop(columns=["genres_list", 'genres'])

        # Filtrar sinopsis que, al usar el splitter, se fragmenten en un solo bloque
        df = self._filter_unified_overview(df)

        return df.reset_index(drop=True)

    def _filter_empty_overview(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Elimina filas donde el overview sea None, NaN o una cadena vacía.
        """
        mask = (
            df[self.overview_column].notna()
            & (df[self.overview_column].astype(str).str.strip() != "")
        )
        return df.loc[mask]

    def _parse_genres(self, genres_value) -> list:
        """
        Convierte el valor de la columna de géneros (que llega como string) 
        en una lista de nombres de géneros. Maneja casos:
          - Si viene como string que describe una lista de dicts, usa ast.literal_eval.
          - Si literal_eval produce algo distinto a una lista, ignora y retorna [].
          - Extrae 'name' de cada dict. Si el literal no es dicts, tampoco lo considera.
          - Si no puede parsear, hace un fallback separando por comas.
        """
        if isinstance(genres_value, list):
            # Si ya fuera lista en memoria (caso excepcional), extraer nombres
            if all(isinstance(ele, dict) and "name" in ele for ele in genres_value):
                return [ele["name"] for ele in genres_value]
            elif all(isinstance(ele, str) for ele in genres_value):
                return [ele.strip() for ele in genres_value if ele.strip()]
            else:
                return []

        if isinstance(genres_value, str):
            txt = genres_value.strip()
            # Si parece una lista JSON-like, intentar literal_eval
            if txt.startswith("[") and txt.endswith("]"):
                try:
                    parsed = ast.literal_eval(txt)
                    if isinstance(parsed, list):
                        # Si es lista de dicts con key 'name'
                        if all(isinstance(ele, dict) and "name" in ele for ele in parsed):
                            return [ele["name"] for ele in parsed]
                        # Si es lista de strings
                        elif all(isinstance(ele, str) for ele in parsed):
                            return [ele.strip() for ele in parsed if ele.strip()]
                        # Si no cumple esas condiciones, retornamos vacío
                        return []
                except (ValueError, SyntaxError):
                    # Si falla literal_eval, cae a fallback
                    pass
            # Fallback: separar por comas si no pudo parsear como lista
            return [g.strip() for g in txt.split(",") if g.strip()]

        # Si no es ni lista ni string
        return []

    def _filter_single_valid_genre(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Conserva únicamente filas donde, tras parsear la columna 'genres',
        la lista auxiliar 'genres_list' tenga EXACTAMENTE 1 elemento,
        y ese único elemento pertenezca a self.valid_genres.
        """
        def is_one_valid_genre(names_list: list) -> bool:
            return (isinstance(names_list, list) and len(names_list) == 1
                    and (names_list[0] in self.valid_genres))

        mask = df["genres_list"].apply(is_one_valid_genre)
        return df.loc[mask]

    def _filter_unified_overview(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Conserva únicamente aquellas filas cuya sinopsis,
        al fragmentarse con SentenceTransformersTokenTextSplitter,
        produce exactamente un bloque de texto.
        """
        unified_indices = []
        for idx, texto in df[self.overview_column].items():
            fragments = self.splitter.split_text(str(texto))
            if len(fragments) == 1:
                unified_indices.append(idx)

        return df.loc[unified_indices]
