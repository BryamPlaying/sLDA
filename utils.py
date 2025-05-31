import pandas as pd
import numpy as np
from tabulate import tabulate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import List, Tuple, Optional

def evaluate_reviews(
    trainer,
    custom_reviews: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Predice una serie de reviews (texto y género real) usando el modelo entrenado en 'trainer',
    muestra una tabla con tabulate (review_text, real_genre, predicted_genre), dibuja la matriz
    de confusión (Action vs Other) y un gráfico de barras con hue=real vs predicted.

    Por defecto, incluye 5 reviews para cada género (todas las que estén en la lista). 
    Si se pasan custom_reviews, se concatenan al final y también se procesan.

    :param trainer: Instancia de LogisticTrainer ya entrenada.
    :param custom_reviews: Lista de tuplas (texto, género_real). Si no se pasa, se usan todas 
                           las reviews definidas en default_reviews.
    :return: DataFrame con columnas [review_text, real_genre, predicted_genre].
    """
    default_reviews = [
        # Action (5)
        ("An explosive action film with thrilling car chases and high-octane shootouts.", "Action"),
        ("A team of elite soldiers infiltrates an enemy stronghold to prevent a global catastrophe.", "Action"),
        ("A martial artist trains under a legendary master to avenge his family’s honor.", "Action"),
        ("A rogue agent must race against time to stop a nuclear device hidden in plain sight.", "Action"),
        ("An undercover cop battles a crime lord in a series of rooftop brawls and gunfights.", "Action"),
    
        # Drama (5)
        ("A poignant drama about a family struggling to keep their farm after years of drought.", "Drama"),
        ("A talented musician returns home and reconciles with estranged relatives.", "Drama"),
        ("An intense courtroom battle reveals dark secrets buried for decades.", "Drama"),
        ("A heart-wrenching story of a young couple navigating love and loss during wartime.", "Drama"),
        ("A biographical drama chronicling the rise and fall of a political leader.", "Drama"),
    
        # Comedy (5)
        ("A hilarious comedy about two mismatched roommates who accidentally steal a priceless artifact.", "Comedy"),
        ("A witty romantic comedy where a bookstore owner falls for a clumsy delivery driver.", "Comedy"),
        ("A slapstick comedy following the misadventures of a novice magician in Las Vegas.", "Comedy"),
        ("A mockumentary-style film about a dysfunctional family starting a backyard food truck.", "Comedy"),
        ("A road-trip comedy featuring three best friends on a quest to find the world’s largest pizza.", "Comedy"),
    
        # Documentary (5)
        ("A thought-provoking documentary exploring the decline of coral reefs in the Pacific Ocean.", "Documentary"),
        ("An in-depth look at the daily lives of firefighters battling wildfires in California.", "Documentary"),
        ("A historical documentary tracing the origins of the Silk Road trade routes.", "Documentary"),
        ("An environmental documentary about the efforts to save endangered tiger populations.", "Documentary"),
        ("A cultural documentary documenting the rituals and traditions of a remote mountain village.", "Documentary"),
    
        # Horror (5)
        ("A chilling horror film set in an abandoned asylum where the walls whisper secrets.", "Horror"),
        ("A supernatural horror about a family tormented by a vengeful spirit in their new home.", "Horror"),
        ("A slasher horror where a masked killer stalks college students during spring break.", "Horror"),
        ("A psychological horror delving into the mind of a man who hears voices in the dark.", "Horror"),
        ("A creature-feature horror where deep-sea divers encounter a monstrous leviathan.", "Horror"),
    
        # Thriller (5)
        ("A tense thriller about a journalist uncovering a corrupt government conspiracy.", "Thriller"),
        ("A psychological thriller where a detective must solve a series of cryptic murders.", "Thriller"),
        ("A techno-thriller involving a hacker racing to prevent a massive cyberattack.", "Thriller"),
        ("A hostage thriller where a bank teller is trapped with armed robbers and must find a way out.", "Thriller"),
        ("A political thriller about an undercover agent infiltrating a terrorist cell.", "Thriller"),
    
        # Western (5)
        ("A classic western about a lone gunslinger defending a frontier town from outlaws.", "Western"),
        ("A gritty western following a bounty hunter tracking a notorious bandit gang.", "Western"),
        ("A family western set on a cattle ranch struggling to survive during a brutal winter.", "Western"),
        ("A revenge western where a former sheriff returns to his haunted hometown for justice.", "Western"),
        ("A western adventure about settlers forging a new life along the wagon trail.", "Western"),
    
        # Animation (5)
        ("A heartwarming animation about a little mouse and his unlikely friendship with a cat.", "Animation"),
        ("A colorful animation where animals embark on a quest to save their enchanted forest.", "Animation"),
        ("A whimsical animation following a young wizard’s first day at a magical academy.", "Animation"),
        ("A family animation about a group of insects working together to build the world’s tallest castle.", "Animation"),
        ("A stop-motion animation featuring a robot discovering human emotions in a post-apocalyptic world.", "Animation"),
    
        # Science Fiction (5)
        ("A science fiction epic about a crew exploring a distant planet threatened by alien invaders.", "Science Fiction"),
        ("A dystopian sci-fi where a rebel fights against a totalitarian regime controlling minds.", "Science Fiction"),
        ("A cyberpunk sci-fi set in a neon-lit city run by powerful megacorporations.", "Science Fiction"),
        ("A hard sci-fi film focusing on astronauts stranded on Mars trying to survive.", "Science Fiction"),
        ("A time-travel sci-fi where scientists race to prevent a paradox that could destroy reality.", "Science Fiction"),
    
        # Crime (5)
        ("A gritty crime drama following a detective unraveling a series of bank heists.", "Crime"),
        ("A noir crime story about a private investigator caught in a web of deception.", "Crime"),
        ("A crime thriller detailing a meticulous cat-and-mouse game between a cop and a criminal mastermind.", "Crime"),
        ("An ensemble crime film where rival gangs clash in a bid for control of the city.", "Crime"),
        ("A biographical crime film about the rise and fall of a notorious mafia kingpin.", "Crime"),
    
        # Music (5)
        ("A musical drama chronicling the life of a jazz singer in 1920s New Orleans.", "Music"),
        ("A concert documentary featuring behind-the-scenes footage of a legendary rock band.", "Music"),
        ("A music biopic about a classical composer struggling for recognition in 19th-century Europe.", "Music"),
        ("A feel-good music film following a high school choir preparing for the national championship.", "Music"),
        ("A music-driven road movie about a folk singer traveling across America to find her roots.", "Music"),
    
        # Adventure (5)
        ("An adventure film about explorers trekking through the Amazon jungle in search of a lost city.", "Adventure"),
        ("A high-seas adventure where pirates hunt for a mythical treasure on a haunted island.", "Adventure"),
        ("An epic adventure chronicling a young hero’s journey across a magical realm to reclaim a kingdom.", "Adventure"),
        ("A survival adventure about a group of climbers trapped on Everest during a deadly storm.", "Adventure"),
        ("A family adventure following siblings who discover a hidden portal to a fantastical world.", "Adventure")
    ]

    # Si se pasan reviews personalizadas, concatenar
    reviews_list = default_reviews.copy()
    if custom_reviews:
        reviews_list.extend(custom_reviews)
    
    # Convertir a DataFrame y predecir
    df_reviews = pd.DataFrame(reviews_list, columns=["review_text", "real_genre"])
    texts_lower = df_reviews["review_text"].str.lower().tolist()
    scores, probs = trainer.predict_on_new_texts(texts_lower)
    df_reviews["predicted_genre"] = np.where(probs >= 0.5, trainer.target_genre, "Other")
    
    # Imprimir tabla con tabulate en formato “fancy_grid”
    table = tabulate(
        df_reviews[["review_text", "real_genre", "predicted_genre"]],
        headers=["Review Text", "Real Genre", "Predicted Genre"],
        tablefmt="fancy_grid",
        showindex=False
    )
    # print(table)
    
    # Matriz de confusión (Action vs Other)
    y_true = (df_reviews["real_genre"] == trainer.target_genre).astype(int)
    y_pred = (df_reviews["predicted_genre"] == trainer.target_genre).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[f"{trainer.target_genre} (True)", "Other (True)"]
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Confusion Matrix: Action vs Other")
    
    # Gráfico de barras: conteo por real_genre y hue=predicted_genre
    sns.countplot(
        data=df_reviews,
        x="real_genre",
        hue="predicted_genre",
        ax=axes[1],
        palette="Set2"
    )
    axes[1].set_title("Cantidad por Género Real vs Predicho")
    axes[1].set_xlabel("Género Real")
    axes[1].set_ylabel("Recuento")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    
    plt.tight_layout()
    plt.show()
    
    # Retornar el DataFrame con predicciones
    return df_reviews