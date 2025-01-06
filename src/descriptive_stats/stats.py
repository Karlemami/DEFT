import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from preprocessing.DataLoader import DataLoader
from collections import Counter


SAVED_MODELS_PATH = "saved_models"
TRAIN_PATH = "data/deft09_parlement_appr"
TEST_PATH = "data/deft09_parlement_test"
LANGUAGE = "fr"



def get_corpus_numbers() -> dict:
    corpus = {}

    loader = DataLoader(TRAIN_PATH, TEST_PATH, LANGUAGE)
    normal = loader.df
    drop_dup = loader.df_unique

    for key, item in {"normal": normal, "drop_dup": drop_dup}.items():
        corpus[key] = {"total": len(item)}
        test = item[item["split"] == "test"]
        train = item[item["split"] == "train"]

        for split_k, split_df in {"test": test, "train": train}.items():
            corpus[key][split_k] = {}
            labels = dict(Counter(split_df["y"].to_list()))

            for label, nb in labels.items():
                corpus[key][split_k][label] = nb

    return corpus


def save_json(corpus: dict):
    with open("data.json", "w") as fp:
        json.dump(corpus, fp)
    return print("fichier json sauvé à data.json")


def load_json() -> dict:
    with open("stats/data.json", "r") as fp:
        data = json.load(fp)
    return data


def make_bar_plot(corpus):
    """Créé et sauvegarde un diagramme en barre qui affiche le nombre d'interventions
    par parti par partition (train/test) et par version du corpus 
    (corpus d'origine vs corpus sans doublons."""
     # la largueur des barres et les espacements ont été ajustés à taton pour
     # rendre le diagramme + lisible et éviter les chevauchements
     # des couleurs personnifiées ont été ajoutées dans l'espoir de rendre le
     # plot plus sympa et surtout plus facile à lire (beaucoup d'info au même
     # endroit !)
     
    partis = list(corpus["normal"]["train"].keys())
    categories = ["train", "test"]
    x = np.arange(len(partis))
    width = 0.17
    fig, ax = plt.subplots()
    drop_dup_colors = ["darkblue", "lightblue"]
    normal_colors = ["darkorange", "orange"]

    for i, cat in enumerate(categories):
        values_normal = [corpus["normal"][cat].get(parti, 0) for parti in partis]
        offset_normal = i * 2 * width
        ax.bar(
            x + offset_normal,
            values_normal,
            width,
            label=f"original {cat.capitalize()}",
            align="center",
            color=normal_colors[i],
        )
        values_drop_dup = [corpus["drop_dup"][cat].get(parti, 0) for parti in partis]
        offset_drop_dup = (i * 2 * width) + width
        ax.bar(
            x + offset_drop_dup,
            values_drop_dup,
            width,
            label=f"sans doublon {cat.capitalize()}",
            align="center",
            color=drop_dup_colors[i],
        )

    ax.set_xlabel("Partis politiques")
    ax.set_ylabel("Nombre d'occurrences")
    ax.set_title("Occurrences des partis politiques (corpus d'origine et corpus sans doublons)")
    ax.set_xticks(x + width * (3 / 2))
    ax.set_xticklabels(partis)
    ax.legend()
    plt.tight_layout()
    plt.savefig("stats/occurences_orig_vs_drop_dup_par_cat.png")

    return print("le diagramme en barre a été enregistré dans le dossier stats")


def make_camember_plot(corpus: dict):
    labels = [
        tuple[0]
        for tuple in sorted(corpus["drop_dup"]["train"].items(), key=lambda x: x[1])
    ]
    sizes_train = [
        tuple[1]
        for tuple in sorted(corpus["drop_dup"]["train"].items(), key=lambda x: x[1])
    ]
    sizes_test = [
        tuple[1]
        for tuple in sorted(corpus["drop_dup"]["test"].items(), key=lambda x: x[1])
    ]

    for nom, sizes in {"train": sizes_train, "test": sizes_test}.items():
        plt.figure()
        plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        plt.title(f"Répartition des partis politiques dans le corpus {nom}")
        plt.savefig(f"stats/occurences_par_partis_{nom}_camember.png")

    return print("les deux camembers sont enregistrés dans le dossier stats")


def make_simple_stats(corpus: dict):
    train = {"nb intervention par parti": corpus["drop_dup"]["train"].values()}
    test = {"nb intervention par parti": corpus["drop_dup"]["test"].values()}

    for nom, data in {"train": train, "test": test}.items():
        df = pd.DataFrame(data)
        stats = df.describe().drop("count")
        stats.to_csv(f"stats/statistiques_descriptives_drop_dup_{nom}.csv")
    return print("les deux csv sont dans le dossier stats")


def main():
    # corpus = get_corpus_numbers()
    # save_json(corpus)
    corpus = load_json()
    make_bar_plot(corpus)
    make_camember_plot(corpus)
    make_simple_stats(corpus)
    return print("Fin du script de statistiques descriptives")


if __name__ == "__main__":
    main()