# author: Benjamin Chamand, Thomas Pelligrini

from kmeans import KMeans
from utils import load_dataset


def main():
    filepath = "./data/self_test.csv"
    #filepath = "./data/iris.csv"

    # chargement des donnees
    data, labels = load_dataset(filepath)

    # initialisation de l'objet KMeans
    kmeans = KMeans(n_clusters=5,
                    max_iter=100,
                    early_stopping=True,
                    tol=1e-6,
                    display=True)

    # calcule les clusters
    kmeans.fit(data)

    # calcule la purete de nos clusters
    score = kmeans.score(data, labels)
    print("Purete : {}".format(score))

    input("Press any key to exit...")


if __name__ == "__main__":
    main()
