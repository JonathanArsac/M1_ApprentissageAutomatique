import numpy as np

from bayes import GaussianBayes
from utils import load_dataset, plot_scatter_hist


def main():
    train_data, train_labels = load_dataset("./data/train.csv")
    test_data, test_labels = load_dataset("./data/test.csv")

    # affichage
    ...

    # Instanciation de la classe GaussianB
    #g = GaussianBayes(priors=None)

    # Apprentissage
    #g.fit(train_data, train_labels)

    # Score
    #score = g.score(test_data, test_labels)
    #print("precision : {:.2f}".format(score))

    input("Press any key to exit...")


if __name__ == "__main__":
    main()
