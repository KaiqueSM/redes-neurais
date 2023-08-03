from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris = datasets.load_iris()

entries = iris.data
outings = iris.target

network = MLPClassifier(
    verbose=True,
    max_iter=1000,
    tol=0.00001,
    activation='logistic',
    learning_rate_init=0.001,
)


network.fit(entries, outings)
print(
    iris.target_names[
        network.predict(
            [[5, 7.2, 5.1, 2.2]]
        )[0]
    ]
)
