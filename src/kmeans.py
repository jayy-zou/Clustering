import numpy as np
import math

class KMeans():
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.means = None

        self.assignments = None
        self.features = None
        self.max_iterations = 500

    def new_assign(self):
        assignment= self.assignments.copy()

        for i in range(np.shape(self.features)[0]):
            distance = np.square(np.linalg.norm(self.features[i] - self.means[0]))
            assign_index = 0
            for j in range(1, np.shape(self.means)[0]):
                new_distance = np.square(np.linalg.norm(self.features[i] - self.means[j]))
                if new_distance <= distance:
                    distance = new_distance
                    assign_index = j

            assignment[i] = assign_index

        self.assignments = assignment

    def new_means(self):
        means = self.means.copy()

        for i in range(np.shape(means)[0]):
            means[i] = np.mean(self.features[np.argwhere(self.assignments == i)], axis=0)

        self.means = means


    def fit(self, features):
        self.features = features

        self.means = features[np.random.choice(range(np.shape(features)[0]), self.n_clusters)]

        self.assignments = np.ones((np.shape(features)[0]))

        self.new_assign()
        delta = self.new_assign()
        self.new_means()

        count = 0
        while not(delta==0) and count < self.max_iterations:
            self.new_assign()
            delta = self.new_assign()
            self.new_means()
            count += 1

    def predict(self, features):
        predictions = np.empty([])

        for i in range(np.shape(features)[0]):
            distance = np.square(np.linalg.norm(self.features[i] - self.means[0]))
            assign = 0
            for j in range(1, np.shape(self.means)[0]):
                new_distance=np.square(np.linalg.norm(self.features[i] - self.means[j]))
                if new_distance < distance:
                    distance = new_distance
                    assign = j

            predictions=np.append(predictions,assign)

        return np.delete(predictions,0)
