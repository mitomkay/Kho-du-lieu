# from main import DecisionTreeID3
import numpy as np
from DecisionTreeID3 import *

class RandomForest(object):
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, min_gain=1e-4):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.trees = []

    def fit(self, data, target):
        for _ in range(self.n_trees):
            tree = DecisionTreeID3(max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_gain=self.min_gain)
            # Tạo mẫu ngẫu nhiên từ tập dữ liệu
            indices = np.random.choice(len(data), size=len(data), replace=True)
            sampled_data = data.iloc[indices]
            sampled_target = target.iloc[indices]
            # Xây dựng cây quyết định trên mẫu
            tree.fit(sampled_data, sampled_target)
            self.trees.append(tree)

    def predict(self, new_data):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(new_data))
        # Bầu cử đa số để đưa ra dự đoán
        final_predictions = {}
        for i in new_data.index:
            votes = {}
            for j in range(len(predictions)):
                vote = predictions[j][i]
                if vote in votes:
                    votes[vote] += 1
                else:
                    votes[vote] = 1
            final_predictions[i] = max(votes, key=votes.get)
        return final_predictions