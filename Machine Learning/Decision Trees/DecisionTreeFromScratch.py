import pandas as pd
import numpy as np 

class DecisionTree:
    def __init__(self, data, label, depth=0, max_depth=5, min_data_split=2):
        self.data = data
        self.label = label
        self.depth = depth
        self.max_depth = max_depth
        self.min_data_split = min_data_split
        self.left = None
        self.right = None
        self.is_leaf = False
        self.predicted_label = None
        self.predictions = []  # Store predictions at each node
        self.feature = None
        self.threshold = None

    def __gini(self, labels):
        """Computes the Gini impurity for a set of labels."""
        if len(labels) == 0:
            return 0
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)  # class probabilities
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def BestSplit(self):
        """Finds the best feature and threshold to split on."""
        if len(self.data) == 0:
            return None, None

        best_feature = None
        best_threshold = None
        best_impurity = float('inf')

        for feature in self.data.columns:
            sorted_values = self.data[feature].sort_values().unique()
            mid_points = (sorted_values[:-1] + sorted_values[1:]) / 2

            for threshold in mid_points:
                left_mask = self.data[feature] <= threshold
                right_mask = self.data[feature] > threshold

                left_labels = self.label[left_mask]
                right_labels = self.label[right_mask]

                if len(left_labels) == 0 or len(right_labels) == 0:
                    continue

                impurity_left = self.__gini(left_labels)
                impurity_right = self.__gini(right_labels)
                weighted_impurity = ((len(left_labels) / len(self.label)) * impurity_left +
                                     (len(right_labels) / len(self.label)) * impurity_right)

                if weighted_impurity < best_impurity:
                    best_impurity = weighted_impurity
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            return None, None

        return {'feature': best_feature, 'threshold': best_threshold}, best_impurity

    def __prediction(self, predictedlabel, nodelabel, treedepth, nodegini):
        # Calculate ratio of 0s to 1s
        unique_labels, counts = np.unique(nodelabel, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))
        zero_one_ratio = label_counts.get(0, 0) / max(label_counts.get(1, 1), 1)  # Avoid division by zero

        # Create prediction dictionary with node metadata
        node_prediction = [{
            'predicted_label': predictedlabel,
            'depth': treedepth,
            'gini': nodegini,
            'zero_one_ratio': zero_one_ratio
        }]

        return node_prediction

    def Train(self):
        """Recursively trains the decision tree and stores predictions with metadata at each node."""
        # Early stopping conditions
        if self.depth >= self.max_depth or len(self.data) <= self.min_data_split:
            self.is_leaf = True
            self.predicted_label = self.label.mode().iloc[0]  # Most common label
            self.predictions = self.__prediction(self.predicted_label, self.label, self.depth, self.__gini(self.label))
            return

        best_split, _ = self.BestSplit()
        if best_split is None:
            self.is_leaf = True
            self.predicted_label = self.label.mode().iloc[0]
            self.predictions = self.__prediction(self.predicted_label, self.label, self.depth, self.__gini(self.label))
            return

        self.feature = best_split['feature']
        self.threshold = best_split['threshold']

        left_mask = self.data[self.feature] <= self.threshold
        right_mask = self.data[self.feature] > self.threshold

        left_data, left_label = self.data[left_mask], self.label[left_mask]
        right_data, right_label = self.data[right_mask], self.label[right_mask]

        # Check if split results in empty subset
        if len(left_data) == 0 or len(right_data) == 0:
            self.is_leaf = True
            self.predicted_label = self.label.mode().iloc[0]
            self.predictions = self.__prediction(self.predicted_label, self.label, self.depth, self.__gini(self.label))
            return

        # Create left and right children
        self.left = DecisionTree(left_data, left_label, self.depth + 1, self.max_depth, self.min_data_split)
        self.right = DecisionTree(right_data, right_label, self.depth + 1, self.max_depth, self.min_data_split)

        # Recursively train the children
        self.left.Train()
        self.right.Train()

        # Collect predictions from child nodes
        self.predictions = self.left.predictions + self.right.predictions

    def Predict(self, X):
        """Predict labels for input data X."""
        if self.is_leaf:
            return self.predictions

        predictions = []
        for _, row in X.iterrows():
            if row[self.feature] <= self.threshold:
                predictions.append(self.left.Predict(pd.DataFrame([row]))[0])
            else:
                predictions.append(self.right.Predict(pd.DataFrame([row]))[0])

        return predictions