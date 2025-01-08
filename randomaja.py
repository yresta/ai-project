import streamlit as st
from joblib import load
import numpy as np
import random

# Fungsi untuk memuat model
def load_model(filename):
    return load(filename)

# Kelas DecisionTree (untuk Decision Tree klasik)
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        unique_classes = np.unique(y)

        # Jika hanya ada satu kelas atau mencapai kedalaman maksimum, buat simpul daun
        if len(unique_classes) == 1 or (self.max_depth and depth >= self.max_depth):
            return unique_classes[0]

        # Pemilihan fitur terbaik untuk pemisahan
        best_split = self._best_split(X, y, n_features)
        left_tree = self._build_tree(*best_split['left'], depth + 1)
        right_tree = self._build_tree(*best_split['right'], depth + 1)

        return {'feature': best_split['feature'], 'value': best_split['value'], 'left': left_tree, 'right': right_tree}

    def _best_split(self, X, y, n_features):
        best_split = {'gini': float('inf')}
        best_left = best_right = None
        best_feature = best_value = None

        features = random.sample(range(n_features), n_features)  # Pilih fitur secara acak
        for feature in features:
            values = np.unique(X[:, feature])
            for value in values:
                left_mask = X[:, feature] <= value
                right_mask = ~left_mask
                left_y = y[left_mask]
                right_y = y[right_mask]

                # Hitung impuritas Gini
                gini_left = self._gini_impurity(left_y)
                gini_right = self._gini_impurity(right_y)
                gini = (len(left_y) * gini_left + len(right_y) * gini_right) / len(y)

                if gini < best_split['gini']:
                    best_split['gini'] = gini
                    best_left = (X[left_mask], left_y)
                    best_right = (X[right_mask], right_y)
                    best_feature = feature
                    best_value = value

        return {'gini': best_split['gini'], 'left': best_left, 'right': best_right, 'feature': best_feature, 'value': best_value}

    def _gini_impurity(self, y):
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)
        return 1 - np.sum(prob ** 2)

    def predict(self, X):
        return [self._predict_row(row, self.tree) for row in X]

    def _predict_row(self, row, tree):
        if isinstance(tree, dict):
            if row[tree['feature']] <= tree['value']:
                return self._predict_row(row, tree['left'])
            else:
                return self._predict_row(row, tree['right'])
        return tree

# Kelas RandomForest
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Bootstrapping (ambil sampel acak dengan pengembalian)
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        # Prediksi dengan mayoritas suara dari semua pohon
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return [self._majority_vote(tree_preds[:, i]) for i in range(len(X))]

    def _majority_vote(self, predictions):
        return np.bincount(predictions).argmax()

# Input fitur dari pengguna
age = st.number_input("Age", min_value=0)
relationships = st.number_input("Relationships", min_value=0)
age_last_milestone_year = st.number_input("Age Last Milestone Year", min_value=0)
milestones = st.number_input("Milestones", min_value=0)
is_top500 = st.number_input("Is Top 500", min_value=0, max_value=1)
has_RoundABCD = st.number_input("Has Round ABCD", min_value=0, max_value=1)
age_first_milestone_year = st.number_input("Age First Milestone Year", min_value=0)
funding_rounds = st.number_input("Funding Rounds", min_value=0)
avg_participants = st.number_input("Avg Participants", min_value=0)
is_otherstate = st.number_input("Is Other State", min_value=0, max_value=1)

# Membuat array fitur
selected_features = np.array([age, relationships, age_last_milestone_year, milestones, is_top500,
                              has_RoundABCD, age_first_milestone_year, funding_rounds, avg_participants, is_otherstate]).reshape(1, -1)

# Tambahkan tombol prediksi
predict_button = st.button("Prediksi")

# Prediksi berdasarkan algoritma (Random Forest) hanya jika tombol ditekan
if predict_button:
    model = load_model("random_forest_model.joblib")
    if model:
        # Menggunakan .predict() untuk model Random Forest
        prediction = model.predict(selected_features)
        st.write(f"Prediksi (Random Forest): {prediction[0]}")

        # Tampilkan hasil prediksi
        if prediction[0] == 1:
            st.write("Prediksi: Sukses")
        else:
            st.write("Prediksi: Gagal")
