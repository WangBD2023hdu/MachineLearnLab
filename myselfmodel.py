from collections import Counter
import math
# implementation of Linear Regression


# Matrix operations
def matrix_transpose(matrix):
    return list(map(list, zip(*matrix)))

def matrix_multiply(A, B):
    return [[sum(a*b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

def identity_matrix(size):
    return [[float(i == j) for i in range(size)] for j in range(size)]

def matrix_inverse(matrix):
    n = len(matrix)
    identity = identity_matrix(n)
    augmented_matrix = [row + identity_row for row, identity_row in zip(matrix, identity)]

    # Applying Gaussian elimination
    for i in range(n):
        # Make the diagonal contain all 1's
        diag_factor = augmented_matrix[i][i]
        for j in range(2 * n):
            augmented_matrix[i][j] /= diag_factor
        
        # Make the other elements in the current column 0
        for k in range(n):
            if k != i:
                factor = augmented_matrix[k][i]
                for j in range(2 * n):
                    augmented_matrix[k][j] -= factor * augmented_matrix[i][j]

    inverse_matrix = [row[n:] for row in augmented_matrix]
    return inverse_matrix

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        # 将 X 转换为包含偏置项的矩阵
        X = self._add_intercept(X)
        
        # 使用最小二乘法计算回归系数
        # beta = (X^T X)^(-1) X^T y
        X_transpose = self._transpose(X)
        X_transpose_X = self._matmul(X_transpose, X)
        X_transpose_y = self._matmul(X_transpose, y)
        beta = self._matmul(self._inverse(X_transpose_X), X_transpose_y)
        
        self.intercept = beta[0][0]
        self.coefficients = [b[0] for b in beta[1:]]

    def predict(self, X):
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not trained yet. Please call the fit method first.")
        
        X = self._add_intercept(X)
        predictions = self._matmul(X, [[self.intercept]] + [[coef] for coef in self.coefficients])
        return [pred[0] for pred in predictions]

    def _add_intercept(self, X):
        # 在 X 的第一列添加全为1的偏置项
        return [[1.0] + row for row in X]

    def _transpose(self, matrix):
        return list(map(list, zip(*matrix)))

    def _matmul(self, A, B):
        # 矩阵乘法
        if type(A[0]) is not list:
            A = list(map(lambda x:[x], A))
        if type(B[0]) is not list:
            B = list(map(lambda x:[x], B))
            # 获取矩阵 A 和矩阵 B 的行数和列数
        return matrix_multiply(A, B)

    def _inverse(self, matrix):
        """
        矩阵求逆
        """
        return matrix_inverse(matrix)


class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        k_nearest_labels = (self.y_train[i] for i in k_indices)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _euclidean_distance(self, x1, x2):
        return math.sqrt(sum((xi - yi) ** 2 for xi, yi in zip(x1, x2)))
    
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # 特征索引
        self.threshold = threshold    # 特征阈值
        self.left = left              # 左子树
        self.right = right            # 右子树
        self.value = value            # 叶子节点的值

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = len(X), len(X[0])
        num_labels = len(set(y))

        # 停止条件
        if depth >= self.max_depth or num_labels == 1 or num_samples <= 1:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        # 选择最优分割
        best_feature, best_threshold = self._best_split(X, y, num_features)
        left_indices, right_indices = self._split(X, best_feature, best_threshold)
        
        # 构建子树
        left_subtree = self._build_tree([X[i] for i in left_indices], [y[i] for i in left_indices], depth + 1)
        right_subtree = self._build_tree([X[i] for i in right_indices], [y[i] for i in right_indices], depth + 1)
        return TreeNode(best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y, num_features):
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        
        for feature_index in range(num_features):
            thresholds = set(x[feature_index] for x in X)
            for threshold in thresholds:
                left_indices, right_indices = self._split(X, feature_index, threshold)
                if not left_indices or not right_indices:
                    continue

                gini = self._gini(y, left_indices, right_indices)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _split(self, X, feature_index, threshold):
        left_indices = [i for i, x in enumerate(X) if x[feature_index] <= threshold]
        right_indices = [i for i, x in enumerate(X) if x[feature_index] > threshold]
        return left_indices, right_indices

    def _gini(self, y, left_indices, right_indices):
        def gini_impurity(indices):
            labels = [y[i] for i in indices]
            label_counts = Counter(labels)
            impurity = 1 - sum((count / len(indices)) ** 2 for count in label_counts.values())
            return impurity

        num_left, num_right = len(left_indices), len(right_indices)
        num_total = num_left + num_right
        print("flag")
        weighted_gini = (num_left / num_total) * gini_impurity(left_indices) + (num_right / num_total) * gini_impurity(right_indices)
        return weighted_gini

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def _predict(self, inputs):
        node = self.root
        while node.value is None:
            if inputs[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

