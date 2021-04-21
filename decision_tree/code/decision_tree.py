import matplotlib.pyplot as plt
import pandas as pd
import urllib.request
import numpy as np
import os


class Data:
    def __init__(self, class_num=None, x=None, y=None):
        self.class_num = class_num
        self.x = x
        self.y = y

    def load_pkl(self, url):
        name = url.split('/')[-1]
        if not os.path.exists(name):
            print("Download:", url)
            urllib.request.urlretrieve(url, name)
        return pd.read_pickle(name)

    def get_data(self, url, drop_names, y_name):
        df = self.load_pkl(url)
        drop_names.append(y_name)
        self.x = df.drop(columns=drop_names).to_numpy()
        self.y = df[y_name].to_numpy()


class Tree_Node:
    def __init__(self, feature=None, threshold=None, value=None, left_node=None, right_node=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left_node = left_node
        self.right_node = right_node


class Cart_Tree:
    def __init__(self, root=None, depth_limit=None, mode=0):
        self.root = root
        self.depth_limit = depth_limit
        self.mode = mode

    def calculate_gini(self, data):
        cnt = np.zeros(data.class_num)
        for i in data.y:
            cnt[i] = cnt[i] + 1
        ret = 1
        for i in range(data.class_num):
            ret = ret - (cnt[i] / len(data.y)) ** 2
        return ret

    def find_value(self, data):
        cnt = np.zeros(data.class_num)
        for i in data.y:
            cnt[i] = cnt[i] + 1
        index = cnt.argmax()
        return index, cnt[index]

    def split_data(self, feature, threshold, data):
        left_data = Data(class_num=data.class_num)
        right_data = Data(class_num=data.class_num)

        split_threshold = data.x[:, feature] <= threshold
        left_data.x = data.x[split_threshold, :]
        left_data.y = data.y[split_threshold]
        right_data.x = data.x[~split_threshold, :]
        right_data.y = data.y[~split_threshold]

        return left_data, right_data

    def find_cut(self, data):
        cur_id, cur_mx = self.find_value(data)
        if cur_mx == len(data.y):
            return None, None, cur_id
        cur_gini = self.calculate_gini(data)

        best_gini = -1
        best_feature = None
        best_threshold = None

        if self.mode == 1:
            selected_features = np.random.choice(
                len(data.x[0]),
                int(len(data.x[0]) ** 0.5 + 0.5),
                replace=False)
        else:
            selected_features = np.arange(len(data.x[0]))

        for i in selected_features:
            arg = data.x[:, i].argsort()
            sort_x = data.x[arg]
            sort_y = data.y[arg]

            pre = np.zeros((len(sort_y), data.class_num))

            for j in range(len(sort_y)):
                if j == 0:
                    pre[j][sort_y[j]] = 1
                else:
                    for k in range(data.class_num):
                        pre[j][k] = pre[j - 1][k]
                    pre[j][sort_y[j]] = pre[j - 1][sort_y[j]] + 1

            for j in range(len(sort_y) - 2, -1, -1):
                if sort_x[j][i] != sort_x[j + 1][i]:
                    left_num = j + 1
                    right_num = len(sort_y) - left_num
                    left_gini = 1
                    right_gini = 1
                    for k in range(data.class_num):
                        left_gini = left_gini - (pre[j][k] / left_num) ** 2
                        right_gini = right_gini - \
                            ((pre[-1][k] - pre[j][k]) / right_num) ** 2

                    choose_gini = cur_gini \
                        - left_num / len(sort_y) * left_gini \
                        - right_num / len(sort_y) * right_gini

                    if choose_gini > best_gini:
                        best_gini = choose_gini
                        best_feature = i
                        best_threshold = (sort_x[j][i] + sort_x[j + 1][i]) / 2

        return best_feature, best_threshold, None

    def build_tree(self, cur, data, depth):
        if depth > self.depth_limit:
            cur.value = self.find_value(data)[0]
            return
        cur.feature, cur.threshold, cur.value = self.find_cut(data)
        if cur.feature is not None:
            left_data, right_data = self.split_data(
                cur.feature, cur.threshold, data)
            cur.left_node = Tree_Node()
            cur.right_node = Tree_Node()
            self.build_tree(cur.left_node, left_data, depth + 1)
            self.build_tree(cur.right_node, right_data, depth + 1)
        elif cur.value is None:
            cur.value = self.find_value(data)[0]

    def train(self, data):
        self.root = Tree_Node()

        if self.mode == 1:
            selected_id = np.random.choice(len(data.y), 20000)
            selected_data = Data(class_num=data.class_num)
            selected_data.x = data.x[selected_id]
            selected_data.y = data.y[selected_id]
            self.build_tree(self.root, selected_data, 0)
        else:
            self.build_tree(self.root, data, 0)

    def predict(self, cur, features):
        if cur.value is not None:
            return cur.value
        if features[cur.feature] <= cur.threshold:
            return self.predict(cur.left_node, features)
        return self.predict(cur.right_node, features)


class Random_Forest:
    def __init__(self, tree_num=None, depth_limit=None, trees=[]):
        self.tree_num = tree_num
        self.depth_limit = depth_limit
        self.trees = trees

    def train(self, data):
        self.trees = []
        for i in range(self.tree_num):
            self.trees.append(Cart_Tree(depth_limit=self.depth_limit, mode=1))
            self.trees[-1].train(data)

    def predict(self, features, class_num):
        cnt = np.zeros(class_num)
        for i in range(self.tree_num):
            vote = self.trees[i].predict(self.trees[i].root, features)
            cnt[vote] = cnt[vote] + 1
        return cnt.argmax()


def perform(tree_num, cur, data):
    tp, fp, fn, tn = 0, 0, 0, 0
    for j in range(len(data.y)):
        ret = 0
        if tree_num == 1:
            ret = cur.predict(cur.root, data.x[j])
        else:
            ret = cur.predict(data.x[j], 2)
        if ret == 1:
            if data.y[j] == 1:
                tp = tp + 1
            else:
                fp = fp + 1
        else:
            if data.y[j] == 1:
                fn = fn + 1
            else:
                tn = tn + 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * precision * recall) / (precision + recall)
    accu = (tp + tn) / (tp + fp + fn + tn)

    return f1_score, accu


def test(tree_num, depth, train_data, valid_data):
    if tree_num == 1:
        train_f1_score = np.zeros(depth)
        train_accu = np.zeros(depth)
        valid_f1_score = np.zeros(depth)
        valid_accu = np.zeros(depth)

        for i in range(0, depth):
            tree = Cart_Tree(depth_limit=i + 1)
            tree.train(train_data)
            train_f1_score[i], train_accu[i] = perform(
                tree_num, tree, train_data)
            valid_f1_score[i], valid_accu[i] = perform(
                tree_num, tree, valid_data)
            print("Depth: ", i + 1)
            print("  Train f1 Score: %.4f" % train_f1_score[i])
            print("  Train Accuracy: %.4f" % train_accu[i])
            print("  Valid f1 Score: %.4f" % valid_f1_score[i])
            print("  Valid Accuracy: %.4f" % valid_accu[i])
            print("")

        x = np.arange(1, depth + 1)
        plt.plot(x, train_accu, "blue", marker='.', label="Train")
        plt.plot(x, valid_accu, "red", marker='.', label="Valid")
        plt.legend(loc="upper left")
        plt.xticks(x)
        plt.xlabel("Depth")
        plt.ylabel("Accuracy")
        plt.show()
        plt.plot(x, train_f1_score, "blue", marker='.', label="Train")
        plt.plot(x, valid_f1_score, "red", marker='.', label="Valid")
        plt.legend(loc="upper left")
        plt.xticks(x)
        plt.xlabel("Depth")
        plt.ylabel("f1 score")
        plt.show()
    else:
        rf = Random_Forest(tree_num=tree_num, depth_limit=depth)
        rf.train(train_data)
        train_f1_score, train_accu = perform(tree_num, rf, train_data)
        valid_f1_score, valid_accu = perform(tree_num, rf, valid_data)
        print("Random Forest:", "num =", tree_num, "depth =", depth)
        print("  Train f1 Score: %.4f" % train_f1_score)
        print("  Train Accuracy: %.4f" % train_accu)
        print("  Valid f1 Score: %.4f" % valid_f1_score)
        print("  Valid Accuracy: %.4f" % valid_accu)
        print("")


data = Data(class_num=2)
data.get_data("https://lab.djosix.com/weather.pkl", [], "RainTomorrow")

arg = np.arange(len(data.y))
np.random.shuffle(arg)
data.x = data.x[arg]
data.y = data.y[arg]

train_data = Data(class_num=2)
train_data.x = data.x[:int(0.8 * len(data.y)), :]
train_data.y = data.y[:int(0.8 * len(data.y))]
valid_data = Data(class_num=2)
valid_data.x = data.x[int(0.8 * len(data.y)):, :]
valid_data.y = data.y[int(0.8 * len(data.y)):]

test(1, 3, train_data, valid_data)
test(100, 20, train_data, valid_data)
