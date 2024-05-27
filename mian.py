# Data Analysis
import pandas as pd
import numpy as np
# Data Visualization
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
# Statistics
import statsmodels.api as sm
# Machine Learning
from sklearn.metrics import mean_squared_error, r2_score
# Warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
# Loggers
import logging
import warnings
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)


# Get the data
df = pd.read_excel(os.path.join('data', 'Dry_Bean_Dataset.xlsx'))
df.duplicated().sum()
df.drop_duplicates(inplace=True)

df.isna().sum(axis=1).sum() # is 0
# see the dataset in the lab directory
# df = pd.read_excel(os.path.join('data', 'Dry_Bean_Dataset.xlsx'), usecols=['Area', 'Perimeter'])
data = df.drop(columns=['Area', 'Class'])
label = df[['Class']]
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=42)

# 初始化决策树分类器
clf = DecisionTreeClassifier(random_state=42)

# 训练模型
clf.fit(train_data, train_label)

# 进行预测
y_pred = clf.predict(test_data)

# 计算准确率
accuracy = accuracy_score(test_label, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 打印分类报告
print("\nClassification Report:")
print(classification_report(test_label, y_pred))

# 打印混淆矩阵
print("\nConfusion Matrix:")
print(confusion_matrix(test_label, y_pred))

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=list(train_data.columns), class_names=list(set(train_label['Class'])))
plt.show()

list(train_data.columns)

set(list(train_label['Class']))

list(train_label['Class'])


df.groupby(by=['Class']).mean()

from scipy.stats import f_oneway


def group_mean_analysis(df, group_col):
    """
    对DataFrame中每一列进行分组均值差异的检验

    参数：
    df: DataFrame，包含数据和分组列
    group_col: str，分组列的列名

    返回：
    results: dict，每一列的检验结果（p值）
    :param df:
    :param group_col:
    :return:
    """
    results = {}
    for col in df.columns:
        if col != group_col:  # 排除分组列
            groups = [df[col][df[group_col] == group] for group in df[group_col].unique()]
            f_statistic, p_value = f_oneway(*groups)
            results[col] = p_value
    return results


# 示例用法
# 假设 df 是你的 DataFrame，'group' 是用于分组的列名
# 调用函数进行每列的分组均值差异检验
results = group_mean_analysis(df, 'Class')

# 打印每列的检验结果（p值）
for col, p_value in results.items():
    print(f"{col}: p值 = {p_value}")


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def plot_boxplot_grouped(df, numeric_cols, group_col):
    """
    使用 seaborn 中的 boxenplot 将每个类别中所有 numeric 特征的分布情况放在一张图上。

    参数:
    - df: pandas DataFrame，包含要绘制箱线图的数据。
    - numeric_cols: list，包含需要绘制的数值型特征列名。
    - group_col: str，指定用于分组的列名。

    返回值:
    - 无。该函数直接展示箱线图。
    """
    # 检查系统中是否有支持中文的字体，如果没有则使用SimHei

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题

    # 初始化绘图窗口并设置尺寸
    fig, axs = plt.subplots(int(len(numeric_cols)/2+1), 2, figsize=(24, 3 * len(numeric_cols)))  # 创建子图数组
    for i, col in enumerate(numeric_cols):
        # 确保所有numeric_cols都在df中
        if col in df.columns:
            sns.boxplot(data=df, x=group_col, y=col, ax=axs[int(i/2)][i%2], palette='Set3', hue=group_col)
            axs[int(i/2)][i%2].set_title(f'{col} 的分布情况')
            axs[int(i/2)][i%2].set_xlabel('类别')
            axs[int(i/2)][i%2].set_ylabel(col)
            axs[int(i/2)][i%2].legend(title=group_col)  # 添加图例
            axs[int(i/2)][i%2].tick_params(labelrotation=45)  # 旋转x轴标签

    # 删除多余的子图
    plt.tight_layout()
    plt.show()

plot_boxplot_grouped(df, ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'], 'Class')

import pandas as pd
import numpy as np

# 定义一个简单的数据集
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
}

# 创建DataFrame
df = pd.DataFrame(data)


# 定义一个函数计算信息熵
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy


# 定义一个函数计算信息增益
def InfoGain(data, split_attribute_name, target_name):
    # 计算总体的信息熵
    total_entropy = entropy(data[target_name])

    # 计算按指定属性分裂后的信息熵
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])

    # 计算信息增益
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


# 定义一个函数用于构建决策树
def ID3(data, original_data, features, target_attribute_name, parent_node_class=None):
    # 如果所有目标值都相同，则停止分裂
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # 如果数据为空，则返回父节点的目标值
    elif len(data) == 0:
        return parent_node_class

    # 如果特征为空，则返回数据中最常见的目标值
    elif len(features) == 0:
        return parent_node_class

    # 如果以上条件都不满足，则继续构建树
    else:
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]

        # 计算信息增益并选择最佳分裂特征
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # 创建根节点
        tree = {best_feature: {}}

        # 从特征列表中移除已选用的特征
        features = [i for i in features if i != best_feature]

        # 递归构建子树
        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data, original_data, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

        return tree


# 使用ID3算法构建决策树
tree = ID3(df, df, df.columns[:-1], 'PlayTennis')

# 打印生成的决策树
print(tree)
