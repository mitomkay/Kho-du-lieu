import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from DecisionTreeID3 import *
from Functions import *
from random_forest import *

if __name__ == "__main__":
    df = pd.read_csv('./data/data2.csv')
    # dfTest = pd.read_csv('./data/data2.csv')
    
    # Danh sách các thuộc tính liên tục
    continuous_columns = ['time', 'age', 'wtkg', 'karnof', 'preanti', 'cd40', 'cd80', 'cd420', 'cd820']
    
    # Chia khoảng (ví dụ: 5 khoảng)
    df = discretize(df, continuous_columns, bins=5)
    
    X = df.drop(columns=['infected'])
    y = df['infected']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tree = DecisionTreeID3(max_depth=22, min_samples_split=2)
    # tree = RandomForest(max_depth=22)
    tree.fit(X_train, y_train)

    # Dự đoán và đánh giá trên tập huấn luyện
    train_predictions = tree.predict(X_train)
    y_train_pred = pd.Series(train_predictions).values
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')

    # Dự đoán và đánh giá trên tập kiểm tra
    test_predictions = tree.predict(X_test)
    y_test_pred = pd.Series(test_predictions).values
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # depth = tree_depth(tree.root)
    # num_leaf_nodes = count_leaf_nodes(tree.root)
    # total_nodes = count_total_nodes(tree.root)
    # num_splits = count_splits(tree.root)

    # print("Chiều sâu của cây:", depth)
    # print("Số lượng nút lá:", num_leaf_nodes)
    # print("Tổng số lượng nút:", total_nodes)
    # print("Số lượng điều kiện phân chia:", num_splits)

    # In kết quả đánh giá
    print("Hiệu suất trên tập huấn luyện:")
    print("Độ chính xác (Accuracy):", train_accuracy)
    print("Độ đặc hiệu (Precision):", train_precision)
    print("Độ nhạy (Recall):", train_recall)
    print("F1-score:", train_f1)

    print("\nHiệu suất trên tập kiểm tra:")
    print("Độ chính xác (Accuracy):", test_accuracy)
    print("Độ đặc hiệu (Precision):", test_precision)
    print("Độ nhạy (Recall):", test_recall)
    print("F1-score:", test_f1)

    # Đánh giá overfitting và underfitting
    if train_accuracy > test_accuracy:
        print("\nMô hình có thể bị overfitting.")
    elif train_accuracy < test_accuracy:
        print("\nMô hình có thể bị underfitting.")
    else:
        print("\nMô hình có hiệu suất tương đối tốt trên cả tập huấn luyện và tập kiểm tra.")
