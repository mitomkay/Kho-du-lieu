import numpy as np 
import pandas as pd

'''
  Hàm tính toán entropy
  Đầu vào: 
    freq(arr) - tần số
  Lưu ý: loại bỏ giá trị tần suất bằng 0 đi vì logarit tại đây không xác định.
'''
def entropy(freq):
    freq_0 = freq[np.array(freq).nonzero()[0]]
    prob_0 = freq_0/float(freq_0.sum())
    return -np.sum(prob_0 * np.log(prob_0))
  
  
def discretize(df, columns, bins):
  for col in columns:
    df[col] = pd.cut(df[col], bins=bins, labels=False, include_lowest=True)
  return df

def tree_depth(node):
  if not node.children:
    return 1
  else:
    return 1 + max(tree_depth(child) for child in node.children)
      
def count_leaf_nodes(node):
  if not node.children:
    return 1
  else:
    return sum(count_leaf_nodes(child) for child in node.children)
  
def count_total_nodes(node):
  if not node.children:
    return 1
  else:
    return 1 + sum(count_total_nodes(child) for child in node.children)

def count_splits(node):
  if not node.children:
    return 0  
  else:
    return 1 + sum(count_splits(child) for child in node.children)
  
def average_node_impurity(node):
  if not node.children:
    return node.entropy
  else:
    return (node.entropy + sum(average_node_impurity(child) for child in node.children)) / (1 + len(node.children))
