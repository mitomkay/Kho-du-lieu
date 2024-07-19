from TreeNode import *
import numpy as np


class DecisionTreeID3(object):
  def __init__(self, max_depth=10, min_samples_split=2, min_gain=1e-4):
    self.root = None
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.min_gain = min_gain

  def fit(self, data, target):
    self.data = data
    self.attributes = data.columns.tolist()
    self.target = target
    self.labels = target.unique()

    ids = data.index
    self.root = TreeNode(ids=ids, entropy=self._entropy(ids), depth=0)
    queue = [self.root]
    while queue:
      node = queue.pop(0)
      if node.depth < self.max_depth and node.entropy > self.min_gain:
        node.children = self._split(node)
        if not node.children:  # Nút lá
          self._set_label(node)
        queue.extend(node.children)
      else:
        self._set_label(node)

  def _entropy(self, ids):
    if len(ids) == 0:
      return 0
    freq = np.array(self.target[ids].value_counts())
    freq_0 = freq[freq.nonzero()]
    prob_0 = freq_0 / float(freq_0.sum())
    return -np.sum(prob_0 * np.log2(prob_0))

  def _set_label(self, node):
    target_ids = node.ids
    node.set_label(self.target[target_ids].mode()[0])

  def _split(self, node):
    ids = node.ids
    best_gain = 0
    best_splits = []
    best_attribute = None
    order = None
    sub_data = self.data.loc[ids, :]
    for att in self.attributes:
      values = self.data.loc[ids, att].unique().tolist()
      if len(values) == 1:
        continue  # Tất cả giá trị đều thuộc 1 tập -> entropy = 0
      splits = []
      for val in values:
        sub_ids = sub_data.index[sub_data[att] == val]
        splits.append([sub_id for sub_id in sub_ids])
      if min(map(len, splits)) < self.min_samples_split:
        continue
      HxS = sum(len(split) * self._entropy(split) / len(ids) for split in splits)
      gain = node.entropy - HxS
      if gain < self.min_gain:
        continue  # Dừng nếu gain nhỏ
      if gain > best_gain:
        best_gain = gain
        best_splits = splits
        best_attribute = att
        order = values
    if best_attribute is None:
      return []
    node.set_properties(best_attribute, order)
    return [TreeNode(ids=split, entropy=self._entropy(split), depth=node.depth + 1) for split in best_splits]

  def predict(self, new_data):
    idx = new_data.index
    labels = {}
    for n in idx:
      x = new_data.loc[n, :]
      node = self.root
      while node.children:
        try:
          node = node.children[node.order.index(x[node.split_attribute])]
        except ValueError:
          break
      labels[n] = node.label if node.label else 0
    return labels