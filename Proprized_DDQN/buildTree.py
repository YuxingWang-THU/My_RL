import numpy as np


class Tree(object):
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity  # capacity是叶子节点个数，也就是存放数据的个数，树的总结点个数为 n = (2 * 叶子节点数 - 1)
        self.tree = np.zeros(2 * capacity)  # 从1开始编号[1,capacity]
        self.data = np.zeros(capacity+1, dtype=object)  # 存叶子节点对应的数据data[叶子节点编号id] = data

    def add(self, p, data):
        idx = self.write + self.capacity  #定位到第一个叶子节点（idx）
        self.data[self.write+1] = data
        self._updatetree(idx, p)
        self.write += 1
        if self.write > self.capacity:  # 存满时，在从左到右从上到下，重新一个个替换
            self.write = 0

    def _updatetree(self, idx, p):
        change = p - self.tree[idx]  # 计算更新的p和叶子结点（idx）的差值，
        self._propagate(idx, change)  # 在把这个差值change往上传播，把这个节点（idx）的父节点和父父节点都加上这个change
        self.tree[idx] = p  # 更新这个节点（idx）的值

    def _propagate(self, idx, change):
        parent = idx // 2  # 计算父节点位置的公式 = 子节点（idx） 除于二向下取整
        self.tree[parent] += change  # 更新父节点的值，也就是向上传播的体现
        if parent != 1:
            self._propagate(parent, change)  # 在把这个差值change往上传播，把这个节点（idx）的父节点和父父节点都加上这个change

    def _total(self):
        return self.tree[1]  # 得到总的p的和

    def get(self, s):
        idx = self._retrieve(1, s)  # 定位到满足要查找的要求的第一个节点(idx)
        index_data = idx - self.capacity + 1  # idx - self.capacity 相减是为了定位到时那个叶子节点，加1是因为存数据时是从1开始的
        return (idx, self.tree[idx], self.data[index_data])

    def _retrieve(self, idx, s):
        left = 2 * idx  # 树的左孩子编号
        right = left + 1
        if left >= (len(self.tree)-1):  # 如果(idx)没有左孩子，也就是（idx）是叶子节点直接返回
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)  # 往左孩子处查找
        else:
            return self._retrieve(right, s - self.tree[left])  # 往右孩子处查找

tree = Tree(5)
tree.add(1,3)
tree.add(2,4)
tree.add(3,5)
tree.add(4,6)
tree.add(6,11)

print(tree.get(4))





