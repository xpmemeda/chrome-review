import sys
import time
import heapq

from typing import List, Optional


class RadixTreeNode:
    def __init__(self, tree: "RadixTree", parent: "RadixTreeNode"):
        self.tree = tree
        self.parent = parent
        self.ref = 0
        self.children = {}
        self.access_time = time.time()
        self.k = []

    def size(self) -> int:
        return len(self.k)

    def index(self) -> int:
        return self.k[0]

    def is_leaf(self):
        return len(self.children) == 0

    def update_access_time(self):
        self.access_time = time.time()

    def prefix(self, k) -> List[int]:
        min_size = min(len(self.k), len(k))

        if min_size == 0:
            return []

        for i in range(min_size):
            if self.k[i] != k[i]:
                return k[:i]

        return self.k[:min_size]

    def child(self, k) -> Optional["RadixTreeNode"]:
        if len(k) == 0:
            return None

        if k[0] in self.children:
            return self.children[k[0]]

        return None

    def __lt__(self, other: "RadixTreeNode"):
        return self.access_time < other.access_time


class RadixTree:
    def __init__(self, capacity: Optional[int]):
        self.nodes = set()
        self.root = RadixTreeNode(self, None)
        self.nodes.add(self.root)

        self.capacity = capacity
        self.free_size = self.capacity
        self.protected_size = 0
        self.evictable_size = 0

    def insert(self, k) -> Optional[RadixTreeNode]:
        p = self.root
        n = self.root.child(k)

        free_size = self.free_size
        protected_size = self.protected_size
        evictable_size = self.evictable_size
        trace: List[RadixTreeNode] = []

        while n:
            n.update_access_time()
            trace.append(n)

            prefix_size = len(n.prefix(k))
            if prefix_size == 0:
                break

            if prefix_size < n.size():
                n = self._split_node(n, prefix_size)[0]

            if n.ref == 0:
                protected_size += n.size()
                evictable_size -= n.size()
            n.ref += 1

            k = k[prefix_size:]
            p = n
            n = n.child(k)

        need = len(k)

        if need == 0:
            self.protected_size = protected_size
            self.evictable_size = evictable_size
            return

        if need > self.free_size + evictable_size:
            for n in trace:
                n.ref -= 1
            return

        if need > self.free_size:
            evict_size = self._evict(need - self.free_size)
            free_size += evict_size
            evictable_size -= evict_size

        free_size -= need
        protected_size += need

        self.free_size = free_size
        self.protected_size = protected_size
        self.evictable_size = evictable_size

        new_n = RadixTreeNode(self, p)
        new_n.k = k
        new_n.ref = 1
        p.children[new_n.index()] = new_n

        self.nodes.add(new_n)

        return new_n

    def release(self, n: RadixTreeNode):

        while n:

            n.update_access_time()
            n.ref -= 1

            if n.ref == 0:
                self.protected_size -= n.size()
                self.evictable_size += n.size()

            n = n.parent

    def match(self, k: List[int]) -> List[int]:
        n = self.root
        r = []

        while n:
            prefix = n.prefix(k)

            k = k[len(prefix) :]
            r.extend(prefix)

            if len(prefix) < n.size():
                break

            n = n.child(k)

        return r

    def nnodes(self):
        count = 0
        stack = [self.root]

        while stack:
            x = stack.pop()
            count += 1
            stack.extend(x.children.values())

        return count

    def _split_node(self, n: RadixTreeNode, size: int):
        if n.size() <= size:
            raise RuntimeError()

        parent = n.parent
        k = n.k

        new_n = RadixTreeNode(self, parent)
        self.nodes.add(new_n)

        n.parent = new_n
        n.k = k[size:]

        new_n.ref = n.ref
        new_n.access_time = n.access_time
        new_n.k = k[:size]
        new_n.children[n.index()] = n

        parent.children[new_n.index()] = new_n

        return new_n, n

    def _evict(self, need) -> int:
        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < need and len(leaves):
            x: RadixTreeNode = heapq.heappop(leaves)

            if x == self.root:
                break
            if x.ref > 0:
                break

            num_evicted += x.size()

            parent = x.parent
            self._delete_leaf(x)

            if parent.is_leaf():
                heapq.heappush(leaves, parent)

        return num_evicted

    def _collect_leaves(self) -> List[RadixTreeNode]:
        ret = []
        stack: List[RadixTreeNode] = [self.root]

        while stack:
            n = stack.pop()
            if n.is_leaf():
                ret.append(n)
            else:
                stack.extend(n.children.values())

        return ret

    def _delete_leaf(self, n: RadixTreeNode):
        if not n:
            raise RuntimeError()
        if not n.is_leaf():
            raise RuntimeError()

        p = n.parent
        p.children.pop(n.index())


if __name__ == "__main__":

    def overflow():
        tree = RadixTree(5)

        k = list(range(10))
        n = tree.insert(k)
        assert not n

    overflow()

    def evict():
        tree = RadixTree(5)

        k = [0, 1, 2, 3, 4]
        n = tree.insert(k)
        assert n
        tree.release(n)
        assert tree.nnodes() == 2

        k = [0, 1, 2, 8, 9]
        n = tree.insert(k)
        assert n
        tree.release(n)
        assert tree.nnodes() == 3

        prefix = tree.match([0, 1, 2, 3, 4])
        assert prefix == [0, 1, 2], prefix

    evict()

    def evict_order():
        tree = RadixTree(5)
        k1 = [0, 1, 2, 3]
        n1 = tree.insert(k1)
        k2 = [0, 1, 2, 4]
        n2 = tree.insert(k2)

        tree.release(n2)
        time.sleep(0.01)
        tree.release(n1)

        k3 = [0, 1, 2, 5]
        n3 = tree.insert(k3)
        tree.release(n3)

        prefix_k1 = tree.match(k1)
        assert prefix_k1 == k1, prefix_k1
        prefix_k2 = tree.match(k2)
        assert prefix_k2 == k2[:3], prefix_k2
        prefix_k3 = tree.match(k3)
        assert prefix_k3 == k3, prefix_k3

    evict_order()

    def test():
        tree = RadixTree(10)
        assert tree.free_size == 10
        assert tree.protected_size == 0
        assert tree.evictable_size == 0

        k = list(range(5))
        n = tree.insert(k)
        assert n
        assert tree.free_size == 5
        assert tree.protected_size == 5
        assert tree.evictable_size == 0
        assert tree.nnodes() == 2
        tree.release(n)
        assert tree.free_size == 5
        assert tree.protected_size == 0
        assert tree.evictable_size == 5
        assert tree.nnodes() == 2
        prefix = tree.match(k)
        assert prefix == k
        assert tree.nnodes() == 2

        k = list(range(8))
        n = tree.insert(k)
        assert n
        assert tree.free_size == 2
        assert tree.protected_size == 8
        assert tree.evictable_size == 0
        assert tree.nnodes() == 3
        tree.release(n)

    test()
