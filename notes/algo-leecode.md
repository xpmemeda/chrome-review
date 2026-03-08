# 数组

## 栈的使用

工作中比较少用到栈这种数据结构，因此做题时常常忘记。

示例

```cpp
/*
LCR 037. 行星碰撞

给定一个整数数组 asteroids，表示在同一行的小行星。

对于数组中的每一个元素，其绝对值表示小行星的大小，正负表示小行星的移动方向（正表示向右移动，负表示向左移动）。每一颗小行星以相同的速度移动。

找出碰撞后剩下的所有小行星。碰撞规则：两个行星相互碰撞，较小的行星会爆炸。如果两颗行星大小相同，则两颗行星都会爆炸。两颗移动方向相同的行星，永远不会发生碰撞。



示例 1：

输入：asteroids = [5,10,-5]
输出：[5,10]
解释：10 和 -5 碰撞后只剩下 10 。 5 和 10 永远不会发生碰撞。
示例 2：

输入：asteroids = [8,-8]
输出：[]
解释：8 和 -8 碰撞后，两者都发生爆炸。
示例 3：

输入：asteroids = [10,2,-5]
输出：[10]
解释：2 和 -5 发生碰撞后剩下 -5 。10 和 -5 发生碰撞后剩下 10 。
示例 4：

输入：asteroids = [-2,-1,1,2]
输出：[-2,-1,1,2]
解释：-2 和 -1 向左移动，而 1 和 2 向右移动。
由于移动方向相同的行星不会发生碰撞，所以最终没有行星发生碰撞。


提示：

2 <= asteroids.length <= 104
-1000 <= asteroids[i] <= 1000
asteroids[i] != 0
*/

#include <iostream>
#include <vector>

void print_vec(const std::vector<int> &vec) {
  for (auto x : vec) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
}

class Solution {
public:
  std::vector<int> asteroidCollision(const std::vector<int> &asteroids) {
    std::vector<int> r;
    for (auto x : asteroids) {
      if (x > 0) {
        r.push_back(x);
      }

      while (!r.empty() && r.back() < -x) {
        r.pop_back();
      }

      if (!r.empty() && r.back() == -x) {
        r.pop_back();
      }
    }

    std::vector<int> l;
    for (size_t i = 0; i < asteroids.size(); ++i) {
      auto x = asteroids[asteroids.size() - 1 - i];

      if (x < 0) {
        l.push_back(x);
      }

      while (!l.empty() && -l.back() < x) {
        l.pop_back();
      }

      if (!l.empty() && -l.back() == x) {
        l.pop_back();
      }
    }

    std::vector<int> result;
    for (size_t i = 0; i < l.size(); ++i) {
      result.push_back(l[l.size() - 1 - i]);
    }
    for (size_t i = 0; i < r.size(); ++i) {
      result.push_back(r[i]);
    }

    return result;
  }
};

int main() {
  print_vec(Solution().asteroidCollision({-1, 5, 10, -5}));
  return 0;
}
```

## 动态规划

核心是最优子结构 + 重叠子问题。把大的问题拆解为小的子问题。

- 最优子结构：一个问题的最优解，可以由它的子问题的最优解构成。换句话说就是，大问题最优，那么子问题也必须最优。
- 重叠子问题：递归分解后，会反复计算同一个子问题。
- Ex.lc321 子问题的最优解可以被短状态表示。每次递归的每个层级，该层级只能保存 O(1) 负责度的状态。

示例

```cpp
/*
188. 买卖股票的最佳时机 IV

给你一个整数数组 prices 和一个整数 k ，其中 prices[i] 是某支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。也就是说，你最多可以买 k 次，卖 k 次。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

 

示例 1：

输入：k = 2, prices = [2,4,1]
输出：2
解释：在第 1 天 (股票价格 = 2) 的时候买入，在第 2 天 (股票价格 = 4) 的时候卖出，这笔交易所能获得利润 = 4-2 = 2 。
示例 2：

输入：k = 2, prices = [3,2,6,5,0,3]
输出：7
解释：在第 2 天 (股票价格 = 2) 的时候买入，在第 3 天 (股票价格 = 6) 的时候卖出, 这笔交易所能获得利润 = 6-2 = 4 。
     随后，在第 5 天 (股票价格 = 0) 的时候买入，在第 6 天 (股票价格 = 3) 的时候卖出, 这笔交易所能获得利润 = 3-0 = 3 。

提示：

1 <= k <= 100
1 <= prices.length <= 1000
0 <= prices[i] <= 1000
*/

/*
解题：

1. 有两种状态：买入状态 和 卖出状态。
2. 有两个坐标：交易位置 和 交易次数。

首先还是要明确有几种状态，然后写出状态转移方程。

buy[i][j] = std::max(buy[i - 1][j], sell[i - 1][j - 1] - prices[i]);
sell[i][j] = std::max(sell[i - 1][j], buy[i - 1][j] + prices[i]);

*/

#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

class Solution {
public:
  int maxProfit(int k, std::vector<int> &prices) {
    // buy[i][j] 在坐标 i 处完成 <= j 次交易，且状态属于持有股票。
    // sell[i][j] 在坐标 i 处完成 <= j 次交易，且状态属于未持有股票。

    std::vector<std::vector<int>> buy(prices.size(), std::vector<int>(k));
    std::vector<std::vector<int>> sell(prices.size(), std::vector<int>(k));

    // 1. init. max algo -> min init.
    buy[0][0] = -prices[0];
    for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
      buy[0][j] = -prices[0];
      sell[0][j] = 0;
    }

    for (size_t i = 1; i < prices.size(); ++i) {
      sell[i][0] = std::max(sell[i - 1][0], buy[i - 1][0] + prices[i]);
      buy[i][0] = std::max(buy[i - 1][0], -prices[i]);
    }

    for (size_t i = 1; i < prices.size(); ++i) {
      for (size_t j = 1; j < static_cast<size_t>(k); ++j) {
        buy[i][j] = std::max(buy[i - 1][j], sell[i - 1][j - 1] - prices[i]);
        sell[i][j] = std::max(sell[i - 1][j], buy[i - 1][j] + prices[i]);
      }
    }

    auto max = std::numeric_limits<int>::min();
    for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
      max = std::max<int>(max, sell[prices.size() - 1][j]);
    }
    return max;
  }
};
```

# 链表

有一些基本操作，组合这些基本操作可完成题目要求。

1. 反转链表：遍历链表，将每个被访问的节点都设置为新的头节点即可。时间复杂度 ``O(n)`` 空间复杂度 ``O(1)`` 。
2. 寻找 ``第 k 个`` 或者 ``倒数第 k 个节点`` 节点，可以使用先后双指针法解题，时间复杂度 ``O(n)`` 空间复杂度 ``O(1)`` 。

# 树

回顾下遍历树的几种方式，基本上有一种方式可以在遍历树的过程中顺便把题解了。

- 先序遍历：根左右，递归实现。
- 中序遍历：左根右，递归实现。
- 后序遍历：左右根，递归实现。
- 层级遍历：队列实现。


# 图

NOTE: 题目给的参数可能不是一张图，比如说给所有边的 Pair ，可以先把图搭出来再去遍历。

```cpp
/**
 *  既可以表示有向图，也可以表示无向图。
 *
 *  n:      节点数量
 *  graph:  相邻节点
 */
void question(int n, const std::vector<std::vector<int>>& graph);
```

NOTE: 一张图要全部 N 节点连通，至少需要 N-1 条边。

一些常见的子问题

1. 判断一张无向图是不是连通图，有几个连通分量。
```cpp
#include <vector>
using namespace std;

void dfs(int u, const vector<vector<int>>& graph, vector<bool>& visited) {
    visited[u] = true;
    for (int v : graph[u]) {
        if (!visited[v]) {
            dfs(v, graph, visited);
        }
    }
}

int countConnectedComponents(int n, const vector<vector<int>>& graph) {
    vector<bool> visited(n, false);
    int count = 0;

    for (int i = 0; i < n; ++i) {
        if (!visited[i]) {
            dfs(i, graph, visited);
            count++;
        }
    }

    return count;
}
```

2. 判断一张有向图是否成环。
```cpp
// 一个常用且高效的方法：DFS + 三色标记（visited / visiting / unvisited） 判断有向图是否有环。
// 核心思想：
// 0 = 未访问（unvisited）
// 1 = 访问中（visiting，当前DFS路径上）
// 2 = 已完成（visited，已退出DFS）
// 如果在DFS过程中，遇到一个状态为 1 的节点，说明走回了当前递归栈里的节点 ⇒ 存在环。

#include <vector>
using namespace std;

bool dfs(int u, const vector<vector<int>>& graph, vector<int>& state) {
    state[u] = 1; // visiting

    for (int v : graph[u]) {
        if (state[v] == 1) {
            // 回到当前递归栈中的节点 -> 有环
            return true;
        }
        if (state[v] == 0) {
            if (dfs(v, graph, state)) {
                return true;
            }
        }
    }

    state[u] = 2; // visited
    return false;
}

bool hasCycle(int n, const vector<vector<int>>& graph) {
    vector<int> state(n, 0); // 0=unvisited

    for (int i = 0; i < n; ++i) {
        if (state[i] == 0) {
            if (dfs(i, graph, state)) {
                return true;
            }
        }
    }
    return false;
}
```

回顾下遍历图的几种方式，基本上有一种方式可以在遍历图的过程中顺便把题解了。

- 广度优先算法：递归实现。
- 深度优先算法：队列实现。

遍历图和遍历树不一样，到达图中的某点可能有多条路径，因此会重复到达，配合 visited 数组达到遍历全图的效果。

# 回溯

本质上是一种有剪枝的暴力求解。

所谓暴力求解就是列出所有可能的场面，然后从中筛选出符合要求的 case 。

回溯则是中途判断合理性，符合要求并不一定所有位置都确定了才能判断。

示例

```cpp
/*
51. N 皇后

按照国际象棋的规则，皇后可以攻击与之处在同一行或同一列或同一斜线上的棋子。

n 皇后问题 研究的是如何将 n 个皇后放置在 n×n
的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 n ，返回所有不同的 n 皇后问题 的解决方案。

每一种解法包含一个不同的 n 皇后问题 的棋子放置方案，该方案中 'Q' 和 '.'
分别代表了皇后和空位。


示例 1：

输入：n = 4
输出：[[".Q..","...Q","Q...","..Q."],["..Q.","Q...","...Q",".Q.."]]

示例 2：

输入：n = 1
输出：[["Q"]]


提示：

1 <= n <= 9
*/

/*
解题：

我们不需要在 N 个皇后摆完后才判断是否符合要求，摆到一半就知道了。
*/

#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

class Solution {
public:
  void printVec(const std::vector<int> &vec) {
    for (auto &x : vec) {
      std::cout << x << " ";
    }
    std::cout << std::endl;
  }

  std::vector<std::vector<std::string>> solveNQueens(int n) {
    std::vector<int> queens;
    std::unordered_set<int> cols;
    std::unordered_set<int> diag1;
    std::unordered_set<int> diag2;

    verifyQueens(n, queens, cols, diag1, diag2);

    std::vector<std::vector<std::string>> r;
    for (auto &x : can_) {
      std::vector<std::string> s;
      for (auto i : x) {
        s.emplace_back(n, '.');
        s.back()[i] = 'Q';
      }
      r.push_back(s);
    }

    return r;
  }

  void verifyQueens(int n, std::vector<int> &queens,
                    std::unordered_set<int> &cols,
                    std::unordered_set<int> &diag1,
                    std::unordered_set<int> &diag2) {
    int row_idx = queens.size();

    std::vector<int> can_cols;
    for (int i = 0; i < n; ++i) {
      if (cols.find(i) != cols.end()) {
        continue;
      }
      if (diag1.find(i - row_idx) != diag1.end()) {
        continue;
      }
      if (diag2.find(i + row_idx) != diag2.end()) {
        continue;
      }
      can_cols.push_back(i);
    }

    if (can_cols.empty()) {
      return;
    }

    if (row_idx == n - 1) {
      can_.push_back(queens);
      can_.back().push_back(can_cols[0]);
      return;
    }

    for (auto c : can_cols) {
      queens.push_back(c);
      cols.insert(c);
      diag1.insert(c - row_idx);
      diag2.insert(c + row_idx);
      verifyQueens(n, queens, cols, diag1, diag2);
      queens.pop_back();
      cols.erase(c);
      diag1.erase(c - row_idx);
      diag2.erase(c + row_idx);
    }
  }

private:
  std::vector<std::vector<int>> can_;
};

int main() {
  Solution().solveNQueens(4);
  return 0;
}
```