# divide_graph 详解

## 功能

`divide_graph.cpp` 的作用是**把平铺的算子图分割成若干个子图（SubGraph）**，每个子图包装成一个 `SubGraphNode`。分割依据是 `is_dynamic` 属性：静态 shape 算子和动态 shape 算子不能放在同一个子图中。

执行完之后，**顶层图里的所有节点都是 SubGraphNode**，即使里面只包含一个 op 也会被包成子图。

---

## 整体流程

`divide_graph()` 分三步：搜索、合并、记录 IO。

```cpp
void divide_graph(GraphPtr g)
{
    // 第一轮：搜索并创建子图
    for (auto &sg : search_subgraphs(g))
    {
        auto sgn = SubGraph::create(g, sg)->as_node(true, true);
        sgn->set_unified_view(false);
        set_subgraph_color(sgn);  // 动态=lightblue, 静态=lightgrey（dot 图可视化用）
    }

    // 第二轮：尝试合并相邻子图
    merge_subgraphs(g);

    // 标记所有子图的 IO tensor
    for (auto n : g->nodes())
    {
        record_io_tensors(SubGraphNode::from_node(n));
    }

    // 验证
    g->check_sg();   // 子图结构完整性检查
    is_DAG(g);       // 确保图仍然是 DAG（无环）
}
```

---

## 各函数详解

### 1. `search_subgraph()` — 从一个节点出发，搜集同类节点

```cpp
static void search_subgraph(NodePtr n,
                            std::set<NodePtr> &ran_nodes,
                            std::vector<NodePtr> &subgraph,
                            bool is_dynamic)
```

这是一个**向上（往 parent 方向）递归**的搜索：

```
从节点 n 出发：
  1. 如果 n 已经访问过，或者 n.is_dynamic() != is_dynamic → 停止
     （动态节点和静态节点不能放同一个子图）
  2. 标记 n 为已访问，加入当前 subgraph
  3. 对每个 parent p：
     ├─ 检查 p 的所有 children 是否都已经被访问过
     │   ├─ 是 (independent=true) → 递归搜索 p
     │   └─ 否 (independent=false) → 跳过 p
```

**为什么要检查 independent？** 如果 p 还有子节点没被处理，就不能把 p 纳入当前子图，否则会破坏依赖关系。只有 p 的所有下游（children）都已归属当前子图时，p 才可以安全归入。

举个例子：

```
    A
   / \
  B   C
   \ /
    D
```

从 D 开始向上搜索。D 的 parent 是 B 和 C：
- 看 B：B 的 children 只有 D，D 已访问 → independent → 收入 B
- 看 C：C 的 children 只有 D，D 已访问 → independent → 收入 C
- 从 B 看 A：A 的 children 是 B 和 C，C 已访问 → independent → 收入 A
- 结果：{D, B, C, A} 构成一个子图

### 2. `search_subgraphs()` — 对整个图分组

```cpp
static std::vector<std::vector<NodePtr>> search_subgraphs(GraphPtr g)
```

```
逆拓扑序遍历所有节点（rbegin → rend，从输出到输入）
  对每个未访问的节点 n：
    以 n.is_dynamic() 为分类依据
    调 search_subgraph() 收集一组节点 → 加入 subgraphs
```

结果是一个二维数组：`subgraphs[i]` = 第 i 个子图的节点列表。**关键分割条件是 `is_dynamic`**：动态 shape 节点和静态 shape 节点必须分到不同子图。

### 3. `merge_subgraphs()` — 合并可以合并的子图

```cpp
static void merge_subgraphs(GraphPtr g)
```

**为什么需要合并？** 第一轮 `divide_graph` 已经把所有节点包成 SubGraphNode 了。但此时顶层图的节点全是 SubGraphNode，再跑一次 `search_subgraphs` 可能发现某些相邻子图可以进一步合并（它们 `is_dynamic` 属性相同且依赖关系允许）。合并过程：
1. 把多个 SubGraphNode `unfold()`（展开为原始算子节点）
2. 用展开后的全部节点创建一个新的大 SubGraphNode
3. 重复直到没有可合并的（`while(true)` 循环）

### 4. `record_io_tensors()` — 标记 IO tensor

```cpp
static void record_io_tensors(SubGraphNodePtr sgn)
```

遍历子图内节点，把 `Input` / `SubGraphInput` 的输出 tensor 和子图的输出 tensor 标记为 `io_tensor=true`。这个标记告诉后续的内存分配器：**这些 tensor 是子图边界上的 tensor，不能随意优化掉**。

---

## 子图数量

最终的子图数量取决于图的拓扑结构，不一定只有两个。如果动态和静态节点**交替出现**，就会切出多个子图：

```
Input (static)
Conv (static)          ──┐
DynReshape (dynamic)   ──┤ 动态静态交替
FC (static)            ──┤
DynSoftmax (dynamic)   ──┘
Output (static)

结果：
  SubGraph_0 (static):   Input, Conv
  SubGraph_1 (dynamic):  DynReshape
  SubGraph_2 (static):   FC
  SubGraph_3 (dynamic):  DynSoftmax, Output
```

但如果所有静态节点连在一起、所有动态节点也连在一起，`merge_subgraphs` 会把同类子图合并，最终可能就是两个子图。

**规则是：相邻且同类（都是 static 或都是 dynamic）的节点尽量合到一个子图，但中间被异类节点隔开的就没法合并。**
