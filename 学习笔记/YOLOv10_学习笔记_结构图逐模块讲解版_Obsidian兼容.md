# YOLOv10 学习笔记：结合网络结构图逐模块讲解版

> 论文：**YOLOv10: Real-Time End-to-End Object Detection**  
> 用途：按网络结构图，从 **输入 → Backbone → Neck/PAN → Dual Heads → 推理输出** 的顺序，把 YOLOv10 的关键模块逐个讲清楚，并结合公式说明每个模块为什么这样设计。

---

# 1. 先建立整体图景：YOLOv10 的结构到底长什么样

可以把 YOLOv10 拆成两条主线来看：

## 1.1 主线一：检测流程主干
整体检测流程仍然保持 YOLO 系列的经典范式：

```text
Input
  ↓
Backbone
  ↓
Neck / PAN
  ↓
Detection Heads
  ├─ One-to-many head（训练辅助）
  └─ One-to-one head（推理保留）
```

这个框架在论文图 2(a) 里画得很清楚：  
**Backbone 和 PAN 仍然是共享的特征提取主干，只是在 Head 部分拆成了两个分支。**

---

## 1.2 主线二：YOLOv10 的创新点分布在哪里
YOLOv10 的创新不是只在某一个模块，而是分散在整个网络路径上：

### 在 Head 层面
- 用 **Dual Assignments** 解决 NMS-free 训练
- 用 **Consistent Matching Metric** 保证双头监督一致

### 在 Backbone / Neck / Downsampling 层面
- 用 **CIB**
- 用 **rank-guided block design**
- 用 **spatial-channel decoupled downsampling**

### 在能力增强模块层面
- 用 **large-kernel convolution**
- 用 **PSA（Partial Self-Attention）**

所以 YOLOv10 不是“加一个注意力模块”那么简单，而是：
> **把训练机制、结构效率、全局建模一起重新设计。**

---

# 2. 从图 2 开始：双头结构到底是怎么工作的

论文图 2(a) 是理解 YOLOv10 的第一关键图。
![[Pasted image 20260328211541.png]]

---

## 2.1 图 2(a) 的结构解读

图 2(a) 表示：

```text
Input
  ↓
Backbone
  ↓
PAN
  ├─ One-to-many Head
  │    ├─ Regression
  │    └─ Classification
  └─ One-to-one Head
       ├─ Regression
       └─ Classification
```

### 这里最关键的结构事实
- **Backbone 和 PAN 是共享的**
- 分叉只发生在 **检测头**
- 两个 head 的结构相同，但**标签分配方式不同**

也就是说，YOLOv10 不是搞了两个完整网络，而是：
- 用同一套特征提取主干
- 接两个训练目标不同的输出头

这样参数和推理成本不会翻倍。

---

## 2.2 为什么要做成双头，而不是直接 one-to-one？

因为 one-to-one 有天然问题：监督太稀疏。

### 传统 YOLO 的 one-to-many
对每个 GT，会给多个正样本。  
优点：
- 训练信号丰富
- 收敛更稳
- 精度通常更高

缺点：
- 推理时会产生很多重复框
- 必须做 NMS

### 纯 one-to-one
每个 GT 只匹配一个预测。  
优点：
- 输出天然无重复
- 可以 NMS-free

缺点：
- 正样本太少
- 训练更难
- 容易掉精度

### YOLOv10 的折中方法
训练时同时保留：
- one-to-many：负责“丰富监督”
- one-to-one：负责“最终部署形式”

推理时只保留：
- one-to-one head

所以本质上是：

> **用 one-to-many 帮 one-to-one 学习。**

---

# 3. Dual Assignments 的数学本质：为什么双头有用

---

## 3.1 统一匹配分数公式

论文定义匹配分数为：

$$
m(\alpha,\beta)=s \cdot p^{\alpha}\cdot IoU(\hat b,b)^{\beta}
$$
其中：

- $p$：分类分数
- $\hat b$：预测框
- $b$：真实框
- $IoU(\hat b,b)$：预测框和 GT 的重叠程度
- $s$：空间先验（预测点是否落在目标区域内）
- $\alpha,\beta$：平衡分类项与回归项的重要程度

---

## 3.2 这个公式在结构图里对应什么位置？

它并不是 Backbone 里的公式，而是发生在 **Head 输出后，做标签分配时**。

也就是：

```text
Backbone/PAN 提取特征
   ↓
Head 输出分类分数 p 和框预测 \hat b
   ↓
通过 m(\alpha,\beta) 计算“预测-真值”的匹配质量
   ↓
决定谁是正样本
```

所以，结构图中的两个 head，不是简单多了两个输出层，  
而是意味着：
- **同样的网络输出**
- 会被送进两套 assignment 规则中

---

## 3.3 one-to-many 与 one-to-one 在公式上的区别

论文写成：

$$
m_{o2m} = m(\alpha_{o2m},\beta_{o2m})
$$
$$
m_{o2o} = m(\alpha_{o2o},\beta_{o2o})
$$
表面看只是参数不同，实际上意味着：

- one-to-many head 用一套“选样本标准”
- one-to-one head 用另一套“选样本标准”

如果这两套标准不一致，问题就来了：

> 同一个 GT，one-to-many 认为 A 是最好样本，one-to-one 却认为 B 更好。  
> 那共享的 Backbone/PAN 就会收到互相打架的梯度。

这就是论文要解决的“监督不一致”。

---

# 4. Consistent Matching Metric：图 2 右半部分到底在说明什么

图 2(a) 右侧和图 2(b) 一起，说明了 “一致匹配度量” 的必要性。

---

## 4.1 监督差距公式

论文把 two heads 的分类监督差距写成：

$$
A = t_{o2o,i} - \mathbb{I}(i\in \Omega)t_{o2m,i}
+ \sum_{k\in \Omega\setminus\{i\}} t_{o2m,k}
$$
其中：

- $\Omega$：one-to-many 为某个 GT 选出的正样本集合
- $i$：one-to-one 选中的那个唯一正样本
- $t_{o2o,i}$：one-to-one 对样本 $i$ 的分类目标
- $t_{o2m,i}$：one-to-many 对样本 $i$ 的分类目标
- $\mathbb I(\cdot)$：指示函数

---

## 4.2 这个公式的结构含义

这个式子不是在说网络结构多了什么层，  
而是在解释 **双头共享 Backbone/PAN 时，为什么必须让它们“偏好同一类样本”。**

核心结论是：

- 如果 one-to-one 选中的样本 $i$ 恰好在 $\Omega$ 里，
  并且它还是 one-to-many 中最优的那个正样本，
  那么监督差距 $A$ 最小。

换句话说：

> **最好让 one-to-one 选中的那个唯一框，就是 one-to-many 最想选的那个框。**

---

## 4.3 一致匹配条件

作者进一步推导出，只要令：

$$
\alpha_{o2o}=r\alpha_{o2m},\qquad
\beta_{o2o}=r\beta_{o2m}
$$
就有：

$$
m_{o2o}=m_{o2m}^{\,r}
$$
因为幂函数是单调的，所以：
- one-to-many 的排序
- one-to-one 的排序

就会一致。

论文默认取：

$$
r=1
$$
于是得到最简单形式：

$$
\alpha_{o2o}=\alpha_{o2m},\qquad
\beta_{o2o}=\beta_{o2m}
$$
这就是 **consistent matching metric**。

---

## 4.4 用图来理解这件事
图 2(b) 统计的是：

> one-to-one 选中的样本，在 one-to-many 排名前 1/5/10 的频率

结果显示：
- 用 consistent metric 时，重合频率更高
- 也就是说两个 head 对“谁是最好样本”的判断更一致

这就是为什么这个公式不是“数学点缀”，而是直接服务于共享特征主干的训练稳定性。

---

# 5. 从图 3 开始：Backbone/Block 级别到底改了什么

图 3 是理解 YOLOv10 结构设计的第二关键图。  
它分三部分：

- 图 3(a)：不同 stage 的 intrinsic rank
- 图 3(b)：CIB 结构
- 图 3(c)：PSA 结构

---

# 6. 图 3(a)：为什么要做 rank-guided block design

---

## 6.1 图 3(a) 在画什么
图 3(a) 横轴是不同 stage，纵轴是归一化后的 intrinsic rank。  
作者发现两个规律：

1. **越深的 stage，rank 越低**
2. **模型越大，深层 stage 的 rank 越低**

低 rank 表示什么？

> 表示该 stage 的特征/卷积权重存在更高冗余。

也就是说：
- 这些 stage 未必需要很重的 block；
- 可以换成更紧凑的结构。

---

## 6.2 Rank 是怎么来的
论文用卷积权重矩阵的数值秩来衡量。  
如果一个卷积层权重 reshape 后为：

$$
W \in \mathbb{R}^{C_o \times (K^2 C_i)}
$$
则看它有多少个奇异值大于阈值。  
这个数值秩越低，说明真正有用的独立方向越少，冗余越高。

---

## 6.3 为什么这会影响结构设计
传统 YOLO 常对所有 stage 用同一种 block。  
这意味着：

- 浅层、深层用一样的复杂度
- 冗余大的 stage 也被分配了很重的容量

作者认为这不合理，所以提出：

> **根据 rank 来决定哪些 stage 可以换成更紧凑 block。**

这就是 **rank-guided block design**。

---

# 7. 图 3(b)：CIB 到底是什么结构

---

## 7.1 先看结构图
图 3(b) 展示的 CIB（Compact Inverted Block）大致可以理解为：

```text
1×1
↓
3×3 DW
↓
1×1
↓
3×3 DW
↓
1×1
```

其中：
- DW = depthwise convolution

它是一个强调：
- **cheap spatial mixing**
- **cheap channel mixing**

的轻量 block。

---

## 7.2 CIB 每层各自干什么

### 第一层 $1\times1$
做通道变换，把特征投影到更适合处理的通道空间。

### 第一层 $3\times3$ DW
只在各自通道内做空间卷积，成本很低。

### 中间 $1\times1$
重新做通道混合。

### 第二层 $3\times3$ DW
再做一次轻量空间建模。

### 最后 $1\times1$
融合输出通道。

---

## 7.3 为什么 CIB 比普通 block 更高效
标准卷积复杂度大致为：

$$
O(K^2 C_{in} C_{out} HW)
$$
而 depthwise conv 复杂度大致为：

$$
O(K^2 C HW)
$$
pointwise conv 复杂度为：

$$
O(C_{in} C_{out} HW)
$$
所以 CIB 的核心是：

> 把“昂贵的空间+通道联合混合”拆成  
> “便宜的空间混合（DW） + 必要的通道混合（PW）”

这在冗余大的 stage 上非常划算。

---

## 7.4 Rank-guided 策略是怎么用 CIB 的
不是所有 stage 都替换成 CIB，而是：

1. 计算每个 stage 的 rank
2. 从最低 rank 的 stage 开始尝试替换成 CIB
3. 只要 AP 不掉，就继续
4. 一旦 AP 掉，就停止

所以 CIB 不是全局替代，而是**按 stage 精准部署**。

---

# 8. 下采样模块：为什么要做 spatial-channel decoupled downsampling

这部分虽然不在图 3 中单独画出，但它属于 Backbone/Stage 里的关键结构替换。

---

## 8.1 传统下采样怎么做
传统 YOLO 常用一个 stride=2 的 $3\times3$ 标准卷积同时完成：

- 空间分辨率下降：
$$
H\times W \rightarrow \frac{H}{2}\times \frac{W}{2}
$$
- 通道数上升：
$$
C \rightarrow 2C
$$
其复杂度量级为：

$$
O\left(\frac{9}{2}HWC^2\right)
$$
参数量量级为：

$$
O(18C^2)
$$
---

## 8.2 YOLOv10 如何改
作者把它拆成两步：

### 第一步：pointwise conv
先做通道变换：

$$
C \rightarrow 2C
$$
### 第二步：depthwise conv with stride 2
再做空间降采样：

$$
H\times W \rightarrow \frac{H}{2}\times \frac{W}{2}
$$
---

## 8.3 新复杂度
论文给出复杂度量级：

$$
O\left(2HWC^2 + \frac{9}{2}HWC\right)
$$
参数量量级：

$$
O(2C^2 + 18C)
$$
---

## 8.4 为什么先调通道再降采样更合理
因为如果先降采样，很多细节已经丢了，再扩通道也回不来。  
而先扩通道，再做 DW 下采样，可以让更多信息以不同通道形式被保留下来。

所以这个设计不仅更省，还更稳。

---

# 9. Accuracy 分支之一：大核卷积放在哪里，为什么这样放

---

## 9.1 大核卷积不应该到处用
大核 depthwise conv 的作用是扩大感受野。  
如果把核从 $3\times3$ 换成 $7\times7$，理论上更能看大范围上下文。

但问题是：

- 浅层特征图很大
- 大核会增加 I/O 和延迟
- 小目标细节可能被污染

所以作者没有全局替换。

---

## 9.2 YOLOv10 的结构决策
在 **CIB 的深层位置**，把第二个 DW 卷积换成大核：

$$
3\times3 \rightarrow 7\times7
$$
并在训练时加入重参数化分支帮助优化。

---

## 9.3 为什么只放在深层
深层特征图分辨率低，所以：
- 大核计算代价没那么夸张
- 更适合建模大范围上下文
- 不会严重影响浅层小目标细节

所以这是一个典型的“按层选模块”设计。

---

# 10. 图 3(c)：PSA 到底是什么结构

---

## 10.1 看结构图
图 3(c) 中 PSA 大致是：

```text
1×1
↓
Split
 ├─ identity branch
 └─ N_PSA × [MHSA + FFN]
↓
Concat
↓
1×1
```

这就是 **Partial Self-Attention**。

---

## 10.2 为什么叫 Partial
因为它不是把所有通道都送进自注意力，  
而是：

- 先把通道分成两半
- 只对其中一半做 MHSA + FFN
- 另一半走轻量旁路
- 最后再拼接融合

所以叫 partial。

---

## 10.3 标准自注意力公式

$$
Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
若 token 数为 $N$，注意力复杂度近似：

$$
O(N^2 d)
$$
这对高分辨率检测来说太贵。

---

## 10.4 PSA 为什么能省
因为只对部分通道做 attention，相当于减少了：
- 参与 Q/K/V 投影的通道数
- attention 的中间表示成本
- FFN 的处理成本

所以它保留了部分全局建模能力，但不会像整块 Transformer 那么贵。

---

## 10.5 PSA 结构图里的每一步在做什么

### 第一个 $1\times1$
把输入映射到适合后续拆分和融合的通道空间。

### Split
按通道一分为二：

$$
X \rightarrow [X_1, X_2]
$$
### PSA branch
只对 $X_2$ 做若干层：

$$
X_2 \rightarrow \text{MHSA} \rightarrow \text{FFN}
$$
### Identity / light branch
$X_1$ 基本不做重计算，保留局部卷积特征。

### Concat + $1\times1$
最终：

$$
Y = Conv_{1\times1}([X_1,\; \text{PSA}(X_2)])
$$
这样就把局部特征和全局特征重新融合起来。

---

## 10.6 为什么 PSA 比直接上 Transformer 更适合 YOLO
论文实验发现：
- PSA 比直接 Transformer block 更快
- 还能得到更好的 AP

直觉原因是：
- 检测 backbone 本来就是 CNN 风格
- 全量注意力太重且有冗余
- partial attention 更符合实时检测需求

---

# 11. PSA 放在网络的什么位置最合理

论文指出，PSA 只放在**最低分辨率 stage 后**。

原因很简单：

若特征图尺寸为 $h\times w$，token 数：

$$
N = h\cdot w
$$
注意力复杂度和 $N^2$ 成正比。  
所以高分辨率 stage：
- token 太多
- 成本爆炸

低分辨率 stage：
- token 少
- 更适合做全局关系建模

这也是为什么 PSA 是“按位置精确投放”的模块，而不是 everywhere attention。

---

# 12. Head 模块：分类头和回归头为什么区别对待

---

## 12.1 论文的观察
作者发现，在 YOLOv8-S 中：
- 分类头 FLOPs 和参数都明显高于回归头
- 但误差分析说明，回归误差更影响最终 AP

也就是：

> **框回归比类别预测更像瓶颈。**

---

## 12.2 因此结构上怎么改
YOLOv10 对分类头做轻量化，用两个 depthwise separable conv + 一个 $1\times1$ conv。

而回归头不做同样幅度的压缩，因为它更关键。

---

## 12.3 这和结构图怎么对应
在图 2(a) 的每个 head 中，都有：
- Classification
- Regression

YOLOv10 的设计思想不是“两个都一样重”，而是：
- **Classification 更轻**
- **Regression 保持更强表达能力**

这说明 Head 级别也做了任务感知设计。

---

# 13. 把所有模块连起来：YOLOv10 的完整逐模块理解

现在可以从网络结构图出发，把整个 YOLOv10 串成一个清晰流程。

---

## 13.1 输入到 Backbone
输入图像进入 Backbone。  
Backbone 由多 stage 组成，其中：

- 某些 stage 使用普通 block
- 某些 rank 低、冗余高的 stage 用 **CIB**
- 小模型的深层 stage 里，CIB 可带 **大核 DW 卷积**
- 下采样使用 **spatial-channel decoupled downsampling**

所以 Backbone 不是简单照搬 YOLOv8，而是做了分 stage 优化。

---

## 13.2 Backbone 到 PAN
PAN 负责多尺度特征融合。  
它将不同分辨率特征进行上采样/下采样和融合，构造更适合检测的多尺度表示。

YOLOv10 的关键不是重写 PAN 思想，而是：
- 让输入 PAN 的 backbone 特征更高效
- 在深层适当加入 PSA，提高全局建模能力

---

## 13.3 PAN 到 Dual Heads
PAN 输出多尺度特征后，送入两套 head：

### one-to-many head
- 用于训练辅助
- 给 Backbone/PAN 提供密集监督

### one-to-one head
- 用于训练 + 推理
- 推理时保留它，直接输出结果

两个 head 的结构相同，但 assignment 不同。

---

## 13.4 Assignment 机制
每个 head 输出：
- 分类分数 $p$
- 边框预测 $\hat b$

再通过匹配分数：

$$
m(\alpha,\beta)=s\cdot p^\alpha \cdot IoU(\hat b,b)^\beta
$$
完成标签分配。

为了让 two heads 一致，约束：

$$
\alpha_{o2o}=r\alpha_{o2m},\qquad
\beta_{o2o}=r\beta_{o2m}
$$
默认 $r=1$。

---

## 13.5 推理输出
推理时：
- 删除 one-to-many head
- 保留 one-to-one head
- 直接输出单预测结果
- **不需要 NMS**

所以 YOLOv10 的 end-to-end，并不是靠 DETR 式 query 机制实现，而是：
> **用双头训练把 YOLO 训成一个可以 one-to-one 输出的检测器。**

---

# 14. 这版结构图讲解，你最该记住的 6 个点

## 点 1：图 2(a) 是论文第一核心图
它告诉你：
- Backbone/PAN 共享
- Head 分成 two branches
- 训练双头，推理单头

---

## 点 2：双头不是为了堆参数，而是为了兼顾两种监督
- one-to-many：训练友好
- one-to-one：部署友好

---

## 点 3：公式不是独立存在的
公式

$$
m(\alpha,\beta)=s\cdot p^\alpha \cdot IoU(\hat b,b)^\beta
$$
对应的是 **Head 之后的标签分配机制**。

---

## 点 4：图 3(a) 说明不是所有 stage 都一样重要
rank 低的 stage 冗余更高，适合换成 CIB。

---

## 点 5：图 3(b) 的 CIB 是效率核心块
用 DW + PW 拆解昂贵卷积，实现轻量化。

---

## 点 6：图 3(c) 的 PSA 是精度增强核心块
只对部分通道做 attention，把全局建模的收益和代价平衡得更好。

---

# 15. 一页速记版

## YOLOv10 结构主线
```text
Input
 ↓
Backbone（部分 stage 用 CIB / 大核 / 解耦下采样）
 ↓
PAN（多尺度融合）
 ↓
Dual Heads
 ├─ one-to-many（训练辅助）
 └─ one-to-one（推理保留）
 ↓
NMS-free output
```

## 关键公式
### 匹配分数
$$
m(\alpha,\beta)=s\cdot p^\alpha \cdot IoU(\hat b,b)^\beta
$$
### 一致匹配条件
$$
\alpha_{o2o}=r\alpha_{o2m},\qquad
\beta_{o2o}=r\beta_{o2m}
$$
### 注意力
$$
Attention(Q,K,V)=Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
---

# 16. 最后一句话总结

如果你用“结构图视角”看 YOLOv10，它的本质就是：

> **共享 Backbone/PAN，双头协同训练；  
> 用一致匹配让 one-to-one 学得像 one-to-many 一样稳；  
> 再通过 CIB、解耦下采样、大核卷积和 PSA，把每个模块放到最划算的位置。**

