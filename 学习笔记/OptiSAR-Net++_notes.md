# OptiSAR-Net++ 论文笔记

> 论文：**OptiSAR-Net++: A Large-Scale Benchmark and Transformer-Free Framework for Cross-Domain Remote Sensing Visual Grounding**

## 1. 论文要解决什么问题

### 1.1 任务背景：RSVG
遥感视觉指代定位（RSVG, Remote Sensing Visual Grounding）要求模型根据自然语言描述，在遥感图像中定位对应目标。

例如：
- “图像左上角的小型 SAR 船只”
- “右侧的 optical transmission tower”

相比普通目标检测，RSVG 不仅要识别类别，还要理解：
- 类别语义
- 属性语义
- 空间方向语义
- 模态语义（optical / SAR）

---

### 1.2 这篇论文提出的新任务：CD-RSVG
作者提出 **CD-RSVG（Cross-Domain Remote Sensing Visual Grounding）**，即跨域遥感视觉指代定位。

核心思想：

- 以往方法通常只在单一模态上做 grounding：
  - optical-only
  - SAR-only
- 现实中 optical 和 SAR 具有互补性：
  - optical：纹理和语义丰富，但依赖天气和光照
  - SAR：全天时全天候，但存在散斑噪声和几何失真

因此论文希望在一个统一框架下，利用 optical 和 SAR 两种遥感数据共同训练，学习跨域共享知识，并完成自然语言引导的目标定位。

---

## 2. 论文认为的核心困难

### 2.1 跨域特征建模困难
optical 和 SAR 的成像机制差异很大：

- optical 更依赖纹理、颜色、边缘
- SAR 更依赖结构、散射特性，且噪声强

如果所有参数完全共享，容易出现负迁移：
- 共享表示同时适配两种分布，边界会变模糊
- SAR 特征和 optical 纹理语义会相互干扰

---

### 2.2 计算效率瓶颈
现有很多 RSVG 方法依赖 Transformer：
- 跨模态 fusion 重
- 解码器复杂
- 训练和推理成本高

作者想用一种更轻量的方案替代重型 Transformer。

---

### 2.3 遥感场景本身就难
RSVG 在遥感里还有额外难点：

- 目标尺度变化大
- 同类目标密集分布
- 背景复杂
- 方位词非常关键，例如：
  - left / right
  - upper / lower
  - middle / corner

所以模型必须同时学会：
- “是什么” —— 类别与属性
- “在哪里” —— 空间区域与方向
- “属于哪个模态语义” —— optical / SAR

---

## 3. 论文的两大贡献

### 3.1 构建数据集：OptSAR-RSVG
作者构建了第一个大规模 CD-RSVG 基准数据集 **OptSAR-RSVG**。

最终规模：
- $46{,}825$ 张图像
- $90{,}148$ 个 image-text-box 三元组
- 共 $16$ 个类别

其中：
- optical：
  - $34{,}359$ 张图像
  - $69{,}970$ instances
  - $14$ 类
- SAR：
  - $12{,}466$ 张图像
  - $20{,}178$ instances
  - $2$ 类

---

### 3.2 提出新模型：OptiSAR-Net++
作者提出 **OptiSAR-Net++**，这是一个 **Transformer-free** 的 CD-RSVG 框架。

核心组件包括：
- **PL-MoE**：跨域特征建模
- **CLIP-based contrastive grounding**：高效 grounding
- **TGDF-SSA**：显式文本引导视觉特征注入
- **Region-aware auxiliary head**：显式空间区域建模

---

## 4. 数据集 OptSAR-RSVG 详解

### 4.1 数据来源
作者整合了 4 个单源数据集：

- RSVGD（optical ships）
- OPT-RSVG（14 类 optical targets）
- SARVG（SAR ships）
- TACMT（SAR transmission towers）

---

### 4.2 数据构建流程

#### Step 1：数据清洗
- 删除非法框
- 用检测器找疑似标注错误样本，再人工核查
- 统一类别命名
- 给类别加模态前缀，例如：
  - `Optical ship`
  - `SAR ship`

---

#### Step 2：数据增强与文本重写
为缓解类别不平衡，作者做了几何增强：

- horizontal flip
- vertical flip
- $180^\circ$ rotation

同时使用 GPT-4o 重写文本描述，并根据变换后的目标位置自动更新方向词。

例如，若图像翻转后，原本的 `upper left` 可能需要改成 `upper right`。

这一阶段新增：
- $9{,}046$ 张图像
- $19{,}136$ 条标注

---

#### Step 3：人工核查与划分
最终数据集按 $8:1:1$ 划分 train / val / test：

- train：
  - $37{,}460$ images
  - $72{,}118$ annotations
- val：
  - $4{,}682$ images
  - $9{,}015$ annotations
- test：
  - $4{,}683$ images
  - $9{,}015$ annotations

---

### 4.3 数据集特点

#### （1）尺度分布差异明显
- optical 目标尺度范围更广
- SAR 目标更多集中在中小尺度

因此模型必须具备较强的多尺度建模能力。

---

#### （2）模态分布不均衡
样本比例约为：
- optical：$77.6\%$
- SAR：$22.4\%$

这与真实遥感数据采集情况接近，但也给跨域学习带来挑战。

---

#### （3）文本描述更丰富
平均每条描述有 $11.13$ 个词，高于一些已有遥感 grounding 数据集。

说明该数据集并不是简单类别标注，而是包含更丰富的：
- 属性词
- 方向词
- 位置词
- 模态词

---

## 5. OptiSAR-Net++ 总体框架

整体流程可分为三层：

1. **共享 CNN Backbone + PL-MoE**
2. **Vision-Language Fusion Neck（TGDF-SSA）**
3. **Detection Heads**
   - box regression head
   - CLIP-based contrastive head
   - region-aware auxiliary head

一个很关键的点是：

> 这篇论文不是把 grounding 当成重型生成问题来做，而是把它转化为一个 **region-text matching** 问题。

也就是说：
- 先得到候选区域特征
- 再和文本 embedding 做匹配
- 选择匹配分数最高的区域作为 grounding 结果

---

## 6. PL-MoE：跨域特征建模模块

PL-MoE 全称：

**Patch-Level Low-Rank Adaptation Mixture of Experts**

它的目标是解决：
- optical 与 SAR 差异过大
- 完全共享表示容易发生负迁移

---

### 6.1 两个核心设计

#### 6.1.1 Patch-level sparse routing
给定特征图：

$X \in \mathbb{R}^{B \times C \times H \times W}$

将其划分为不重叠 patch：

$$
N = \frac{H}{p_h} \times \frac{W}{p_w}
$$

其中：
- $p_h$：patch 的高
- $p_w$：patch 的宽

即把 feature map 切成规则网格上的局部 patch $P_{ij}$，每个 patch 独立做 expert routing。

注意：
- patch 是在 **feature map** 上切，不是在原图上切
- 是 **non-overlapping patches**
- 不是 image-level routing，也不是 token-level dense routing

---

#### 6.1.2 Shared LoRA Experts
每个 expert 不是完全独立的大卷积模块，而是：

- 共享卷积变换：$C_{\text{shared}}$
- 每个 expert 只带一组轻量 LoRA 参数：$(A_e, B_e)$

第 $e$ 个 expert 的输出为：

$$
E_e(P) = C_{\text{shared}}(P) + \alpha \cdot P A_e B_e
$$

其中：
- $A_e \in \mathbb{R}^{C \times r}$
- $B_e \in \mathbb{R}^{r \times C}$
- $r \ll C$
- $\alpha$ 为可学习缩放因子

这样做的直觉是：

- 共享部分学跨域共性
- LoRA expert 学域特有偏置
- 以较低参数代价实现跨域解耦

---

### 6.2 路由时的相似度
在 PL-MoE 中，论文明确写的是：

- 使用 **cosine similarity-based Top-K gating**
- 并配合 load balancing loss 防止 expert collapse

也就是说，**MoE 路由这一步明确使用余弦相似度**。

---

### 6.3 Patch 是怎么切的
如果 feature map 尺寸为 $H \times W$：

- 当 $p = 1$ 时，相当于 token-level routing
- 当 $p = 2$ 时，是较细粒度 patch routing
- 当 $p = 4$ 时，patch 更大
- 当 $p = \infty$ 时，相当于 image-level routing

论文消融实验表明：
- image-level routing 最差
- token-level routing 也不错，但会破坏一定空间一致性
- **patch-level routing（尤其 $p=2$）最佳**

---

### 6.4 SAR 和 optical 是怎么处理的
这里特别容易误解。

论文里 **没有** 把一张 SAR 图和一张 optical 图先拼在一起做特征融合，也没有设计：

- 一条 optical 分支
- 一条 SAR 分支
- 然后再做显式 fusion

更准确地说：

- optical 样本和 SAR 样本在同一个统一框架中联合训练
- 每张图各自通过共享 backbone 得到自己的 feature map
- 再在各自的 feature map 上单独切 patch
- 这些 patch 进入同一套 PL-MoE 中统一路由和学习

所以它不是：
- optical-SAR pair fusion

而是：
- **shared backbone + expert decoupling 的 cross-domain representation learning**

---

## 7. CLIP-based Visual Grounding

### 7.1 核心思想
传统 RSVG 常常被当成生成式或复杂解码任务来做，而 OptiSAR-Net++ 把它转成一个检索式问题。

作者先从多尺度特征 $\{F^l\}$ 中生成视觉 embedding：

$$
E^l = g_{\text{vis}}(F^l) \in \mathbb{R}^{B \times D \times H_l \times W_l}
$$

文本端使用 CLIP text encoder，将候选描述编码为：

$$
T = \{t_k\}_{k=1}^{K}, \quad t_k \in \mathbb{R}^{D}
$$

之后计算 region-text 相似度图，作为分类 logits。

推理时，输入单条文本 embedding $t$，从候选区域中选择匹配分数最高的区域作为输出。

---

### 7.2 这里的相似度是不是余弦？
不能一概而论。

- **MoE routing**：明确是余弦相似度
- **TGDF-SSA**：明确写成内积
- **CLIP matching head**：论文正文没有把公式明确写死成 cosine similarity

所以最准确的结论是：

> 论文中多处“相似度”并不全都是余弦相似度；只有 MoE gating 这一处是明确写成 cosine similarity。

---

## 8. Fine-grained Adversarial Negative Sampling

这是论文里非常关键的设计之一。

### 8.1 为什么需要 hard negatives
普通对比学习里，如果负样本和正样本差得太远，模型不会真正学会细粒度语义。

但是在遥感 grounding 中，真正困难的负样本往往只差一点点，例如：

- `upper left` $\leftrightarrow$ `upper right`
- `SAR ship` $\leftrightarrow$ `optical ship`

这些文本词面相似，但语义上是关键性的错误。

---

### 8.2 论文怎么做
作者构造对抗负样本：
- 替换方向词
- 替换模态词
- 构造与正样本非常接近但语义相反的文本

这样模型就被迫真正理解：
- 方位语义
- 模态语义
- 细粒度语义差异

---

### 8.3 为什么效果强
从消融实验看，这个模块带来的提升最大，说明论文性能提升很大程度上来自：

> 不是模型更大，而是监督信号更精准。

---

## 9. TGDF-SSA：显式视觉-语言融合

TGDF-SSA 全称：

**Text-Guided Dual-Gate Fusion with Spatial Shuffle Attention**

它负责把文本语义显式注入到视觉特征中。

---

### 9.1 动机
作者认为，传统多层 cross-attention 在遥感图像里有两个问题：

- 对空间结构建模不够显式
- 在密集目标场景中容易产生定位歧义

因此他们提出一个更轻量、更可控的显式语义注入模块。

---

### 9.2 具体形式
先把视觉特征投影到多头空间：

$$
\phi(V) \in \mathbb{R}^{B \times n_h \times d_h \times H \times W}
$$

再把文本 embedding 映射成每个 head 的指导向量：

$$
\psi(T) \in \mathbb{R}^{B \times N \times n_h \times d_h}
$$

计算视觉-文本相似度：

$$
A_{b,m,h,w} = \max_{n \in [1,N]} \left\langle \phi(V)_{b,m,:,h,w}, \psi(T)_{b,n,m,:} \right\rangle
$$

这里使用的是 **内积**，不是论文明确声明的余弦相似度。

再做温度和偏置校准：

$$
\tilde{A}_{b,m,h,w} = \frac{A_{b,m,h,w}}{\tau} + b_m
$$

$$
A^\sigma = \sigma(\tilde{A}) \cdot s
$$

最终通过门控机制更新视觉特征：

$$
\tilde{V} = V + \beta \cdot V \odot [\alpha \cdot (A^\sigma - 1)]
$$

其中：
- $\tau$：可学习温度
- $b_m$：每个 head 的 bias
- $\sigma(\cdot)$：Sigmoid
- $s$：可学习缩放项

---

### 9.3 它到底融合了什么
这里要特别明确：

- **不是 optical feature 和 SAR feature 的显式融合**
- 而是 **每个样本自己的视觉特征与文本特征做融合**

也就是说：
- optical 与 SAR 之间的跨域交互主要通过共享 backbone / PL-MoE 实现
- TGDF-SSA 负责的是 **vision-language fusion**

---

## 10. Region-aware Auxiliary Head

为了进一步增强空间定位能力，作者引入了一个仅在训练阶段启用的区域辅助头。

---

### 10.1 思想
将特征图上的每个位置分配到预定义的空间网格中，让模型显式学习空间区域先验：

- upper left
- upper right
- middle
- bottom
- 等

---

### 10.2 标签定义
对特征图位置 $(i,j)$，其网格标签为：

$$
y_{i,j} = \text{row}(i) \cdot c + \text{col}(j), \quad y_{i,j} \in \{0, \dots, G-1\}
$$

其中：
- $c$：网格列数
- $G$：总网格数

---

### 10.3 区域损失
给定预测：

$$
P = h_{\text{grid}}(Z) \in \mathbb{R}^{B \times G \times H_l \times W_l}
$$

区域辅助损失为：

$$
L_{\text{region}} =
\frac{1}{BHW}
\sum_{b=1}^{B}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
\text{CE}(P_{b,:,i,j}, y_{i,j})
$$

作者把这一路理解为学习 **Where**，而 TGDF-SSA 更偏向学习 **What**。

---

## 11. 总损失函数

整体目标函数包括四部分：

- 回归损失：$L_{\text{reg}}$
- CLIP 分类损失：$L_{\text{cls}}$
- 空间辅助损失：$L_{\text{region}}$
- MoE 负载均衡正则：$L_{\text{lb}}$

其中回归损失还结合了：
- IoU-based loss
- Distribution Focal Loss（DFL）

---

## 12. 实验设置

### 12.1 评测数据集
论文在两个数据集上评测：

1. **OptSAR-RSVG**
2. **DIOR-RSVG**

---

### 12.2 评价指标
使用以下指标：

- $\text{Pr@0.5}$
- $\text{Pr@0.6}$
- $\text{Pr@0.7}$
- $\text{Pr@0.8}$
- $\text{Pr@0.9}$
- $\text{meanIoU}$
- $\text{cumIoU}$

其中：

$$
\text{Pr@t} =
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}\big(\text{IoU}(\hat{b}_i, b_i) \ge t\big) \times 100\%
$$

$$
\text{meanIoU} =
\frac{1}{N}
\sum_{i=1}^{N}
\text{IoU}(\hat{b}_i, b_i) \times 100\%
$$

$$
\text{cumIoU} =
\frac{
\sum_{i=1}^{N} \text{Area}(\hat{b}_i \cap b_i)
}{
\sum_{i=1}^{N} \text{Area}(\hat{b}_i \cup b_i)
}
\times 100\%
$$

---

### 12.3 训练细节
作者的实现细节包括：

- optimizer：AdamW
- 初始学习率：$2 \times 10^{-3}$
- weight decay：$0.025$
- momentum：$0.9$
- cosine decay
- 共训练 $300$ epochs
- 使用 $8$ 张 RTX 4090

模型总参数：
- $95.6$M

其中冻结的 MobileCLIP2 文本分支约：
- $63.4$M

真正参与训练的参数约：
- $32.2$M

---

## 13. 主要实验结果

## 13.1 OptSAR-RSVG 上的 SOTA 表现
在 OptSAR-RSVG 上，OptiSAR-Net++ 达到：

- $\text{meanIoU} = 82.76\%$
- $\text{cumIoU} = 90.70\%$

超过了先前方法，同时参数量更少。

在 optical test set 上：
- $\text{Pr@0.9} = 68.89\%$

说明其高精度框定位能力尤其强。

---

### 13.2 数据效率很高
只用部分训练数据时，性能依然很强：

- 用 $60\%$ 数据，optical meanIoU 依然超过很多 baseline 的全数据训练结果
- 用 $30\%$ 数据时，也能接近甚至超过一些更大模型的全数据结果

说明该方法具有很好的 sample efficiency。

---

### 13.3 空间语义更稳健
论文还做了文本 masking 实验：
- 把一部分描述替换成仅有类别名
- 测试模型对空间语义缺失的鲁棒性

结果显示：
- OptiSAR-Net++ 的性能下降相对较小
- 说明它对类别、位置、空间语义的利用更均衡

---

### 13.4 在 DIOR-RSVG 上也有很好泛化
在单域 optical benchmark DIOR-RSVG 上，OptiSAR-Net++ 依然表现很强。

尤其在：
- $\text{Pr@0.9}$
- $\text{cumIoU}$

上取得最佳或接近最佳结果。

这说明该框架不仅适用于 cross-domain setting，也能很好泛化到单域场景。

---

## 14. 消融实验结论

## 14.1 各模块贡献
论文依次加入以下模块：
- adversarial negative sampling
- PL-MoE
- TGDF-SSA + auxiliary head

结果表明：

1. **对抗负样本** 带来的提升最大  
   说明细粒度语义监督非常关键

2. **PL-MoE** 提供稳定增益  
   用很小的参数开销改善跨域表示

3. **TGDF-SSA + auxiliary head** 继续提升  
   说明显式 semantic injection 和 spatial supervision 确实有效

---

## 14.2 MoE 配置结论
论文比较了：
- expert 数量 $n$
- Top-$k$ 选择
- patch 粒度 $p$

最终结论：

- $n=8$ 更好
- $k=2$ 最优
- patch-level routing 最优
- $p=2$ 在性能和稳定性之间平衡最好

可以理解为：
- image-level routing 太粗
- token-level routing 太细
- patch-level routing 刚好兼顾灵活性与空间一致性

---

## 15. 你问过的几个关键问题整理

## 15.1 Patch-level sparse routing 是怎么分 patch 的？
在 backbone 输出的 feature map 上切不重叠 patch：

$$
N = \frac{H}{p_h} \times \frac{W}{p_w}
$$

不是在原图上切，也不是把 optical 和 SAR 拼起来再切。

---

## 15.2 SAR 和 optical 图像是一起切的吗？
不是。

更准确地说：
- 每张图各自通过共享 backbone 得到自己的 feature map
- 再各自切 patch
- 这些 patch 再进入同一个 PL-MoE 系统统一路由

所以不是图像级配对融合，而是共享参数下的跨域表示学习。

---

## 15.3 SAR 和 optical 有没有显式特征融合？
没有那种典型的：
- optical branch
- SAR branch
- 再做 cross-modal feature fusion

论文更像是：
- **SAR / optical 在共享 backbone + PL-MoE 中做跨域建模**
- **每个样本各自和文本做融合**

也就是说，显式融合的是：
- **visual feature + text feature**

而不是：
- **optical feature + SAR feature**

---

## 15.4 多次计算相似度，都是余弦相似度吗？
不是。

最准确的区分如下：

1. **PL-MoE gating**
   - 明确是 cosine similarity

2. **TGDF-SSA**
   - 公式明确写的是内积：
     $$
     \left\langle \phi(V), \psi(T) \right\rangle
     $$

3. **CLIP matching head**
   - 论文正文没有明确把公式写死成 cosine similarity

所以不能说全文所有相似度都统一使用余弦相似度。

---

## 16. 这篇论文的创新点总结

### 创新点 1：提出 CD-RSVG 任务
把遥感 grounding 从单域扩展到跨域。

### 创新点 2：构建第一个大规模跨域 benchmark
为该方向提供了标准化评测基础。

### 创新点 3：提出 Transformer-free grounding 范式
通过 CLIP-style contrastive matching 实现高效定位。

### 创新点 4：设计适合遥感任务的关键模块
包括：
- PL-MoE
- adversarial negative sampling
- TGDF-SSA
- region-aware auxiliary head

这些模块都明显针对遥感场景中的：
- 方向语义
- 空间语义
- 跨域差异
- 高密度目标

---

## 17. 论文局限性

### 17.1 模态仍然较少
当前只验证了：
- optical
- SAR

还没有扩展到：
- infrared
- multispectral
- hyperspectral
- temporal sequence

---

### 17.2 长程依赖建模可能不如大型 Transformer
作者也承认，虽然 CLIP-based retrieval 更高效，但在更大规模场景中，长程依赖建模能力可能不如重型 Transformer。

---

### 17.3 SAR 类别仍偏少
虽然整体是 16 类，但 SAR 端只有 2 类，跨域语义的丰富度还有提升空间。

---

## 18. 最终总结

这篇论文最重要的意义不是单纯“刷高了指标”，而是同时完成了三件事：

1. **提出新任务：CD-RSVG**
2. **构建新基准：OptSAR-RSVG**
3. **提出新范式：轻量化、非 Transformer 的跨域 grounding 框架**

一句话概括：

> OptiSAR-Net++ 将遥感 visual grounding 从“单域 + 重 Transformer”推进到了“跨域 + 轻量对比学习”的新路线。

---

## 19. 适合答辩或汇报时的简版总结

**OptiSAR-Net++ 主要面向跨域遥感视觉指代定位问题。作者首先构建了首个 optical-SAR 联合的 CD-RSVG 数据集 OptSAR-RSVG；然后提出一个 Transformer-free 框架，通过 PL-MoE 做跨域特征解耦，通过 CLIP 风格对比学习将 grounding 转换为区域-文本匹配问题，并利用 adversarial negative sampling、TGDF-SSA 和 region-aware auxiliary head 增强细粒度语义理解和空间建模能力。实验表明，该方法在跨域数据集和公开单域数据集上都取得了很强的性能，同时具备更好的效率与数据利用率。**
