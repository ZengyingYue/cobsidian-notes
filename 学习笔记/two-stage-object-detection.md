# 目标检测与多尺度特征笔记

## 1. R-CNN

### 1.1 要解决的问题
目标检测不仅要判断“是什么”，还要判断“在哪里”。

输出通常包括：
- 类别
- 边界框位置

---

### 1.2 核心思路
R-CNN 的基本流程：

1. 用 **Selective Search** 生成候选区域（region proposals）
2. 把每个候选框从原图裁出来并 resize 到固定大小
3. 每个候选框单独送入 CNN 提取特征
4. 用 **SVM** 做分类
5. 用 **Bounding Box Regression** 修正框的位置
6. 用 **NMS** 去掉重复框

一句话概括：

> **Selective Search + CNN feature + SVM + bbox regression**

---

### 1.3 优点
- 首次把 CNN 特征成功用于目标检测
- 比传统手工特征方法精度提升明显
- 奠定了 two-stage detector 的基本范式

---

### 1.4 缺点
- **很慢**：每个 proposal 都要单独跑一次 CNN
- 重复计算严重：很多 proposal 高度重叠
- 训练流程复杂：CNN、SVM、bbox regressor 分阶段训练
- 不是端到端

---

### 1.5 本质问题
R-CNN 最核心的低效点在于：

> **卷积没有共享**

也就是同一张图中大量重叠区域被重复卷积很多次。

---

## 2. SPPnet

### 2.1 SPPnet 出现的原因
R-CNN 有两个主要问题：

1. 每个候选框都要单独跑 CNN，计算代价极高
2. CNN 后接全连接层，要求固定输入尺寸，所以每个 proposal 必须 resize，可能带来形变

SPPnet 的目标就是：
- **共享卷积计算**
- **支持任意尺寸候选区域**

---

### 2.2 核心思想
SPPnet 的流程：

1. 整张图只过一次卷积网络
2. 得到共享 feature map
3. 将候选框映射到 feature map 上
4. 对每个候选区域做 **Spatial Pyramid Pooling**
5. 输出固定长度特征
6. 后接分类与回归

一句话概括：

> **整图卷积一次，在 feature map 上对每个候选区域做 SPP**

---

### 2.3 什么是 Spatial Pyramid Pooling
SPP 的核心不是让每个 bin 的实际大小相同，而是让每层的 **bin 数量固定**。

例如三层金字塔：

- `1×1`
- `2×2`
- `4×4`

那么总输出 bin 数为：

$$1 + 4 + 16 = 21$$

如果通道数为 \(C\)，最终输出维度为：

$$21C$$

---

### 2.4 为什么不同尺寸 RoI 也能输出同样长度
例如两个 RoI 尺寸分别是：
- `13×17`
- `15×18`

它们做 `4×4` 池化时，都会被划分为 16 个 bin，但每个 bin 覆盖的实际区域大小不一定相同。

所以固定的是：
- 输出网格结构
- 输出维度

不是：
- 每个 bin 的物理尺寸

可记为：

> **SPP 固定的是输出网格数，不是每个网格的真实大小**

---

### 2.5 SPPnet 比 R-CNN 好在哪里
- 整张图只卷积一次，速度提升巨大
- 支持任意尺寸 RoI
- 保留多尺度空间信息
- 是从“逐框卷积”走向“共享特征图”的关键过渡

---

### 2.6 SPPnet 的不足
虽然 SPPnet 比 R-CNN 快很多，但还有两个重要问题：

1. **训练仍是 multi-stage**
2. **只能较好地 fine-tune 全连接层，前面的卷积层没有充分联合优化**

因此它还不是一个足够统一、足够端到端的检测框架。

---

## 3. Fast R-CNN

### 3.1 Fast R-CNN 解决什么问题
Fast R-CNN 主要是在 SPPnet 基础上，把训练和检测过程进一步统一化。

目标是：
- 保留共享卷积特征图的高效性
- 让整个检测网络更容易联合训练

---

### 3.2 核心结构
Fast R-CNN 的流程：

1. 整张图输入 CNN，得到共享 feature map
2. 外部方法（如 Selective Search）给出 proposals
3. 将 proposals 映射到 feature map
4. 对每个 proposal 做 **RoI Pooling**
5. 经全连接层后，输出两个分支：
   - softmax 分类
   - bbox regression

一句话概括：

> **shared feature map + RoI Pooling + softmax + bbox regression**

---

### 3.3 RoI Pooling 是什么
RoI Pooling 的作用是：

> **把任意大小的 RoI 变成固定大小的特征图**

比如输出固定为 `7×7`。

不管输入 RoI 是 `13×17`、`15×18` 还是其他尺寸，最终都变成：

$$
7 \times 7 \times C
$$

供全连接层使用。

---

### 3.4 RoI Pooling 和 SPP 的关系
本质上，RoI Pooling 可以看作 SPP 的简化版本。

- **SPP**：多层池化，如 `1×1 + 2×2 + 4×4`
- **RoI Pooling**：通常只做单层固定网格，如 `7×7`

所以更准确地说：

> **RoI Pooling 是 SPP 思想在目标检测中的简化和标准化版本**

---

### 3.5 Fast R-CNN 相比前代的优势
#### 相比 R-CNN
- 不再对每个 proposal 单独卷积
- 卷积特征共享，速度大幅提升
- 不需要单独训练 SVM

#### 相比 SPPnet
- 训练框架更完整
- 更容易联合优化卷积层
- 分类和回归在一个统一网络中完成

---

### 3.6 训练方式
Fast R-CNN 采用统一的多任务损失：

$L = L_{cls} + \lambda [u \ge 1] L_{loc}$

其中：
- $(L_{cls})$：分类损失
- $(L_{loc})$：边框回归损失
- 只有前景样本才计算定位损失

这意味着：
- 分类和回归一起训练
- 卷积层也可以联合 fine-tune

---

### 3.7 局限
Fast R-CNN 仍有一个明显瓶颈：

> **proposal 还是靠 Selective Search 等外部算法生成**

所以虽然检测头快了，但整个系统还不够快。

---

## 4. Faster R-CNN

### 4.1 Faster R-CNN 的核心贡献
Faster R-CNN 的关键突破是：

> **把候选框生成这一步也神经网络化**

也就是引入了 **RPN（Region Proposal Network）**。

因此可以把 Faster R-CNN 理解为：

> **Fast R-CNN + RPN**

---

### 4.2 总体结构
完整流程：

1. 输入整张图
2. backbone 提取共享 feature map
3. **RPN** 在 feature map 上生成 proposals
4. proposals 经过 RoI Pooling
5. detection head 输出类别与边框回归结果
6. 经过 NMS 得到最终检测框

---

### 4.3 RPN 是什么
RPN 的任务是：

> **给定 feature map，快速生成高质量候选框**

它不是最终分类器，它只做：
- 是否有物体（objectness）
- 边框回归

---

### 4.4 Anchor 机制
在 feature map 的每个位置，RPN 会放置若干个预设参考框，叫 **anchors**。

常见配置：
- 3 种尺度
- 3 种宽高比

那么每个位置就有 9 个 anchors。

每个 anchor 预测：
1. objectness score（前景/背景）
2. bbox regression（如何修正 anchor）

---

### 4.5 RPN 如何生成 proposals
RPN 会产生大量 anchor 对应的预测框，然后进行筛选：

1. 应用 bbox regression
2. 裁剪到图像边界内
3. 去掉过小框
4. 按 objectness 排序
5. 做 NMS
6. 保留 top-N proposals

这些 proposal 再送给 Fast R-CNN 检测头。

---

### 4.6 正负样本定义
#### RPN 的 anchor 标注
- 正样本：IoU 高于阈值（如 0.7），或某 GT 对应 IoU 最高的 anchor
- 负样本：IoU 低于阈值（如 0.3）

#### Detector 的 RoI 标注
- 前景：通常 IoU > 0.5
- 否则背景

---

### 4.7 Faster R-CNN 的意义
- proposal generation 可学习化
- proposal 与 detection 共用卷积特征
- 检测速度和精度都进一步提升
- 奠定了现代 two-stage 检测器的经典范式

---

## 5. FPN（Feature Pyramid Network）

### 5.1 为什么需要 FPN
目标检测里一个核心难点是：

> **目标尺度差异很大**

CNN 本身不同层的特征具有不同性质：

- 浅层：分辨率高、细节丰富、语义弱
- 深层：分辨率低、语义强、定位细节差

如果只在顶层特征图上做检测：
- 大目标问题不大
- 小目标往往不友好

---

### 5.2 FPN 的核心思想
FPN 要解决的问题是：

> **把高层的强语义传递给低层的高分辨率特征**

从而构建一组：
- 多尺度
- 都具备较强语义的特征图

---

### 5.3 基本结构
假设 backbone 输出特征为：

- `C2`
- `C3`
- `C4`
- `C5`

FPN 构造新的金字塔特征：

- `P2`
- `P3`
- `P4`
- `P5`

其中 `P` 层才是真正用于检测的多尺度特征。

---

### 5.4 两个关键结构
#### 1）Top-down pathway（自上而下路径）
从高层特征开始，上采样逐层往下传递语义信息。

#### 2）Lateral connections（横向连接）
把对应 backbone 层的特征接入，与上采样特征融合。

典型形式：

$P_l = \text{Upsample}(P_{l+1}) + \text{Conv1x1}(C_l)$

再接一个 `3×3 conv` 做平滑。

---

### 5.5 为什么用 1×1 conv
- 统一各层通道数
- 便于不同层之间相加
- 做语义对齐和压缩

---

### 5.6 为什么用 3×3 conv
- 融合局部上下文
- 缓解上采样带来的混叠效应
- 提升特征质量

---

### 5.7 FPN 的本质收益
FPN 的关键价值在于：

> **让低层特征也具有较强语义信息**

这样：
- 小目标可以在高分辨率层检测
- 大目标可以在低分辨率层检测
- 每层都“既看得清，又懂语义”

---

### 5.8 FPN 在检测中的使用
#### 在 Faster R-CNN 中
- RPN 可在 `P2~P5` 各层上生成 proposals
- proposal 根据大小分配到合适层做 RoI Pooling

#### 在 one-stage detector 中
也常用于多层特征图上直接做分类和回归，如 RetinaNet

---

### 5.9 FPN 的历史意义
FPN 把传统“图像金字塔”的思想，变成了网络内部的“特征金字塔”。

优点：
- 高效
- 通用
- 对多尺度尤其是小目标检测帮助显著

它已经成为现代检测器的重要基础模块。

---

## 6. 演化主线总结

### 6.1 方法演化关系
#### R-CNN
- Selective Search 提 proposals
- 每个 proposal 单独卷积
- SVM 分类
- 很慢

#### SPPnet
- 整图共享卷积
- proposal 在 feature map 上做 SPP
- 速度大幅提升
- 但训练仍不够统一

#### Fast R-CNN
- shared feature map
- RoI Pooling
- softmax + bbox regression
- 联合训练
- 仍依赖外部 proposal

#### Faster R-CNN
- 引入 RPN
- proposal 网络化
- proposal 与 detector 共享特征
- two-stage 框架成熟

#### FPN
- 为检测器提供强大的多尺度特征表示
- 特别改善小目标检测
- 成为 many latest detectors 的基础模块

---

### 6.2 一句话记忆
- **R-CNN**：先找框，再单独卷积，再分类
- **SPPnet**：整图卷积一次，RoI 上做金字塔池化
- **Fast R-CNN**：共享卷积 + RoI Pooling + 联合训练
- **Faster R-CNN**：Fast R-CNN + RPN
- **FPN**：让不同尺度的特征图都拥有强语义

---

## 7. 重点速记

### 7.1 RoI Pooling vs SPP
- **SPP**：多层池化（`1×1`、`2×2`、`4×4`）
- **RoI Pooling**：通常单层固定网格（如 `7×7`）
- 本质：RoI Pooling 是 SPP 的简化版

### 7.2 Faster R-CNN 的本质
- RPN 负责找“哪里可能有物体”
- detector 负责判断“这到底是什么，并精修边框”

### 7.3 FPN 的本质
- 深层：语义强
- 浅层：分辨率高
- FPN：把高层语义传给低层，让所有层都更适合检测

---

## 8. 最终总结

整条演化路线，本质上在不断解决三个核心问题：

1. **如何减少重复计算**
   - R-CNN → SPPnet / Fast R-CNN

2. **如何把候选框生成也纳入网络**
   - Fast R-CNN → Faster R-CNN

3. **如何更好处理多尺度目标**
   - Faster R-CNN → FPN

可以用一句话串起来：

> **目标检测的发展，就是从逐框计算，走向共享特征；从外部 proposal，走向可学习 proposal；从单层特征，走向多尺度强语义特征。**
