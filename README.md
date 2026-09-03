# 多拓本时序融合 — Rubbings as a time series

同一通碑的历代拓本，不是冗余，而是**同一潜在浮雕场在不同时刻的多次观测**。
本项目把它做成一个反问题，并回答：**这串观测到底能测出什么、测不出什么。**

**论文草稿**：[`paper/MANUSCRIPT.md`](paper/MANUSCRIPT.md) ｜ **Day 0 结论**：[`FINDINGS-day0.md`](FINDINGS-day0.md)
｜ **研究设计**：[`lit/design.md`](lit/design.md)

## 环境

```bash
CONDA=/home/sam/miniconda3/envs/heritage-opt/bin/python   # torch 2.13+cu130, RTX 4080 SUPER
$CONDA -m pytest tests -q          # 16 项正确性测试
```

不需要新建环境；`heritage-opt` 就是 idea-01（铅丹反演）用的那个。
字体用 `/usr/share/fonts/truetype/arphic/ukai.ttc`（繁体楷书，对《九成宫》全文 0 缺字）。

## 流水线

```bash
$CONDA src/npm_harvest.py                      # 台北故宫目录 → 8414 条
$CONDA src/inventory.py                        # 归并 → 410 通碑有 ≥2 拓本
$CONDA src/npm_detail.py 16059 20595 24521 27587 24592     # 详情页 + 釋文 + IIIF
$CONDA src/npm_images.py                       # IIIF 全分辨率下载（CC BY 4.0）
$CONDA src/scale.py                            # 逐本物理尺度 → 是不是同一块石头？
$CONDA src/build_stacks.py data/interim/stacks3.npz 20595 24592 27587
$CONDA src/exp_identify.py                     # 可辨识性（主结果）
$CONDA src/exp_real.py                         # 《九成宫》真实反演
$CONDA src/exp_rate.py --seeds 3               # 干扰项消融 + 拓本数消融
$CONDA src/exp_late.py                         # 最早存世拓本越晚，融合越值钱
$CONDA src/figures.py                          # 图 1/3/4
$CONDA src/fig_mechanism.py                    # 图 2
$CONDA src/fig_results.py                      # 图 5/6
$CONDA src/fig_real.py                         # 图 7
$CONDA src/fig_ablation.py                     # 图 8
$CONDA src/figcheck.py                         # 版面几何检测（越界/重叠/压数据）
```

**版面检测**：`src/figcheck.py` 渲染每张图后读取所有文字对象的 bounding box，
程序化判定四类缺陷——超出画布、文字互相重叠、文字压在曲线/柱子/误差棒上、
标注侵入相邻面板。参考线（`axhline`/`axvspan`）与视野外的刻度已排除，不算误报。
当前 **8 张图 0 问题**。

## 代码

| 文件 | 作用 |
|---|---|
| `src/physics.py` | 拓制前向模型：纸张架桥（可分离二次结构元开运算）+ 阈值上墨 + 采集噪声 + 逐本纸损 |
| `src/weather.py` | 单调风化：棱线圆化 / 深度衰减 / 嵌套石花；字形 → V 形刻槽浮雕 |
| `src/torchmodel.py` | 可微 GPU 版（与 numpy 一致到 1e-5）；min-plus 精确欧氏距离变换 |
| `src/fuse.py` | 多字联合反演：共享手法 / 共享单调风化 / 速率律 / Cauchy 数据项 |
| `src/synth.py` | 合成基准（真值已知） |
| `src/segment.py` | 剪裱本 → 单字（2-D 点阵拟合，取基频而非倍频） |
| `src/corpus.py` | 跨拓本字序对齐（位移容忍 NCC + 带状 DTW，**无需 OCR**） |
| `src/shiwen.py` | 釋文 → 逐字损坏矩阵 + 单调性检验 |
| `src/scale.py` | 逐本字距（mm）→ 同一物体判定 |
| `src/evaluate.py` | 基线（阈值 / 堆叠 / RL 反卷积）与指标（IoU / 识别率） |

## 核心结论

| 结论 | 数字 | 出处 |
|---|---|---|
| 博物馆释文本身即逐字损坏标签，单调一致性 | **99.3 %**（35/5353 违反） | `shiwen.py` |
| 五本损字率沿年代单调上升 | 0.2 → 1.9 → 2.1 → 2.2 → 4.4 % | 同上 |
| 三本真拓字距一致 | 32.9 / 33.5 / 32.2 mm（±2 %） | `scale.py` |
| **24521 是缩刻本，不是原碑拓本** | 17.2 mm（约一半） | `scale.py` + 释文异常 |
| 跨拓本字序对齐（无 OCR、无释文） | 448 字 × 4 本；416 字 × 3 本 | `corpus.py` |
| 拓片单字锚定到释文位置 | 787 格 | `anchor_text.py` |
| **外部验证：博物馆标"读不出"的字，石花面积** | **2.0–3.2 倍**，p = 3e-6 … 1e-4（315 字） | `validate_external.py` |
| 风化速率可辨识区间（合成基准） | 残差 1 % 内 a ∈ [0.035, 0.075] mm/世纪 | `exp_identify.py` |
| **拓本数门槛** | 2–3 张时 a 低估一半；**≥4 张**误差 ≤13 % | `exp_rate.py` |
| 采集噪声反而有利 | clean 0.048 → 加纸纹+污渍 0.073（真值 0.075） | `exp_rate.py` |
| 融合复原 **不优于**阈值化最早拓本 | 四种情形全部持平或略低 | `exp_late.py` |
| 《九成宫》实测棱线圆化率 | **0.024 mm/世纪**（10 % 区间 0.004–0.036） | `exp_real_profile.py` |
| 同期两本宋拓被判为不同手法 | λ = 0.41 vs 1.43 mm | `exp_real.py` |
| 未断代拓本定年 | 中位误差 **146 年**（9 次留一） | `exp_dating.py` |
| 前向模型 numpy↔torch 一致 | < 1e-5；16 项测试全过 | `tests/` |

## 数据与授权

全部图像与元数据来自**台北故宫典藏资料检索**（`digitalarchive.npm.gov.tw`），
600 万画素图档 **CC BY 4.0**，低阶图 CC0，IIIF Image API v2，无需申请。
署名格式：`○○○○○ 國立故宮博物院，臺北，CC BY 4.0 @ www.npm.gov.tw`。
