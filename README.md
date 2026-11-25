# SPC · Semantic Prototype Calibration

一张图像来自相机还是生成模型？SPC 借用视觉语言模型对两句话的理解来训练检测器。先让文本编码器分别读入 `AI generated` 和 `a real photo`，把它们在模型中的表示作为分类器判断“生成”和“真实”的起点。再用带标签的图像调整分类器，让这两个起点更适合检测任务。整个过程中图像编码器保持冻结。预测新图像时，只需提取一次图像特征，再由训练好的线性分类器给出两个类别的分数。

本仓库实现了 Weihan Cai、Hao Tan、Zichang Tan、Jun Wan 和 Xinping Gao 的论文 [Unleashing the Potential of Vision-Language Models for Generalizable AI-Generated Image Detection](https://arxiv.org/abs/2608.04935)。

## 论文结果

检测器在 GenImage 的 Stable Diffusion v1.4 训练集上学习，再到其他来源的图像上测试。下表为论文报告的平均平衡准确率（%）：先分别计算真实图像和生成图像的识别率，再取两者的平均值。

| 冻结的图像编码器 | 训练方法 | In-the-wild | Controlled |
|---|---|---:|---:|
| PE | 随机初始化的线性分类器 | 84.8 | 95.2 |
| PE | SPC | **92.9** | **96.5** |
| MoEViE | 随机初始化的线性分类器 | 81.4 | 93.5 |
| MoEViE | SPC | **98.5** | **95.5** |

In-the-wild 对 Chameleon、SynthWildX、WildRF、SocialRF 和 CommunityAI 的九个测试子集等权平均。Controlled 对 GenImage、UnivFakeDetect、AIGI-Now 和 AIGI-Holmes 四个基准的成绩等权平均。[论文表 I 和表 II](https://arxiv.org/html/2608.04935v3#S4)

## 安装

从项目根目录运行以下命令。环境文件适用于 Linux x86_64，包含训练分类器和生成测试报告所需的依赖。若从原始图像开始，还需准备 CUDA GPU，并安装所选编码器的上游依赖。

```bash
conda env create -f environment.yml
conda activate spc
```

下面以 **PE-Core-G14-448** 为例。将 PE 源码放到 `methods/PE/`，把模型文件放到 `methods/PE/checkpoints/PE-Core-G14-448/PE-Core-G14-448.pt`。如果编码器已安装在其他位置，可将 `SPC_METHODS_DIR` 设为包含 `PE/` 等方法子目录的路径。[PE 源码版本](https://github.com/facebookresearch/perception_models/tree/3e352cca) · [PE 权重](https://huggingface.co/facebook/PE-Core-G14-448)

## 用自己的图片训练和评估

**1. 准备图像。** 以下示例用 GenImage SD v1.4 训练，用 Chameleon 测试。可以换成自己的数据，只需让训练集和测试集分别包含 `real/` 和 `fake/` 两个类别目录。

```text
datasets/
├── GenImage/sdv4/train/
│   ├── real/
│   └── fake/
└── Wild-Datasets/Chameleon/
    ├── real/
    └── fake/
```

**2. 用 PE 提取特征。** 前两条命令把每张图像编码一次，将结果按类别保存，供后续实验重复使用。第三条命令让同一个 PE 模型读入两个类别描述，保存分类器的初始权重。每个编码器只需为这对提示词运行一次文本提取。

```bash
python scripts/extract_features.py --method PE \
  --val-dataset datasets/GenImage/sdv4/train \
  --output-dir outputs/PE/train-features/SD14-train.h5

python scripts/extract_features.py --method PE \
  --val-dataset datasets/Wild-Datasets/Chameleon \
  --output-dir outputs/PE/test-features/In-the-wild/Chameleon.h5

python scripts/extract_text_embeddings.py --method PE \
  --fake-text "AI generated" --real-text "a real photo"
```

**3. 训练分类器。** 已准备好上述图像特征和文本初始权重时，可直接从这里开始。训练脚本使用 AdamW，批大小为 128。以下命令采用论文中 SPC 的训练轮数和学习率，并在训练前及每轮结束后评估。添加测试集时，把对应的特征文件放在 `outputs/PE/test-features/<基准组>/<子集>.h5` 下。只准备 Chameleon 时，报告仅覆盖这个子集。

```bash
python scripts/train.py --method PE --train_name SD14-train.h5 \
  --enable_spc true --exp_name pe-spc --epochs 2 --lr 2e-5 --gpu 0
```

要比较随机初始化的线性基线，复用同一批图像特征，运行：

```bash
python scripts/train.py --method PE --train_name SD14-train.h5 \
  --enable_spc false --exp_name pe-linear --epochs 2 --lr 1e-3 --gpu 0
```

**4. 查看结果。** `outputs/PE/results/pe-spc/` 下的 `results.md` 是各测试集的成绩，`training_summary.png` 展示各轮的评估曲线，以及真实图像和生成图像到分类边界的距离分布，`probe.pt` 保存训练好的分类器。要用保存的分类器重新评估并同时输出 PDF，运行：

```bash
python scripts/test.py --method PE --exp_name pe-spc --save_pdf
```

## 其他编码器

同一套流程也支持 MoEViE、OpenCLIP、SigLIP2 和 MetaCLIP2（命令行名称为 `MetaCLIP`）。切换模型时，为该模型重新提取图像与文本特征，并在所有命令中使用一致的方法名和输出目录。DINOv3 是论文的纯视觉基线，训练时使用 `--enable_spc false`。各适配器的权重路径和加载方式见 [encoders/](encoders/)。

## 引用

```bibtex
@article{cai2026spc,
  title   = {Unleashing the Potential of Vision-Language Models for Generalizable AI-Generated Image Detection},
  author  = {Cai, Weihan and Tan, Hao and Tan, Zichang and Wan, Jun and Gao, Xinping},
  journal = {arXiv preprint arXiv:2608.04935},
  year    = {2026}
}
```
