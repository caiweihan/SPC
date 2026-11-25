# AIDetection

以下命令均从项目根目录 `AIDetection/` 执行。命令行中的数据路径和输出路径可以相对于根目录填写。

## 创建环境

项目环境包含特征提取、训练、评估和绘图的公共依赖，使用 Python 3.10 和 CUDA 11.8 版 PyTorch，适用于 Linux x86_64。[environment.yml](environment.yml)

```bash
conda env create --file environment.yml
conda activate aidetection
```

## 准备编码器

从原始图片或文本提取特征时，请根据所选编码器的上游说明安装其依赖。编码器默认从 `methods/<方法>/` 读取源码和模型文件。资源存放在项目外时，可设置 `AIDETECTION_METHODS_DIR` 指向包含各方法子目录的根目录。

| 方法子目录 | 上游源码版本或包依赖 | 权重和 tokenizer 路径，相对于方法子目录 |
|---|---|---|
| PE | [perception_models](https://github.com/facebookresearch/perception_models) @ `3e352cca` | `checkpoints/PE-Core-G14-448/PE-Core-G14-448.pt` |
| MoEViE | [moe_vie](https://github.com/facebookresearch/moe_vie) @ `7c34cb2f` + 补丁 | `checkpoints/MoEViE-H14-448/MoEViE-H14-448.pt` |
| OpenCLIP | `open_clip_torch==3.3.0` | `checkpoints/CLIP-ViT-bigG-14-laion2B-39B-b160k/` |
| MetaCLIP | [metaclip](https://github.com/facebookresearch/metaclip) @ `f47f7841` | `checkpoints/metaclip2_bigG14_378px_worldwide.pt`、`checkpoints/xlm-v-base/` |
| SigLIP2 | `transformers==4.51.3` | `checkpoints/siglip2-giant-opt-patch16-384/` |
| DINOv3 | [dinov3](https://github.com/facebookresearch/dinov3) @ `6e50ab28` + 补丁 | `checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth` |

将上游源码检出到对应的方法子目录。对新检出的 MoEViE 和 DINOv3 应用项目补丁。MoEViE 补丁包含 V100 专家计算适配和严格权重检查。DINOv3 补丁包含本实验用的 ViT-7B 精度、图像扰动和编译器配置。

```bash
resources="${AIDETECTION_METHODS_DIR:-$PWD/methods}"
git -C "$resources/MoEViE" apply "$PWD/encoders/patches/moevie.patch"
git -C "$resources/DINOv3" apply "$PWD/encoders/patches/dinov3.patch"
```

加载资源后，可以用一张图片检查编码器。将 `--image` 指向本地图片，成功时程序输出 `INFERENCE_OK`。

```bash
CUDA_VISIBLE_DEVICES=0 python -m encoders.pe --image datasets/cat.jpeg
```

## 提取特征

图像提取接收按类别组织的 ImageFolder 数据，输出按类别存储的 H5。除 DINOv3 外，其余方法均可提取 SPC 的文本嵌入，输出包含 `e_f` 和 `e_r` 的 NPZ。命令和输出约定见对应脚本开头。[图像提取](scripts/extract_features.py) · [文本提取](scripts/extract_text_embeddings.py)

## 训练与测试

例如，使用 MoEViE 的图像特征训练线性分类器，并从已保存的检查点重新生成测试报告：

```bash
python scripts/train.py --method MoEViE --train_name SD14-train.h5 --enable_spc true --exp_name init --epochs 2 --lr 2e-5 --gpu 0
python scripts/test.py --method MoEViE --exp_name init
```

`--train_name` 填写 `outputs/<方法>/train-features/` 下的文件名。`--exp_name` 对应 `outputs/<方法>/results/` 下的实验目录。

## 合并特征

合并 H5 特征时，`--equal-samples` 从每个输入文件无放回抽取相同数量的样本，数量取各文件样本总数的最小值。每个文件的样本数按所有类别合计。`--seed` 可固定抽样结果。省略 `--equal-samples` 时合并全部样本。

```bash
python scripts/merge_h5.py a.h5 b.h5 --equal-samples --seed 42 -o merged.h5
```
