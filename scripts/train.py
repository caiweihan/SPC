"""Train the two-output linear probe and report epoch 0 and each training epoch.

Run the following commands from the project root.
For a saved run, use python scripts/test.py --method METHOD --exp_name NAME to regenerate the report.

python scripts/train.py --method PE --enable_spc true --train_name SD14-train.h5 --exp_name init-test10 --epochs 1 --lr 5e-4

python scripts/train.py --method PE     --enable_spc true  --train_name SD14-train-fake-jpeg96.h5 --exp_name init   --epochs 2 --lr 1e-3 --gpu 0
python scripts/train.py --method PE     --enable_spc false --train_name SD14-train.h5 --exp_name noInit --epochs 2 --lr 1e-3 --gpu 1
python scripts/train.py --method MoEViE --enable_spc true  --train_name SD14-train-fake-jpeg96.h5 --exp_name init   --epochs 2 --lr 1e-3 --gpu 2
python scripts/train.py --method MoEViE --enable_spc false --train_name SD14-train.h5 --exp_name noInit --epochs 2 --lr 1e-3 --gpu 3
python scripts/train.py --method DINOv3 --enable_spc false --train_name SD14-train.h5 --exp_name noInit --epochs 2 --lr 1e-3 --gpu 0

python scripts/train.py --method PE     --enable_spc true  --train_name SD14-train.h5 --exp_name noTrain-similar   --epochs 0 --lr 2e-5 --gpu 0 --text_emb_path outputs/PE/prompt_ablation/similar.npz
python scripts/train.py --method MoEViE --enable_spc true  --train_name SD14-train.h5 --exp_name noTrain-similar   --epochs 0 --lr 2e-5 --gpu 1 --text_emb_path outputs/MoEViE/prompt_ablation/similar.npz

python scripts/train.py --method PE     --enable_spc true  --train_name SD14-train.h5 --exp_name init   --epochs 2 --lr 2e-5 --gpu 0
python scripts/train.py --method PE     --enable_spc false --train_name SD14-train.h5 --exp_name noInit --epochs 2 --lr 1e-3 --gpu 1
python scripts/train.py --method MoEViE --enable_spc true  --train_name SD14-train.h5 --exp_name init   --epochs 2 --lr 2e-5 --gpu 2
python scripts/train.py --method MoEViE --enable_spc false --train_name SD14-train.h5 --exp_name noInit --epochs 2 --lr 1e-3 --gpu 3
python scripts/train.py --method DINOv3 --enable_spc false --train_name SD14-train.h5 --exp_name noInit --epochs 2 --lr 1e-3 --gpu 0

python scripts/train.py --method OpenCLIP  --enable_spc true  --train_name SD14-train.h5 --exp_name init   --epochs 2 --lr 2e-5 --gpu 0
python scripts/train.py --method OpenCLIP  --enable_spc false --train_name SD14-train.h5 --exp_name noInit --epochs 2 --lr 1e-3 --gpu 1
python scripts/train.py --method SigLIP2   --enable_spc true  --train_name SD14-train.h5 --exp_name init   --epochs 2 --lr 2e-5 --gpu 2
python scripts/train.py --method SigLIP2   --enable_spc false --train_name SD14-train.h5 --exp_name noInit --epochs 2 --lr 1e-3 --gpu 3
python scripts/train.py --method MetaCLIP2 --enable_spc true  --train_name SD14-train.h5 --exp_name init   --epochs 2 --lr 2e-5 --gpu 2
python scripts/train.py --method MetaCLIP2 --enable_spc false --train_name SD14-train.h5 --exp_name noInit --epochs 2 --lr 1e-3 --gpu 3

python scripts/train.py --method MoEViE     --enable_spc true  --train_name SD14-train.h5 --exp_name prompt-similar --text_emb_path outputs/MoEViE/prompt_ablation/similar.npz --epochs 2 --lr 2e-5 --gpu 0
python scripts/train.py --method MoEViE     --enable_spc true  --train_name SD14-train.h5 --exp_name prompt-opposite --text_emb_path outputs/MoEViE/prompt_ablation/opposite.npz --epochs 2 --lr 2e-5 --gpu 1
python scripts/train.py --method MoEViE     --enable_spc true  --train_name SD14-train.h5 --exp_name prompt-unrelated_i --text_emb_path outputs/MoEViE/prompt_ablation/unrelated_i.npz --epochs 2 --lr 2e-5 --gpu 2
python scripts/train.py --method MoEViE     --enable_spc true  --train_name SD14-train.h5 --exp_name prompt-unrelated_ii --text_emb_path outputs/MoEViE/prompt_ablation/unrelated_ii.npz --epochs 2 --lr 2e-5 --gpu 3

python scripts/train.py --method MoEViE     --enable_spc true  --train_name SD14-train.h5 --exp_name init-lr5e-6   --epochs 2 --lr 5e-6 --gpu 0
python scripts/train.py --method MoEViE     --enable_spc true  --train_name SD14-train.h5 --exp_name init-lr2e-5   --epochs 2 --lr 2e-5 --gpu 1
python scripts/train.py --method MoEViE     --enable_spc true  --train_name SD14-train.h5 --exp_name init-lr8e-5   --epochs 2 --lr 8e-5 --gpu 2
python scripts/train.py --method MoEViE     --enable_spc true  --train_name SD14-train.h5 --exp_name init-lr3e-4   --epochs 2 --lr 3e-4 --gpu 3
python scripts/train.py --method MoEViE     --enable_spc true  --train_name SD14-train.h5 --exp_name init-lr1e-3   --epochs 2 --lr 1e-3 --gpu 0
python scripts/train.py --method MoEViE     --enable_spc true  --train_name SD14-train.h5 --exp_name init-lr4e-3   --epochs 2 --lr 4e-3 --gpu 1
"""

import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import test as evaluation

SEED = 42
BATCH_SIZE = 128


def parse_bool(value):
    if value.lower() not in ("true", "false"):
        raise argparse.ArgumentTypeError("请输入 true 或 false")
    return value.lower() == "true"


def create_parser():
    parser = argparse.ArgumentParser(description="Train a linear classifier on features, optionally using SPC text initialization.")
    parser.add_argument("--method", default="MoEViE", help="特征提取方法，对应 outputs 下的目录")
    parser.add_argument("--enable_spc", type=parse_bool, metavar="{true,false}", default=True, help="是否启用 SPC")
    parser.add_argument("--train_name", default="OpenSDI-train.h5", help="train-features 下的训练特征文件名")
    parser.add_argument("--exp_name", default="test", help="results 下的实验目录名")
    parser.add_argument("--text_emb_path", default=None, help="指定 SPC 文本特征 NPZ；默认使用方法目录中的 text_embeddings.npz")
    parser.add_argument("--gpu", type=int, default=None, help="指定物理 GPU 编号，例如 --gpu 2；未指定时沿用 CUDA_VISIBLE_DEVICES")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--save_pdf", action="store_true", help="在 PNG 之外同时保存 PDF 版本的训练汇总图")
    return parser


def load_text_embeddings(path, feat_dim):
    """Load the fake/real text vectors; bias is initialized separately."""
    with np.load(path, allow_pickle=False) as data:
        e_f = data["e_f"].astype(np.float32)
        e_r = data["e_r"].astype(np.float32)
    if e_f.shape[0] != feat_dim or e_r.shape[0] != feat_dim:
        raise ValueError(
            f"{path} 中嵌入维度 ({e_f.shape[0]}) 与特征维度 {feat_dim} 不匹配"
        )
    print(f"Text emb    : loaded from {path}")
    return e_f, e_r


def build_train_dataset(fake, real):
    x = torch.cat([torch.from_numpy(fake), torch.from_numpy(real)], dim=0)
    y = torch.cat([torch.zeros(len(fake), dtype=torch.long), torch.ones(len(real), dtype=torch.long)], dim=0)
    return TensorDataset(x, y)


@torch.no_grad()
def build_classifier(args, feat_dim, device, text_path):
    model = evaluation.build_head("linear2", feat_dim).to(device)
    if args.enable_spc:
        if args.init_mode != "random":
            e_f, e_r = load_text_embeddings(text_path, feat_dim)
            model.weight.copy_(torch.from_numpy(np.stack([e_f, e_r])))
        model.bias.copy_(model.bias.new_tensor([0., 1.]))
    else:
        model.bias.zero_()
    return model


def train_epoch(model, loader, criterion, optimizer, device, single_logit):
    model.train()
    for xb, yb in loader:
        logits = model(xb.to(device, non_blocking=True))
        targets = yb.to(device, non_blocking=True)
        if single_logit:
            # Dataset labels are fake=0, real=1. The single logit is positive for fake.
            loss = criterion(logits[:, 0], 1.0 - targets.float())
        else:
            loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def run_training(args, build_model, *, extra_row=None):
    if args.epochs < 0:
        raise ValueError("--epochs 必须是非负整数")
    device = evaluation.select_device(args.gpu)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    directory = evaluation.OUTPUTS / args.method
    print(f"Head        : {args.head}")
    print(f"SPC         : {'enabled' if args.enable_spc else 'disabled'}")
    print(f"Init        : {args.init_mode if args.enable_spc else 'random'}")
    print(f"Threshold   : {args.decision_threshold} (signed distance)")
    fake, real = evaluation.load_h5_group(directory / "train-features" / args.train_name, args.enable_spc)
    loader = DataLoader(build_train_dataset(fake, real), batch_size=BATCH_SIZE, shuffle=True)
    feat_dim = fake.shape[1]
    model = build_model(args, feat_dim, device, args.text_emb_path or directory / "text_embeddings.npz")
    single_logit = args.head == "linear1"
    criterion = nn.BCEWithLogitsLoss() if single_logit else nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    groups = evaluation.collect_test_groups(directory / "test-features")

    history = {}
    for epoch in range(args.epochs + 1):
        if epoch > 0:
            train_epoch(model, loader, criterion, optimizer, device, single_logit)
        print(f"\n===== Evaluate at epoch {epoch} =====")
        scores, markdown, distances = evaluation.evaluate(
            model, groups, device, args.method, args.enable_spc, args.decision_threshold,
            collect_distances=(epoch == args.epochs))
        for group, values in scores.items():
            series = history.setdefault(group, {})
            for name, accuracy in values.items():
                series.setdefault(name, []).append(accuracy)

    result_dir = directory / "results" / args.exp_name
    result_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": vars(args),
                "feat_dim": feat_dim, "history": history}, result_dir / "probe.pt")
    evaluation.write_report(result_dir, markdown, history, distances, args.method,
                            args.decision_threshold, args.save_pdf, extra_row=extra_row)


def main():
    parser = create_parser()
    parser.set_defaults(head="linear2", decision_threshold=0.0)
    parser.add_argument("--init_mode", choices=("text", "random"), default="text",
                        help="SPC 权重初始化：text 使用文本嵌入，random 使用随机权重")
    run_training(parser.parse_args(), build_classifier)


if __name__ == "__main__":
    main()
