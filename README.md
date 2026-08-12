# Industrial Defect Inspector

[![CI](https://github.com/m0ndego/industrial-defect-inspector/actions/workflows/ci.yml/badge.svg)](https://github.com/m0ndego/industrial-defect-inspector/actions/workflows/ci.yml)

一个可复现的工业外观缺陷检测项目：自己实现卷积自编码器基线，并使用 Anomalib 2.6.0
复现 PatchCore，在 MVTec AD Bottle 上比较图像级检测和像素级定位效果。

## 目标与验收

- 输入：单张产品图片、图片目录或 MVTec AD Bottle。
- 输出：异常分数、阈值判断、JSON、热力图和叠加图。
- 指标：image AUROC、pixel AUROC、image F1、平均推理延迟。
- 目标：PatchCore image AUROC ≥ 0.95、pixel AUROC ≥ 0.90。

> 当前仓库先提供完整、可测试的工程代码。真实 MVTec 指标必须在下载官方数据并运行后填写，
> 不预先编造结果。

## 方法直觉

```text
正常训练图片 ──┬──> 卷积自编码器 ──> 重建误差 ──> 异常图
               └──> 预训练 ResNet-18 ──> 正常特征记忆库 ──> PatchCore 最近邻距离

测试图片 ───────────────────────────────────────────────> 分数 + 热力图
```

- **自编码器**学习重建正常瓶子。测试图中重建不好的区域被视为可疑区域。
- **PatchCore**保存正常图片的局部深度特征；距离所有正常特征都很远的局部块被视为异常。
- 部署阈值只由留出的正常校准集第 99 百分位确定，不使用测试标签。

## 环境

- Windows 10/11
- Python 3.12
- NVIDIA RTX 50 系列：PyTorch CUDA 13.0
- CPU 环境也能运行自编码器合成 smoke test，但完整 PatchCore 会更慢

```powershell
# GPU 开发环境
uv sync --extra cuda --extra dev

# 验证环境
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run ruff check .
uv run pytest
```

## 数据准备

从 [MVTec AD 官方页面](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) 获取并解压数据。
数据采用 CC BY-NC-SA 4.0，仅限非商业用途，不要提交到 GitHub。

```powershell
uv run python -m defect_inspector prepare-data --source D:\path\to\mvtec_anomaly_detection
```

程序会把正常训练图片固定划分为 80% 训练和 20% 校准，并复制/硬链接 Bottle 测试图片及掩码。
如果目标目录非空，程序会拒绝覆盖。

## 训练、评估和推理

```powershell
uv run python -m defect_inspector train --model autoencoder
uv run python -m defect_inspector evaluate --model autoencoder

uv run python -m defect_inspector train --model patchcore
uv run python -m defect_inspector evaluate --model patchcore

uv run python -m defect_inspector predict --model patchcore --input path\to\image.png
uv run python -m defect_inspector predict --model patchcore --input path\to\folder
```

生成内容位于 `artifacts/<model>/`，包括模型、元数据、`metrics.json`、预测 JSON 和结果图；
这些运行产物默认不会被 Git 跟踪。

完整的每日目标、亲手练习和复盘问题见 [`LEARNING_PLAN.md`](LEARNING_PLAN.md)。

## 结果

| 模型 | Image AUROC | Pixel AUROC | Image F1 | 平均延迟 |
| --- | ---: | ---: | ---: | ---: |
| 卷积自编码器 | 待实测 | 待实测 | 待实测 | 待实测 |
| PatchCore / ResNet-18 | 待实测 | 待实测 | 待实测 | 待实测 |

最终结果必须来自 `artifacts/*/metrics.json`。结果对比时保持数据划分、图像大小和随机种子一致。

## 工程保证

- 划分可复现且训练集、校准集无重叠。
- 缺少掩码、损坏图片、缺少模型和非空目标目录会明确报错。
- CUDA 不可用时回退 CPU；显存不足时不偷偷修改实验参数。
- CI 使用合成数据训练 1 个 epoch，不下载数据集或预训练权重。

## 局限

- 仅验证 Bottle，不能直接代表其他材质和产品。
- 自编码器可能把缺陷也重建出来，因此只是可解释基线。
- PatchCore 使用 ImageNet 预训练特征，不是端到端训练。
- MVTec AD 是研究基准且许可禁止商业使用，商业落地需要自有数据和重新校准阈值。

## 简历表述模板

> 基于 PyTorch 与 Anomalib 构建工业缺陷检测与像素级定位系统，实现卷积自编码器基线、
> PatchCore 复现、正常样本阈值校准、命令行批量推理、自动测试与 GitHub Actions；在
> MVTec AD Bottle 上取得 **填写实测指标**。

## 参考

- Roth et al., [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265)
- [Anomalib](https://github.com/open-edge-platform/anomalib), Apache-2.0
- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), CC BY-NC-SA 4.0
