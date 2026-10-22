# 架构模板（Architecture Templates）

## 来源

本目录的 `*.yaml` 来自开源项目 **sd-mecha 1.1.7** 的内置 `model_configs`
（仓库：`ljleb/sd-mecha`，经 ComfyUI 插件 `comfy-mecha` 分发），许可为 **MIT**，
全文见同目录 `LICENSE-sd-mecha`（Copyright (c) 2024 ljleb）。

- `sd1-ldm.yaml`：SD1.x（LDM 键布局）
- `sdxl-sgm.yaml`：SDXL（SGM / ComfyUI 单文件键布局）

原样保留上游文件名，便于追溯与后续同步。

## 数据格式

```yaml
identifier: sd1-ldm
components:
  diffuser:                     # 本项目映射为 UNET
    model.diffusion_model.input_blocks.0.0.weight:
      {shape: [320, 4, 3, 3], dtype: float32, aliases: [...], optional: false}
```

`model_toolkit/templates.py` 负责三件事：

1. 组件名适配：`diffuser→UNET`、`clip_l→CLIP`、`clip_g→CLIP`（本项目只区分 CLIP，SDXL 双编码器合并）、`vae→VAE`；
2. dtype 适配：`float32` → `torch.float32`；
3. 覆盖率判定：模型键被模板覆盖 ≥ 95% 时认定为该架构，组件归属直接用模板成员关系判定，不再靠前缀猜。

`scheduler` 组件被忽略（模型文件通常不保存，本项目也不分析）。

## 已知缺陷（重要）

上游 `ema` 段的键名**丢了所有点**，例如：

```
model_ema.diffusion_modelinput_blocks00bias
```

对真实 checkpoint 几乎 0 命中（本机 SD1 模型 1133 键中仅 `model_ema.decay` /
`model_ema.num_updates` 2 个命中）。因此本项目 **不把 ema 段用于匹配**，
EMA 仍由 `model_ema.` 前缀规则识别。升级上游数据前不要改动这一约定。

## 校验

```bash
python tools/validate_templates.py <模型路径> [...]
```

只读 safetensors 头部，打印覆盖率与各组件命中数。本机实测：

| 模型 | 键数 | 判定 | 覆盖率 |
|---|---:|---|---:|
| SD1.5 fp16（Alpha_Anime_2.5D） | 1133 | `sd1-ldm` | 99.8% |
| SDXL（NoobXL_100） | 2514 | `sdxl-sgm` | 100.0% |

SD1 的 UNET/VAE/CLIP 分组为 686 / 248 / 197，与迁移前的前缀分类**逐键一致**；
SDXL 2514 键同样逐键一致（差异 0 条）。
