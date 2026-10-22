# 架构模板（Architecture Templates）

## 来源

本目录的 `*.yaml` 来自开源项目 **sd-mecha 1.1.8 / 1.1.7** 的内置 `model_configs`
（仓库：`ljleb/sd-mecha`，经 ComfyUI 插件 `comfy-mecha` 分发），许可为 **MIT**，
全文见同目录 `LICENSE-sd-mecha`（Copyright (c) 2024 ljleb）。

上游 `model_configs/` 共 11 个 YAML，其中 `singleton-mecha.yaml` 是 4 行内部标记
（`keys: 1`），不是架构定义，未纳入。**其余 10 份全部收录**，文件名与上游一致，便于追溯。

## 10 份模板

| 文件 | 架构 | 组件（键数） | 参与自动识别 | 本机验证 |
|---|---|---|---:|---|
| `sd1-ldm.yaml` | SD1.x（ComfyUI 单文件） | CLIP 197 / VAE 248 / UNET 686 / EMA 688* | ✅ | ✅ 100% |
| `sd1-kohya.yaml` | SD1.x（kohya 训练格式：`unet.*`/`te.*`） | CLIP 196 / VAE 248 / UNET 686 | ✅ | — |
| `sdxl-sgm.yaml` | SDXL（ComfyUI 单文件） | CLIP 587 / VAE 248 / UNET 1684 | ✅ | ✅ 100% |
| `sdxl-kohya.yaml` | SDXL（kohya：`te1.*`/`te2.*`） | CLIP 714 / VAE 248 / UNET 1680 | ✅ | — |
| `sdxl-kohya_but_diffusers.yaml` | SDXL（kohya + diffusers UNet） | CLIP 714 / VAE 248 / UNET 1680 | ✅ | — |
| `sdxl-diffusers_unet_only.yaml` | SDXL（仅 UNet） | UNET 1680 | ✅ | — |
| `flux-flux.yaml` | Flux | UNET 780 / VAE 244 / CLIP 196 / T5XXL 219 | ❌ | — |
| `sd3-comfyui.yaml` | SD3 | UNET 492 / VAE 244 / CLIP 716 / T5XXL 220 | ❌ | — |
| `sd1-supermerger_blocks.yaml` | SD1.x 块定义 | BLOCKS 26 | ❌ | — |
| `sdxl-supermerger_blocks.yaml` | SDXL 块定义 | BLOCKS 21 | ❌ | — |

\* `EMA` 段数据损坏（见下），不计入覆盖率。
「参与自动识别」= 模板只含 `UNET/VAE/CLIP/EMA` 这些本项目节点能表达的组件。
Flux/SD3 的 T5XXL、supermerger 的 BLOCKS 无法映射，因此**数据保留但不参与识别**——
等节点支持 T5 或新增组件类型后，去掉 `REPRESENTABLE_COMPONENTS` 的限制即可启用。

## 运行时用编译索引，不直接解析 YAML

纯 Python 解析这 10 份 YAML 需要 **数秒**（`yaml.SafeLoader` 约 14 s，`CSafeLoader` 约 1.8 s），
而 `index.json` 解析只需 **约 170 ms**（首次加载，进程内缓存）。

- `index.json` 是**生成物**，由 `tools/build_template_index.py` 从 YAML 编译而来；
- 改了任何 YAML 都必须重新生成，否则 `--check` 会报不同步：

```bash
python tools/build_template_index.py           # 生成/更新
python tools/build_template_index.py --check   # 校验是否同步
```

索引里每个键存 `[shape, dtype, optional]`，dtype 是上游字符串（`float32`），
运行时由 `templates.DTYPE_ALIASES` 转成 `torch.dtype`。

## 组件名适配

| 上游组件 | 本项目 | 说明 |
|---|---|---|
| `diffuser` | UNET | |
| `clip_l` / `clip_g` | CLIP | 本项目只区分 CLIP，SDXL 双编码器合并统计 |
| `vae` | VAE | |
| `ema` | EMA | 只保留数据，**不参与匹配**（见下） |
| `scheduler` | — | 忽略（模型文件通常不保存） |

## 已知缺陷（重要）

上游 `ema` 段的键名**丢了所有点**，例如 `model_ema.diffusion_modelinput_blocks00bias`。
对本机 SD1 模型 1133 键中仅 `model_ema.decay` / `model_ema.num_updates` 2 个命中。
因此本项目 **不把 ema 段用于匹配**，EMA 仍由 `model_ema.` 前缀规则识别。
升级上游数据前不要改动这一约定。

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

## 覆盖边界（为什么没有"市面全部架构"）

- sd-mecha 最新版 1.1.8 的 `model_configs` 仍是这 10 份，没有更多架构可拿；
- ComfyUI 0.42.4 支持约 50 个架构（Flux/Flux2/Flux3、SD3、HunyuanDiT/Video、PixArt、
  Lumina2、Cascade、SVD、LTXV、Cosmos、ZImage 等），但其许可为 **GPL-3.0**，
  检测表不能搬进本项目；其检测函数还依赖 CUDA（CPU 环境下直接断言失败），
  无法在插件里稳定调用；
- 因此本项目只在**有模板**的架构上做逐键组件判定，其余架构仍走前缀规则。
