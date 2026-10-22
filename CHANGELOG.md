# 更新日志

## [1.2.0] - 2026-10-23

首个实际可用的发布。相对 1.0.0 基线，包含 8 个缺陷修复、模块化重构、架构模板与性能改进。

### 破坏性变更

- **模型检查器输入名改为英文**：`模型` → `model_name`、`显示细节` → `show_details`、
  `显示图表` → `show_charts`。界面工作流按控件顺序保存值，不受影响；
  以 JSON 直接调用 `/api/prompt` 的脚本需要改键名。
- **组件提取器产物目录**改为 `ComfyUI/models/model_toolkit_extracted/`（原 `output/`）。
- **组件替换器的替换源**改为上述提取目录；原先指向 `components/` 模板，实际从未可用。

### 修复

- CLIP 文本编码器权重被误判进 VAE（SD1.5 实测 197 个 CLIP 键中 192 个错分）
- 组件替换器永远返回「无法加载替换组件」
- UNET 检查器的块数量恒为 1（`input 12 / middle 3 / output 12` 现正常）
- 模型元信息永远显示「未包含」（safetensors 头部根本没被读）
- 旧版 ComfyUI 导入即崩（版本兼容检查写在 API 调用之后）
- `generate_text_report` 的潜伏 `NameError`
- 替换时不校验 `component` 归属与张量形状
- 同名模型内容变化后仍返回旧报告（新增 `IS_CHANGED` 文件指纹）

### 新增

- 架构模板 10 份（YAML，来自 sd-mecha，MIT），并编译为运行时索引
- 架构家族识别与能力矩阵：Flux / SD3 / AuraFlow / PixArt / HunyuanDiT /
  Stable Cascade / SVD / LTXV / MiniMax H3
- 分析只读 safetensors 头部，不再加载权重
- `tools/validate_templates.py`（真实模型覆盖率校验）、`tools/build_template_index.py`

### 重构与性能

- `nodes/checkpoint_inspector.py` 1128 行 → 5 行，拆分为 `model_toolkit/` 下 7 个模块
- 图表直取 Agg 画布，去掉 PNG/base64 往返：高级模式 1619 ms → 894 ms（-45%）
- 移除 EMA 模板死循环：基础分析 299.5 ms → 31.3 ms（-90%）
- 分析改读文件头：5.5 ms vs 全量加载 139.6 ms（25 倍）
- 移除进程级副作用：不再改全局 matplotlib 样式与 root logger 配置

### 工程

- 79 个 unittest + 4 个黄金样本场景（文本与图像哈希）
- 分支保护：main 需 PR + 线性历史；dev 允许自助合并
- 许可由 MIT 变更为 GPL-3.0（`components/architectures/` 的模板仍为 MIT）
- `requirements.txt` 不再声明 torch（避免重装/降级宿主环境）

## [1.0.0] - 2026-10-22

历史重排后的基线版本：仓库结构、组件数据、四个节点、文档与 1.0.0 版本号。
