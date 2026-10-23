# 更新日志

## [1.3.0] - 2026-10-24

### 破坏性变更

- **删除「🔬 UNET检查器」节点**：其 UNET 块结构分析已包含在模型检查器的
  「基础」报告中（且附带组件占比与图表），属于重复功能。旧工作流中的该
  节点将失效，请改用「🔍 模型检查器」。
- 模型检查器新增 `model_dir` 输入。以 JSON 直接调用 `/api/prompt` 的脚本
  不传该键时默认「（全部）」，沿用旧行为（在所有模型目录中查找），无需修改。

### 新增

- **模型检查器新增「模型目录」选择**：顶部新增 `model_dir` 下拉，列出
  `ComfyUI/models/` 下第一层子目录（checkpoints / diffusion_models / loras…），
  模型列表随选择联动；默认「（全部）」合并所有目录。已注册目录（含
  unet/clip 历史别名）走 folder_paths，未注册目录（ipadapter/onnx 等）
  由插件自行递归扫描。
- 新增 `GET /api/model_toolkit/folder_files`：返回各模型目录的文件清单，
  供前端联动下拉使用（仅在 ComfyUI 运行时注册，测试环境不引入依赖）。
- 新增前端扩展 `js/model_dir_filter.js`（`WEB_DIRECTORY = "./js"`）：
  含旧工作流兼容（保留已选值）与刷新按钮后重新过滤。

### 工程

- 新增 `model_toolkit/folders.py`（目录枚举 / 路径解析）与
  `tests/test_folders.py`；删除 `nodes/unet_inspector.py`、
  `tests/test_unet_blocks.py` 与 golden.json 的 unet 场景
  （预期行为变化：节点已不存在；basic / advanced 样本哈希不变）。
- 单测总数 89（原 79 − 5 个 UNET 用例 + 15 个新用例）；黄金样本 3/3 通过。

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
