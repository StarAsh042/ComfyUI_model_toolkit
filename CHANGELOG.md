# 更新日志

## [Unreleased]

### 新增

- **交互式模型架构报告（P1/P2）**：新增 `model_toolkit/structure/`（UNet 块结构
  解析）、`model_toolkit/report.py`（报告契约与 LRU 缓存）与
  `model_toolkit/server.py`（ComfyUI 路由注册），以及 `web/report/` 报告页
  （概览 / 架构图 / 键浏览 / 精度 / 元信息 / 原始 JSON / 原理图解）。
- 报告页使用本地 ECharts，不访问 CDN；数据经 `textContent` 渲染，CSV 导出
  防公式注入，Markdown 导出转义，JSON/CSV/Markdown 均可下载。
- 模型检查器新增 `Open Interactive Report` 按钮：先
  `POST /api/model_toolkit/report/prepare` 清缓存并按当前模型文件重建报告，
  再在新标签页打开 `/model_toolkit/report?dir=&model=`；按钮为
  `serialize=false` 的 DOM widget，不写入 `widgets_values`。
- 新增路由：`GET /model_toolkit/folder_files`、`GET /model_toolkit/report/data`、
  `GET /model_toolkit/report`、静态 `GET /model_toolkit/report/assets/*`、
  `POST /model_toolkit/report/prepare`；`dir`/`model` 必须同时通过目录白名单与
  文件清单，解析后的真实路径须位于 `models/<dir>` 或 `extra_model_paths` 内。
- 本阶段节点仍保留原有文字与图片输出，计划在 P4 / 2.0.0 移除。

### 工程

- 新增 `tests/test_structure_unet.py`、`tests/test_report.py`、`tests/test_server.py`
  与 `tests/frontend/report_url.test.mjs`、`report_logic.test.mjs`；
  Python 单测 144 项、Node 前端测试 34 项全部通过。

## [1.3.0] - 2026-10-24

### 破坏性变更

- **删除「🔬 UNET Inspector」节点**：其 UNET 块结构分析已包含在模型检查器的
  `Basic` 报告中（且附带组件占比与图表），属于重复功能。旧工作流中的该
  节点将失效，请改用「🔍 Model Inspector」。
- **模型检查器不再提供「（全部）」目录**：`model_dir` 默认 `checkpoints`，
  模型列表严格只显示所选目录内的模型；以 JSON 直接调用 `/api/prompt`
  的脚本不传该键时也按 `checkpoints` 解析，不再跨目录回退。
- **界面与输出改为英文**：节点显示名、分类、显示模式（`Basic` / `Advanced`）、
  输出名、文本报告、图表与错误信息均为英文。中文 i18n 翻译在后续版本提供。

### 新增

- **模型检查器新增「模型目录」选择**：顶部新增 `model_dir` 下拉，列出
  `ComfyUI/models/` 下第一层子目录（checkpoints / diffusion_models / loras…），
  默认 `checkpoints`，模型列表随选择严格联动。已注册目录（含 unet/clip
  历史别名）走 folder_paths，未注册目录（ipadapter/onnx 等）由插件自行
  递归扫描。
- 新增 `GET /api/model_toolkit/folder_files`：返回各模型目录的文件清单，
  供前端联动下拉使用（仅在 ComfyUI 运行时注册，测试环境不引入依赖）。
- 新增前端扩展 `js/model_dir_filter.js`（`WEB_DIRECTORY = "./js"`）：
  目录切换后只保留当前目录的文件，刷新按钮后重新过滤；路由不可用时仅
  `checkpoints` 退回静态列表。
- 路径解析增加模型名边界校验：拒绝绝对路径与 `..` 穿越（模型名可能来自
  工作流 JSON）。

### 修复

- **1.2.0 工作流控件错位**：`model_dir` 加在首位后，旧工作流按位置保存的
  三值数组会整体串位。节点加载时在 `configure` 恢复控件前识别旧格式并
  右移一位，`model_dir` 补为 `checkpoints`，中文显示模式转为 `Basic` /
  `Advanced`。
- **旧 1.3.0 工作流迁移**：`（全部）` 改为 `checkpoints`；新版前端保存的
  `widgets_values_named` 同样归一化，避免命名恢复绕过位置迁移。
- `tests/golden_check.py` 在 Windows GBK 控制台下输出 emoji 会崩溃，
  现统一按 UTF-8 输出。

### 工程

- 新增 `model_toolkit/folders.py`（目录枚举 / 严格路径解析）与
  `tests/test_folders.py`；删除 `nodes/unet_inspector.py`、
  `tests/test_unet_blocks.py` 与 golden.json 的 unet 场景。
- 新增 `js/widget_migration.js`、Node 单测 12 项与 1.2.0 工作流夹具。
- Python 单测 88 项全部通过；Node 前端测试 12/12；黄金样本 3/3 通过。
- **黄金基线重建**：报告/图表改为英文且默认目录变为 checkpoints，文本与
  图像哈希变化属预期；环境记录同步为 ComfyUI 0.42.5。

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
