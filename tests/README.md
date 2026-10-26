# 测试与黄金样本

环境：ComfyUI 自带 Python（3.11.6）+ torch 2.14.0+cpu。
ComfyUI 根目录默认 `D:\AIGC\ComfyUI`，可用环境变量 `COMFYUI_DIR` 覆盖。

## 单元与契约测试（G1 / G2 / G6）

```powershell
cd D:\AIGC\ComfyUI\custom_nodes\ComfyUI_model_toolkit
& "D:\AIGC\ComfyUI\.ext\python.exe" -m unittest discover -s tests -v
```

共 278 项；其中 16 项需要本机真实 SD1/SDXL 模型，模型不存在时自动跳过。
CI（GitHub Actions）没有 ComfyUI 与真实模型，使用
`tests/ci/folder_paths.py` 最小替身运行纯逻辑测试：

```bash
COMFYUI_DIR="$PWD/tests/ci" python -m unittest discover -s tests -v
```

替身会创建空的 checkpoints / loras / diffusion_models 目录与一个占位
`.safetensors`，因此目录枚举类测试仍可执行；真实模型用例显示为 skipped。
Runner 还需安装 `tests/requirements-ci.txt`（numpy / pyyaml，仅测试与模板工具链使用，
不属于插件运行依赖）；`aiohttp` 已列入插件运行依赖。工作流定义见
`.github/workflows/ci.yml`（Python + Node + Ruff，聚合 job `ci`）。
工作流在跑单元测试前会执行 `python tools/build_template_index.py --check`，
保证 `index.json` 与 YAML 源数据同步。

## 前端纯函数测试（Node 24+）

```powershell
cd D:\AIGC\ComfyUI\custom_nodes\ComfyUI_model_toolkit
node --test tests/frontend/*.test.mjs
```

共 63 项：控件迁移 12 项（1.2.0/1.3.0 四/三控件 → 2.0.0 两控件）、报告 URL 7 项、
报告页纯函数 26 项（格式化、过滤/排序/分页、`resolveKeyView` 跨分页与空前缀、
状态/菜单文案、CSV 文件名/失败计划/公式注入、Markdown 转义、文件名清洗、
100k 键与 20k 行性能）、UNet 交互视图 10 项（基础分辨率换算、阶段与块前缀、
键过滤、上下文维度、U 形布局对齐、跳连/文本上下文/时间条件连线、移动端折叠）、
原理图数据联动 4 项（事实提取、缺模块置灰、引用与论文链接）、报告文档契约 4 项
（无位图导出承诺/无键名可选章节、分页平面表与两个 CSV 范围、Markdown 不含键名）。
报告页的 DOM 渲染与节点按钮需要在真实 ComfyUI 中验收。

`tests/fixtures/workflow_1.2.0_model_inspector.json` 是 1.2.0 保存的工作流，
使用与黄金样本相同的本机模型，用于在 ComfyUI 中手动验证控件位置迁移：
加载后 `model_dir` 应为 `checkpoints`，`model_name` 不变，`show_details` /
`show_charts` 两个旧控件已被移除。

## 黄金样本（G9，需真实模型，约 15 秒）

```powershell
& "D:\AIGC\ComfyUI\.ext\python.exe" tests\golden_check.py
```

2.0.0 起覆盖三个场景：完整报告 JSON、`include_keys=False` 的报告、
替换器错误路径。比对报告 JSON 的规范化 SHA256（`sort_keys` + 紧凑分隔符）与
摘要字段；节点零输出后不再比对文本/图像哈希。黄金样本需要本机真实模型，
**不在 CI 运行**。

## 服务层路由（P2）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/model_toolkit/folder_files` | 各模型目录的文件清单（目录联动下拉） |
| GET | `/model_toolkit/report/data?dir=&model=&keys=1` | 报告 JSON；`keys=0` 去掉键名清单 |
| GET | `/model_toolkit/report` | 报告页 `web/report/index.html`；未安装时返回 404 `REPORT_PAGE_MISSING` |
| GET | `/model_toolkit/report/assets/*` | `web/report/` 存在时静态托管 |
| POST | `/model_toolkit/report/prepare` | 清空报告缓存并以当前模型文件重建；返回 `{"ok": true, "schema_version": ...}` |

路由由 `model_toolkit/server.py` 集中注册，根包只调用一次；`dir`/`model` 必须同时通过
目录白名单与文件清单，解析后的真实路径必须位于 `models/<dir>` 或 `extra_model_paths`
注册目录内。错误响应固定为 `{"error": "...", "code": "..."}`，不含堆栈与绝对路径。

报告页（`web/report/`）通过 `/model_toolkit/report?dir=&model=` 在新标签页打开；
页面从 `/api/model_toolkit/report/data` 取 JSON，图表使用本地 ECharts，不访问 CDN。
Architecture 标签页的 UNet 结构视图（`web/report/views/unet.js`）与 Principles 标签页的
原理示意图（`web/report/principles/principles.js`）都是自绘 SVG，不依赖任何图表库，
导出的 SVG 保留原有样式。点击节点按钮会先 `POST /api/model_toolkit/report/prepare` 重建报告，再导航到报告页，
因此页面始终反映磁盘上的当前模型文件；该请求不执行工作流、也不写文件。
节点按钮是 `serialize=false` 的 DOM widget，不写入 `widgets_values`。
2.0.0 起 Model Inspector 为零输出节点（`RETURN_TYPES = ()`），
执行 `prepare_report` 只校验模型并预热缓存；后续新增/删除控件时仍需迁移
`widgets_values`，避免旧工作流按位置错位。

## 约定

- `test_behavior.GroupModelKeysTest` 与 `RealCheckpointGroupingTest` 锁定 B1 修复后的分类：
  CLIP 文本编码器键必须归 CLIP；真实模型分组数为 UNET 686 / CLIP 197 / VAE 248 / EMA 2，
  未分类键归 `Other`。再改分类逻辑时，先改测试再改实现。
- `test_structure_unet.py` 锁定 UNet 结构解析：合成 LDM / diffusers 键名验证命名与阶段，
  真实 SD1 / SDXL 只读头部验证阶段数、上下文维度；`InvariantMixin` 要求每个 UNET 键
  恰好归属一次（参数与键数守恒，含 `totals.by_stage` 覆盖 down/middle/up）。
- `test_report.py` 锁定报告契约：JSON 可序列化、无服务器绝对路径、元信息按数据处理、
  `include_keys` 行为与错误码（`MODEL_NOT_FOUND` / `MODEL_LOAD_FAILED` / `ANALYSIS_FAILED`）。
  报告缓存键包含 realpath、mtime、大小与 `include_keys`；改文件后需 mtime 变化才失效。
- `test_server.py` 锁定服务层：`dir`/`model` 白名单与路径包含性（含 `extra_model_paths`
  注册目录）、`keys=0/1`、prepare 请求体校验与强制重建、`ReportError` 到 HTTP 状态码的映射、
  错误响应不泄露堆栈，以及 `server` 不可用时跳过路由注册。
- `test_contract.py` 锁定节点契约：Model Inspector 在 2.0.0 为零输出
  （`RETURN_TYPES = ()`、`OUTPUT_NODE = True`、`FUNCTION = prepare_report`），
  只有 `model_dir` / `model_name` 两个输入；其余节点契约不变。
- `test_index.py` 的 `PrepareReportTest` 要求节点执行只预热报告缓存、
  不得调用 `ModelUtils.load_model`（全量加载）；报告 `summary.model_dir`
  记录解析时使用的目录。
- `test_packaging.PyprojectTest` 锁定发布元数据：`pyproject.toml` 版本必须与
  `__version__` 一致、依赖动态引用 `requirements.txt`、`[tool.comfy]` 发布信息存在；
  发布前必须同时修改两处版本号。
- `test_packaging.RuntimeImportClosureTest` 用 AST 扫描 19 个包内 Python 文件，
  运行期第三方导入必须恰好是 `aiohttp` / `safetensors` / `torch`；前两者必须在
  `requirements.txt` 声明，`torch` 由宿主提供。`pyyaml` 只允许出现在
  `tests/requirements-ci.txt` 与 dev 依赖中。
- `test_release_integrity.py` 静态锁定发布守卫与 CI：workflow 级串行、仅 main 发布、
  版本守卫位于发布 action 之前且不读 secret、不以 tag 为基线；索引工具在
  utf-8 / cp936 下均以 `INDEX_SYNC_OK` 开头退出 0，缺文件/漂移/缺 CSafeLoader
  使用对应 ASCII 令牌。测试不访问网络。
- 黄金样本基线于 2026-10-22 因 B1/B3/B4/B2 修复重建过；
  2026-10-24 因界面/输出英文化、默认目录改为 `checkpoints` 再次重建；
  2026-10-25（2.0.0）随节点零输出切换为报告 JSON 规范化哈希，旧文本/图像
  哈希退役。
- `golden_check.py --write` 仅用于重建基线；重建前必须确认行为变化是预期的，并在提交信息里写明原因。
