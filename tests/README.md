# 测试与黄金样本

环境：ComfyUI 自带 Python（3.11.6）+ torch 2.14.0+cpu。
ComfyUI 根目录默认 `D:\AIGC\ComfyUI`，可用环境变量 `COMFYUI_DIR` 覆盖。

## 单元与契约测试（G1 / G2 / G6）

```powershell
cd D:\AIGC\ComfyUI\custom_nodes\ComfyUI_model_toolkit
& "D:\AIGC\ComfyUI\.ext\python.exe" -m unittest discover -s tests -v
```

## 前端纯函数测试（Node 24+）

```powershell
cd D:\AIGC\ComfyUI\custom_nodes\ComfyUI_model_toolkit
node --test tests/frontend/*.test.mjs
```

共 48 项：控件迁移 12 项、报告 URL 7 项、报告页纯函数 15 项（格式化、过滤/排序/分页、
CSV/Markdown 转义、文件名清洗）、UNet 交互视图 10 项（基础分辨率换算、阶段与块前缀、
键过滤、上下文维度、U 形布局对齐、跳连/文本上下文/时间条件连线、移动端折叠）、
原理图数据联动 4 项（事实提取、缺模块置灰、引用与论文链接）。
报告页的 DOM 渲染与节点按钮需要在真实 ComfyUI 中验收。

`tests/fixtures/workflow_1.2.0_model_inspector.json` 是 1.2.0 保存的工作流，
使用与黄金样本相同的本机模型，用于在 ComfyUI 中手动验证控件位置迁移：
加载后 `model_dir` 应为 `checkpoints`，`show_details` 应为 `Basic`，模型选择不变。

## 黄金样本（G9，需真实模型，约 15 秒）

```powershell
& "D:\AIGC\ComfyUI\.ext\python.exe" tests\golden_check.py
```

覆盖三个场景：`Basic` 报告、`Advanced` 报告 + 图表、替换器错误路径。
比对文本 SHA256、图像张量 shape/dtype/完整字节 SHA256。

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
节点按钮是 `serialize=false` 的 DOM widget，不写入 `widgets_values`，因此不会影响
1.2.0 / 1.3.0 工作流的控件位置迁移；P2 阶段节点仍保留原有文字与图片输出。

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
- 黄金样本基线于 2026-10-22 因 B1/B3/B4/B2 修复重建过；
  2026-10-24 因界面/输出英文化、默认目录改为 `checkpoints` 再次重建，
  文本与图表哈希变化均属预期（原因见 CHANGELOG）。
- `golden_check.py --write` 仅用于重建基线；重建前必须确认行为变化是预期的，并在提交信息里写明原因。
