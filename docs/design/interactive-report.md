# 交互式报告与架构特化视图 — 实施计划（v2）

> 状态：**已批准（2026-10-24）**；P0 代码完成（含默认目录/英文调整）；P1 数据层完成；P2 服务层、报告页骨架与节点按钮完成（2026-10-24，按钮先重建报告再打开；待界面验收）；P3 完成（2026-10-24：自绘 UNet U 形交互视图 + 5 张原理示意图，Node 48 项 / Python 144 项 / 黄金 3/3 与真实浏览器验证通过；待用户界面验收）
> 日期：2026-10-24
> 关联：1.3.0「模型目录」选择（PR #18）；README 待开发项「交互式组件浏览器」「参数差异比较」
> 本文替代 v1 设计稿。按第 14 节拆 PR 实施。

---

## 0. 你的决策（已写入方案）

| # | 决策 | 对方案的影响 |
|---|---|---|
| 1 | 图表库用 **ECharts** | 本地 vendor，不走 CDN |
| 2 | 页面入口用 **新标签页** | 节点按钮 `window.open`；不做节点内 iframe |
| 3 | **不加**节点 JSON 输出插槽 | 页面内部仍走 HTTP 取 JSON，但节点不再多一个输出 |
| 4 | **取消**原有文字和图片输出 | 节点变为零输出节点；删除 matplotlib 图表链路 |
| 5 | v1 **不做**双模型对比 | 契约与页面结构预留，不实现 |
| 6 | 页面按架构**特化展示**，先做 UNet | 建立可插拔的 structure/view 体系；v1 只实现 UNet |
| 7 | 需要展示**架构原理**（参考 PicDoc 的专业度） | 自绘 SVG 原理图 + 论文引用 + 与实际模型数据联动 |
| 8 | 先出计划 | 本文；批准后才动代码 |
| 9 | 去掉「（全部）」，默认 `checkpoints` | 模型列表严格只显示所选目录；不再跨目录回退 |
| 10 | 界面与输出先全英文 | 中文 i18n 翻译后置，不在 1.3.0/2.0.0 阻塞范围内 |

---

## 1. 结论摘要

把模型检查器从「文字 + 图片节点」改造成「**模型架构浏览器**」：

- 节点只保留两个选择项（模型目录、模型文件）和一个「打开交互报告」按钮；
- 报告页按 **概览 → 架构 → 原理 → 键名 → 精度 → 元信息 → 原始 JSON** 组织；
- **UNet 特化视图**用 U 形结构图展示 down / mid / up 阶段、ResBlock、SpatialTransformer、下采样/上采样、跳连、时间与文本条件，并把真实参数量标在图上；
- **原理页**用自绘 SVG 讲解 Latent Diffusion、UNet 数据流、ResBlock、SpatialTransformer、SDXL 差异，标注论文出处；
- 所有数字标注**数据来源**（权重 / 模板 / 约定 / 未知），不猜测；
- 页面与 ComfyUI 扩展入口分目录，页面资源不拖慢 ComfyUI 启动；
- 这是一次**破坏性发布（2.0.0）**：旧的文字/图片输出被移除。

---

## 2. 目标与非目标

### 目标

1. 用一份结构化报告数据同时支撑：概览、架构图、原理图、键名浏览器、导出。
2. UNet 视图做到「图上每个块都能对应到真实权重和参数量」，可点击下钻到键名。
3. 原理图达到论文配图级专业度：模块、连接、维度、条件注入位置准确，且标明出处。
4. 为 DiT（Flux/SD3）与 LLM 留出清晰的扩展点，不在 v1 硬编码未来架构。
5. 移除节点输出后，页面成为唯一结果面；旧工作流有明确迁移说明。

### 非目标（v1）

- 不做双模型对比、不做历史报告、不做账号/权限。
- 不做权重热力图、不加载权重做数值分析。
- 不在 v1 实现 DiT/LLM 特化视图（只留接口与占位提示）。
- 不复制 PicDoc 或论文原图；原理图全部原创绘制并引用论文。

---

## 3. 平台约束（决定架构的事实）

| 约束 | 事实 | 影响 |
|---|---|---|
| 节点输出是类型化插槽 | `STRING`/`IMAGE` 等；网页不是插槽类型 | 交互 UI 必须走前端页面 |
| `INPUT_TYPES` 只在启动时求值 | 服务端无法做控件间联动 | 模型目录联动靠 JS + 路由（1.3.0 已实现） |
| `WEB_DIRECTORY` 下的 `.js` 会被**递归自动 import** | ComfyUI `server.py:421-433` | 页面脚本不能放 `js/`，必须放 `web/report/` |
| ComfyUI 默认无鉴权 | 报告页与 ComfyUI 同源 | XSS/路径遍历必须从第一版守住 |
| 零输出节点可行 | `RETURN_TYPES = ()` + `OUTPUT_NODE = True` + 返回 `{}`（ComfyUI 自带 `SaveImageWebsocket` 即此模式） | 可以彻底取消文字/图片输出 |
| 旧工作流按控件**位置**保存值 | `widgets_values` 是数组 | 新增/删除控件需要迁移逻辑（见 12.3） |

---

## 4. 输出模型重构（2.0.0 破坏性变更）

### 4.1 变更前后

| | 现在（1.3.0） | 2.0.0 |
|---|---|---|
| 输入 | `model_dir`、`model_name`、`show_details`、`show_charts` | `model_dir`、`model_name` |
| 输出 | `STRING` 分析报告、`IMAGE` 可视化图表 | **无输出**（`RETURN_TYPES = ()`） |
| 执行语义 | 输出文字和图片 | 校验模型并预热报告缓存；按钮打开页面 |
| 图表实现 | matplotlib（2048×1280 PNG） | 浏览器 ECharts / 自绘 SVG |
| 文字报告 | 节点输出 | 页面「导出 Markdown」（前端生成） |
| 依赖 | numpy、matplotlib、Pillow、safetensors、pyyaml | safetensors、pyyaml（其余移除） |
| 版本 | 1.3.0 | **2.0.0** |

### 4.2 节点契约

```python
class CheckpointInspector:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_dir": (list_model_dirs(), {"default": DEFAULT_DIR}),
            "model_name": (folder_paths.get_filename_list("checkpoints"),),
        }}

    RETURN_TYPES = ()
    FUNCTION = "prepare_report"
    OUTPUT_NODE = True          # 保证工作流有效、可入队
    CATEGORY = "model_toolkit/Analysis"

    def prepare_report(self, model_name, model_dir=DEFAULT_DIR):
        build_report(model_dir, model_name)   # 校验 + 预热缓存
        return {}
```

前端按钮：

```js
// js/report_button.js（会被 ComfyUI 自动加载）
const tab = window.open("about:blank", "_blank")                          // 1. 同步打开，避免弹窗拦截
await fetch(REPORT_PREPARE_PATH, { method: "POST", body: payload })       // 2. 重建报告并预热缓存
tab.location.href = buildReportUrl(origin, dir, model)                    // 3. 导航到报告页
```

> P2 实现调整（2026-10-24 用户要求）：按钮必须先重建报告再打开页面，
> 保证页面不会读到旧分析结果。刷新走独立的 POST prepare 路由，不把整张
> 工作流加入队列，避免误触发生图；P4 后该路由与节点 `prepare_report`
> 合并为同一执行语义。

### 4.3 为什么分阶段移除输出

**不要**在页面还没验收时先删输出——那会出现「功能真空期」。计划把移除放在最终发布 PR：

1. 先加数据层与页面，旧输出仍在（可随时对照）；
2. 页面通过验收后，再删输出、图表代码与依赖（2.0.0）。

### 4.4 迁移影响（必须写进 CHANGELOG/README）

- 旧工作流里连到「分析报告」「可视化图表」的 `ShowText` / `SaveImage` 等节点会断开；
- 需要重新添加检查器节点（或删除悬空连线）；
- 旧的 `show_details`、`show_charts` 控件消失，页面内有对应视图开关；
- 1.3.0 工作流的前两个控件 `model_dir`、`model_name` 位置不变，可正常映射
  （旧的「（全部）」按 12.3 改为 `checkpoints`）；
- 1.2.0 工作流需要控件迁移（见 12.3）。

---

## 5. 总体架构

```
模型文件
   │
   ▼
[分析层] ModelUtils / AnalysisMixin / PrecisionMixin
   │  读取 safetensors 头部（.ckpt 全量加载）
   ▼
[结构层] structure.unet.extract()  ──►  stages/blocks/capabilities/来源标注
   │
   ▼
[报告层] report.build_report()  ──►  一份 dict（唯一数据源）
   │                              ├─ 页面 JSON（/model_toolkit/report/data）
   │                              ├─ 页面 Markdown/CSV 导出（前端）
   │                              └─ 缓存（LRU）
   ▼
[服务层] model_toolkit/server.py
   │  GET  /model_toolkit/report          页面
   │  GET  /model_toolkit/report/data     报告 JSON
   │  POST /model_toolkit/report/prepare  重建报告并预热缓存（按钮调用）
   │  GET  /model_toolkit/report/assets/* 页面静态资源
   ▼
[前端] web/report/  ── ECharts（图表）+ 自绘 SVG（架构/原理）+ 原生 JS
   ▲
   │ 新标签页
[节点] js/report_button.js  ── 按钮
```

文件职责：

| 文件 | 职责 | 不负责 |
|---|---|---|
| `model_toolkit/report.py` | 组装报告、缓存、序列化 | HTTP、渲染 |
| `model_toolkit/structure/unet.py` | 从键名还原 UNet 结构 | 读文件 |
| `model_toolkit/structure/__init__.py` | 按架构选择提取器（registry） | 具体解析 |
| `model_toolkit/server.py` | 路由、参数校验、线程池 | 业务计算 |
| `web/report/report.js` | 页面编排、tab、数据请求 | 架构解析 |
| `web/report/views/unet.js` | UNet SVG 渲染与交互 | 计算参数 |
| `web/report/principles/*` | 原创原理图 + 文案 + 引用 | 读取模型 |
| `js/report_button.js` | 节点按钮与 URL 构造 | 渲染报告 |

依赖方向单向：`web → server → report → structure → core`。

---

## 6. 数据层

### 6.1 `build_report()`

```python
def build_report(model_dir, model_name, include_keys=True) -> dict:
    """读取模型、计算统计、提取结构，返回可 JSON 序列化的报告。"""
```

- 只读 safetensors 头部；`.ckpt/.pt` 回退全量加载（沿用现状）；
- 每次调用先查缓存（见 10.2）；
- 分析失败抛出明确异常，由路由转成 500 / 页面错误态。

### 6.2 结构提取框架

```
structure/
  __init__.py   # extract(model_data, groups, arch) -> dict | None
  common.py     # Stage/Block 数据类、参数累加、键归属、来源标注
  unet.py       # LDM 命名 + diffusers 命名
  generic.py    # 无法特化时的组件级回退
```

选择规则：

1. 架构模板命中 SD1/SDXL（覆盖率 ≥95%）且存在 `input_blocks` / `down_blocks` → `unet` 提取器；
2. Flux/SD3 等 MMDiT → v1 返回 `generic`，页面显示「DiT 特化视图规划中」；
3. 其余 → `generic`。

未来新增 `mmdit.py`、`llm.py` 时，只需注册提取器与对应前端 view，不改报告层。

### 6.3 UNet 结构提取（v1 核心）

同时支持两种键名：

| 结构 | LDM / CompVis / kohya | diffusers |
|---|---|---|
| 输入卷积 | `input_blocks.0.0` | `conv_in` |
| 下阶段 | `input_blocks.N.*` | `down_blocks.N.*` |
| 中间 | `middle_block.*` | `mid_block.*` |
| 上阶段 | `output_blocks.N.*` | `up_blocks.N.*` |
| 下采样 | `input_blocks.N.0.op` | `down_blocks.N.downsamplers.0` |
| 上采样 | `output_blocks.N.*.conv` | `up_blocks.N.upsamplers.0` |
| 时间嵌入 | `time_embed.*` | `time_embedding.*` |
| 尺寸条件 | `label_emb.*` | `add_embedding.*` |

提取规则（全部从权重推导，不硬编码层数）：

1. **阶段**：按 `input_blocks` / `down_blocks` 的索引归组；检测到 `op` / `downsamplers` 即为该阶段的降采样；
2. **分辨率**：以「相对分辨率」表示（1/1、1/2、1/4…），由降采样次数推导；绝对像素是推理设置，不由权重决定，页面提供 512/1024/自定义 开关；
3. **通道**：从 ResBlock 最后一个卷积的 `[out, in, k, k]` 取 `out`；
4. **块类型**：
   - ResBlock：含 `in_layers`/`out_layers`/`emb_layers`（LDM）或 `resnets`（diffusers）；
   - Transformer：含 `transformer_blocks`（LDM）或 `attentions`（diffusers）；
   - 自注意力：`attn1`；交叉注意力：`attn2`；
   - 交叉注意力上下文维度：从 `attn2.to_k.weight` 的 `[inner, context]` 推导（768 / 1024 / 2048 等）。
5. **注意力头数**：权重文件不保存 head 数，**不猜测**，显示「未在权重中保存」；
6. **参数量**：每个键归属到唯一节点；无法归属的键进入 `unattributed` 桶，保证：
   `sum(所有节点 params) + unattributed.params == UNET 组件参数量`（EMA 已排除）。

真实模型核对（本机）：

| 模型 | input_blocks | middle | output_blocks | 下采样 | 特殊模块 |
|---|---|---|---|---|---|
| SD1.5 | 0–11 | 0–2 | 0–11 | 3 个 | time_embed |
| SDXL | 0–8 | 0–2 | 0–8 | 2 个 | time_embed + label_emb |

### 6.4 结构 JSON（示例，节选）

```jsonc
"structure": {
  "kind": "unet",
  "naming": "ldm",
  "latent": { "in_channels": 4, "base_size": 512, "latent_size": 64 },
  "capabilities": {
    "time_embedding": true,
    "self_attention": true,
    "text_cross_attention": true,
    "size_conditioning": false,
    "dual_text_encoder": false,
    "refiner": false,
    "v_pred": false
  },
  "io": {
    "conv_in":  { "params": 11520, "in_channels": 4, "out_channels": 320, "source": "weights" },
    "conv_out": { "params": 1290,  "in_channels": 320, "out_channels": 4,  "source": "weights" },
    "time_embed": { "params": 1234567, "source": "weights" }
  },
  "stages": [
    {
      "id": "down.0",
      "direction": "down",
      "level": 0,
      "resolution_ratio": 1.0,
      "channels": 320,
      "blocks": [
        { "id": "input_blocks.1.0", "kind": "resblock", "params": 1234567,
          "children": [
            { "kind": "norm", "params": 640 },
            { "kind": "conv", "params": 921600, "shape": [320, 320, 3, 3] },
            { "kind": "time_mlp", "params": 409600 }
          ] },
        { "id": "input_blocks.1.1", "kind": "transformer", "params": 456789,
          "attention": { "self": true, "cross": true, "context_dim": 768,
                          "inner_dim": 320, "heads": null } }
      ],
      "transition": { "kind": "downsample", "params": 184320, "stride": 2,
                       "source": "convention" }
    }
  ],
  "middle": { "blocks": [ /* ResBlock, Transformer, ResBlock */ ] },
  "up": [ /* 与 down 对称，含 skip 连接来源 */ ],
  "extras": [ { "kind": "label_embed", "params": 1234567 } ],
  "unattributed": { "params": 0, "key_count": 0 },
  "totals": { "params": 860000000, "by_stage": [/* ... */] }
}
```

每项可带 `source`：

| 来源 | 含义 |
|---|---|
| `weights` | 由张量形状/键名直接得出（最可信） |
| `template` | 由 sd-mecha 架构模板判定 |
| `convention` | 由约定推导（如 Downsample 的 stride=2） |
| `user` | 页面上的用户选择（如基准分辨率） |
| `unknown` | 权重中不存在（如注意力头数） |

> 专业性的一条硬规则：**不把推断值伪装成事实**。例如分辨率标注为「相对」并注明受推理设置影响；头数显示"未保存"而不是猜 8。

---

## 7. 服务层

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/model_toolkit/report` | 返回 `web/report/index.html`，参数 `dir`、`model` |
| GET | `/model_toolkit/report/data?dir=&model=&keys=1` | 返回报告 JSON；`keys=0` 时不含键名清单 |
| POST | `/model_toolkit/report/prepare` | 清缓存并重建报告（JSON `{dir, model}`）；按钮打开页面前调用 |
| GET | `/model_toolkit/report/assets/*` | `routes.static()` 托管 `web/report/` |

实现要点：

- `model_toolkit/server.py` 集中注册；`__init__.py` 只调用一次；
- 仅在 `server` 已加载且 `PromptServer.instance` 存在时注册（沿用 1.3.0）；
- `await asyncio.to_thread(build_report, ...)`，避免 `.ckpt` 阻塞事件循环；
- 参数校验：`dir` 必须在 `list_model_dirs()`，`model` 必须在 `list_dir_files(dir)`；解析后 `realpath` 必须位于 `models_dir` 内；
- GET 只读；POST prepare 只清缓存并重建报告，不写任何文件、不排队执行工作流；
- 错误 JSON：`{ "error": "...", "code": "..." }`，不返回堆栈。

---

## 8. 前端

### 8.1 页面结构

```
┌───────────────────────────────────────────────────────────────┐
│ 模型名 · 架构 · 参数量 · 精度 · 大小 · 目录      [复制链接] [导出] │
├───────────────────────────────────────────────────────────────┤
│ 概览 | 架构 | 原理 | 键名 | 精度 | 元信息 | 原始 JSON            │
├───────────────────────────────────────────────────────────────┤
│ 概览：组件环形图 + 参数条形图 + 架构能力提示                     │
│ 架构：UNet 特化 U 形结构图（v1）                                 │
│ 原理：Latent Diffusion / UNet 数据流 / ResBlock / SDXL 差异      │
│ 键名：搜索 + dtype 过滤 + 排序 + 组件→块→层折叠树                │
│ 精度：分布饼图 + 每组件精度                                     │
│ 元信息：键值表                                                  │
│ 原始 JSON：只读 + 复制                                          │
└───────────────────────────────────────────────────────────────┘
```

默认 tab：有 UNet 特化结构时打开「架构」，否则「概览」。
页面文案与节点保持一致，先全英文；中文 i18n 后置（见第 16 节决策 10）。

### 8.2 UNet 特化视图（v1 重点）

布局（U 形，从左到右阅读方向）：

```
        文本条件 (CLIP) ──────────── cross-attn ──────────────┐
                                                              ▼
latent ─ conv_in ─ [L0: Res, SA/CA] ─┐                  ┌─ [L2: Res, SA/CA] ─ ... ─ conv_out ─ latent'
                                    │  downsample       ▲ upsample
                 [L1: Res, SA/CA] ──┘                  └── [L1': Res, SA/CA]
                                    │                       ▲
                                   [Middle: Res, SA/CA, Res]
        时间步 t ──► time_embed ──► 注入每个 ResBlock（AdaGN/scale-shift）
```

每个阶段卡片显示：

- 阶段编号、相对/绝对分辨率、通道数；
- 块序列（ResBlock / Transformer，带 SA/CA 徽标）；
- 参数量与占比（hover 显示明细）；
- 点击卡片 → 右侧详情面板：子层列表（norm/conv/attn/ff）、shape、dtype、参数量；
- 「在键名中查看」→ 切到键名 tab 并按其前缀过滤；
- 跳连用曲线明确画出；未检出的模块（如 SD1.5 的 label_emb）灰显并标注「本模型未包含」。

交互：

- 缩放/平移/适应窗口/导出 SVG；
- 视图开关：参数 / 占比 / 仅结构；
- 基准分辨率选择（SD1.5 默认 512、SDXL 默认 1024，可改）；
- 移动端：横向滚动 + 卡片折叠，不强行塞进一屏。

实现：**自绘 SVG**（`views/unet.js`），不用 ECharts Graph——节点少、布局固定、可精确控制标注与导出；ECharts 只负责概览/精度图。

### 8.3 原理页（专业性）

内容分四层，全部原创 SVG：

| 图 | 内容 | 与实际模型联动 |
|---|---|---|
| `ldm-overview.svg` | 文本 → CLIP → UNet（迭代去噪）→ VAE 解码；图像 → VAE 编码 | 高亮本模型使用的文本编码器/是否含 refiner |
| `unet-dataflow.svg` | down / mid / up、跳连、时间与文本条件注入 | 按真实阶段数、通道数、注意力有无着色 |
| `resblock.svg` | GroupNorm → SiLU → Conv → 时间嵌入 MLP → Conv → 残差 | 显示本模型 ResBlock 的实际参数量 |
| `spatial-transformer.svg` | GroupNorm → proj_in → 自注意力 → 交叉注意力 → FFN → proj_out | 标注实际 context_dim；头数标「未知」 |
| `sdxl-diffusion.svg` | 双文本编码器、pooled + size conditioning、UNet 规模、可选 refiner | 根据 `label_emb` / 双 CLIP 检测结果灰显 |

每条原理配 2–4 句人话解释 + 公式（可选）+ 论文引用：

- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022（arXiv:2112.10752）
- Podell et al., *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*, 2023（arXiv:2307.01952）
- 未来 DiT：Peebles & Xie 2023（arXiv:2212.09748）；SD3/MMDiT：Esser et al. 2024（arXiv:2403.03206）
- 未来 LLM：Vaswani et al. 2017（arXiv:1706.03762）

> **不复制 PicDoc 或论文原图**：PicDoc 是商业工具、未声明可复用许可；论文图通常受版权保护。我们只参考其"模块框 + 箭头 + 维度标注"的表达方式，自行绘制，并在页面注明「示意图为原创，参考论文 X」。

### 8.4 键名浏览器

- 树：组件 → 块（`input_blocks.3` / `down_blocks.1`）→ 其余路径；
- 搜索：子串、忽略大小写、150 ms 防抖、命中高亮与计数；
- 过滤：dtype、未归类、仅注意力层；
- 排序：名称 / 参数量；
- 渲染：默认折叠，按需展开；可见行超过 2,000 时窗口化；
- 行内显示键名、shape、dtype、参数量；单击复制键名。

### 8.5 导出

- JSON（完整报告）
- CSV（键名表；单元格按 CSV 注入规则转义）
- Markdown（前端生成：概览 + 组件 + 精度 + 元信息 + 可选键名）
- SVG / PNG（当前体系结构图）

### 8.6 DiT / LLM 扩展点

- `structure.extract()` 返回 `kind`；前端 `views/registry.js` 按 `kind` 选择 view；
- v1 对 `kind != "unet"` 显示通用组件视图 + 「DiT/LLM 特化视图规划中」；
- 未来 `mmdit.py`：double/single blocks、AdaLN、joint attention；
- 未来 `llm.py`：embedding → N × (attention + MLP) → logits，支持 GQA/RoPE 标注；
- 原理图按 `architecture.id` 选择，不与结构提取强耦合。

---

## 9. 专业性规则（验收会逐条检查）

1. **可溯源**：每个数字标注来源（weights/template/convention/user/unknown）。
2. **不猜测**：头数、绝对分辨率等权重中不存在的信息，明确写「未保存/受推理设置影响」。
3. **不误导**：SDXL 的 refiner、Flux 的 T5 等未在本模型中出现时，灰显并注明。
4. **一致性**：图上参数量之和必须等于组件统计（允许 `unattributed` 桶），由单测强制。
5. **可核对**：每个块可点开看到构成它的权重键名。
6. **有出处**：原理图配论文引用；示意图原创声明。
7. **专业排版**：统一网格、对齐、字号层级、图例；不使用 AI 生成位图（文字易错、不可编辑）。

---

## 10. 性能

| 场景 | 方案 |
|---|---|
| safetensors | 只读头部（约 5 ms） |
| `.ckpt/.pt` | 全量加载；首次可能数秒，必须缓存 |
| 页面图表 | 浏览器端渲染；报告页不调用 matplotlib |
| 页面体积 | ECharts 仅在报告页加载，不影响 ComfyUI 主界面启动 |
| 键名渲染 | 按需展开 + 防抖 + 超 2,000 行窗口化 |
| 路由线程 | `asyncio.to_thread`，不阻塞 ComfyUI Web 服务 |

缓存：

- 进程内 LRU，`key = (realpath, mtime_ns, size, include_keys)`，`maxsize=8`；
- 只缓存 report dict（含 structure），不缓存张量；
- 同名覆盖时 `mtime/size` 变化自动失效，与 `IS_CHANGED` 指纹一致。

页面性能目标（本机，供验收参考）：

- SD1.5：数据请求 < 1 s（缓存命中 < 200 ms），首屏 < 2 s；
- SDXL：数据 < 2 s，首屏 < 3 s；
- 键名搜索 2,500 键：输入到结果 < 100 ms。

---

## 11. 安全边界

| 风险 | 措施 |
|---|---|
| 路径遍历 | `dir`/`model` 白名单 + `realpath` 包含性校验 + 拒绝绝对路径 |
| XSS | 所有来自模型的数据用 `textContent`；SVG 文本同样转义；CSP `default-src 'self'`；不引入 CDN |
| CSV 注入 | 以 `= + - @` 开头的单元格加 `'` 前缀 |
| 信息泄露 | 不返回服务器绝对路径，只返回相对 `models/<dir>` 的路径 |
| 资源滥用 | GET 只读；POST prepare 仅重建单个报告；LRU 上限；不提供批量接口 |
| 同源风险 | ComfyUI 默认无鉴权：报告页不提供任何写操作；原理图文案为静态可信内容，模型数据一律当不可信输入 |

---

## 12. 兼容与迁移

### 12.1 旧工作流

- 2.0.0 移除输出：连到报告/图表的节点会断开，需要重新添加检查器节点；
- 1.3.0 的两个控件位置不变；
- 页面无需工作流即可打开（书签/分享 URL 可用）。

### 12.2 旧版 ComfyUI

- 路由注册沿用守卫：`server` 未加载或 `PromptServer.instance` 不存在时跳过；
- 路由不可用时，节点按钮提示「当前环境不支持交互报告」，不影响 ComfyUI 启动。

### 12.3 控件位置与取值迁移（**已实现，待界面验收**）

ComfyUI 按位置保存 `widgets_values`。1.3.0（PR #18）把 `model_dir` 加在最前面，导致 **1.2.0 保存的工作流** 载入后错位：

```
1.2.0 保存: [model_name, show_details, show_charts]
1.3.0 读取: model_dir=model_name, model_name=show_details, ...   ← 错
```

现行规则：

- 1.2.0 三值数组（第 2 个是 `基础` / `高级`）→ 整体右移一位，
  `model_dir` 填 `checkpoints`；
- 旧 1.3.0 四值数组里的 `（全部）` → `checkpoints`；
- 中文显示模式 `基础` / `高级` → `Basic` / `Advanced`；
- 新版前端还会保存 `widgets_values_named`，同样归一化，避免按名恢复绕过迁移；
- 迁移只做一次，随后写回标准格式；用 1.2.0 工作流 JSON 做回归测试。

实现：`js/widget_migration.js`（纯函数，12 项 Node 单测）在 `configure`
调用原实现前改写 `info.widgets_values` / `info.widgets_values_named`
（**不能放在 `onConfigure`**：新版 ComfyUI 在控件恢复后才调用它，为时已晚）；
夹具为 `tests/fixtures/workflow_1.2.0_model_inspector.json`。Python 88 项、
黄金样本 3/3、Node 12 项均通过；剩余验收是在真实 ComfyUI 中载入夹具。

---

## 13. 测试与验收

### 13.1 自动化（AI 完成）

| 层 | 内容 |
|---|---|
| 结构解析 | 合成 LDM/diffusers 键 → 阶段/块/通道/注意力；真实 SD1.5、SDXL golden |
| 不变量 | 节点参数量之和 + unattributed == UNET 参数量；每键只归属一个节点 |
| 报告契约 | `schema_version`、字段类型、summary/keys 差异、路径不泄露 |
| 路由 | 非法 `dir`/`model`、路径遍历、`keys=0/1`、错误码 |
| 缓存 | 命中/失效（mtime/size 改变） |
| 前端纯函数 | 分组/搜索/排序/CSV/Markdown 转义，`node --test` |
| 语法 | `node --check`、`py_compile` |
| golden | 报告 JSON 规范化哈希；重构期间旧文本/图像 golden 作为过渡 |
| 打包 | 依赖移除后 `requirements.txt` 与 `test_packaging.py` 一致 |

### 13.2 用户验收（AI 不能代替）

1. 重启 ComfyUI、硬刷新；节点只剩两个选择项和一个按钮。
2. 选 SD1.5 模型 → 打开报告 → 架构页显示 4 个 down 阶段、3 个 downsample、12 个 input/output block。
3. 换 SDXL 模型 → 显示 3 个 down 阶段、2 个 downsample、`label_emb`（尺寸条件）被识别。
4. 点任一 ResBlock → 详情里能看到真实键名与参数量；点「在键名中查看」能过滤。
5. 原理页四张图显示正常；引用可点击；灰显模块与本模型一致。
6. 导出 JSON/CSV/Markdown/SVG 各一次，文件可正常打开。
7. 旧工作流（1.2.0 与 1.3.0 各一份）载入：控件不串位；2.0.0 后无输出节点，确认断开提示符合预期。
8. `.ckpt` 模型首次打开与二次刷新耗时对比（缓存生效）。
9. 安全回归：故意构造含 `<script>` 的元信息样本，页面只显示文本。

---

## 14. 实施计划（PR 拆分）

> 前置：PR #18（`model_dir`）先合并，否则 P1 从该分支叠加开发、合并后再 rebase 到 dev。
> P0 代码与自动化测试已完成（2026-10-24），等待用 1.2.0 工作流做界面验收。

| 阶段 | 内容 | 主要文件 | 规模 | 完成标准 |
|---|---|---|---|---|
| **P0** | 本文档评审 + 修 #18 控件迁移 + 去掉「全部」/英文化 | `docs/design/*`、`js/*`、Python 节点与测试 | 小 | 你用 1.2.0 工作流验证不错位、目录严格、界面英文 |
| **P1** | 数据层：`build_report` + structure/unet + 契约测试 | `model_toolkit/report.py`、`model_toolkit/structure/*`、`tests/*` | ~450 行 | 旧文本/图像 golden 不变；结构不变量测试通过 |
| **P2** | 服务层 + 页面骨架 + 节点按钮 | `model_toolkit/server.py`、`__init__.py`、`js/report_button.js`、`web/report/{index.html,report.js,report.css,report_logic.js}`、ECharts vendor | ~550 行 | 页面能打开；概览/键名/精度可用；旧输出仍在 |
| **P3** | UNet 特化视图 + 原理页 | `web/report/views/unet.js`、`web/report/principles/*`、报告补充 `capabilities` | ~700 行 | SD1.5/SDXL 结构图与权重一致；原理图与引用齐全 |
| **P4** | 2.0.0 破坏性发布：移除输出/图表/依赖 | `inspector.py`、`__init__.py`、删除 `charts.py`、`tests/test_charts.py`、`requirements.txt`、README/CHANGELOG | ~150 行改动 + 删除 | 节点零输出；依赖精简；契约测试更新；用户验收通过 |
| **P5**（可选） | DiT 视图（Flux/SD3）+ LLM 基础 | `structure/mmdit.py`、`views/dit.js`、原理图 | 后续 | 单独排期 |

依赖关系：P0 → P1 → P2 → P3 → P4；P5 独立。

> P3 交付（2026-10-24）：`web/report/views/unet.js`（自绘 U 形视图：阶段/块、跳连、
> 文本与时间条件、相对与绝对分辨率、块详情与「View in keys」前缀过滤、缩放/导出/
> 开关/基础分辨率、窄屏折叠）与 `web/report/principles/principles.js`（5 张自绘原理
> 示意图 + 论文引用，数值与当前报告联动）。旧文字/图片输出仍保留，留到 P4。

每个 PR 的目标分支均为 `dev`，按项目流程 PR + 单测 + 用户界面验收。

---

## 15. 风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 键名命名变体多 | 结构解析漏块/错块 | LDM + diffusers 双解析；模板辅助；真实模型 golden；`unattributed` 桶暴露漏网 |
| 注意力头数不可知 | 图上标注缺失 | 明示「未保存」，不猜 |
| 删输出导致旧工作流断裂 | 用户升级受挫 | 分阶段；CHANGELOG 醒目；页面 URL 可独立使用 |
| 控件位置迁移 | 静默错位 | P0 修复 + 旧工作流回归 |
| 前端代码量约 1,500+ 行 | 维护成本 | 纯函数/DOM 分离；无框架无构建；分 PR |
| SVG 原理图工作量大 | 延期 | 先做 4 张核心图；其余渐进补充；数据链接优先 |
| 原理图版权 | 法律风险 | 全部原创；只参考表达方式；引用论文 |
| `.ckpt` 阻塞 | Web 卡顿 | `to_thread` + LRU |
| XSS | 可调用同源 API | textContent + CSP + 注入测试 |
| ECharts 体积 | 仓库变大 | 只影响报告页；本地 vendor + NOTICE |

---

## 16. 默认决策（无异议即执行）

| # | 默认 | 说明 |
|---|---|---|
| 1 | 分阶段移除输出（P4 才删） | 避免功能真空；页面验收后再 2.0.0 |
| 2 | 删除 `show_details`、`show_charts` | 页面内有视图开关，避免死控件 |
| 3 | 文字报告改为页面「导出 Markdown」 | 前端生成；不再有 Python 文本格式化器 |
| 4 | UNet 图用 U 形布局 | 与扩散模型惯例一致；移动端可折叠 |
| 5 | 分辨率默认相对，绝对值为页面可选项 | 绝对分辨率受推理设置影响，不是权重事实 |
| 6 | 原理图全部原创 SVG | 不复制 PicDoc/论文图；引用论文 |
| 7 | 版本 2.0.0 | 移除输出属于破坏性变更 |
| 8 | 先在 #18 修控件迁移 | 防止 1.3.0 静默错位 |
| 9 | 去掉「（全部）」，默认 `checkpoints` | 目录与模型严格一致，避免跨目录同名文件误选 |
| 10 | 界面与输出先英文，i18n 后置 | 先把专业内容做对；翻译层后续统一引入 |

---

## 17. 术语与学习点

| 术语 | 人话 | 在本项目的位置 |
|---|---|---|
| 结构提取 | 从一堆权重键名反推出「模型长什么样」 | `structure/unet.py` |
| 特化视图 | 不同架构用不同的专业画法，而不是一张通用表格 | `views/unet.js` |
| 数据来源标注 | 告诉用户这个数字是权重算的、约定推的，还是不知道 | 第 6.4 节 `source` |
| 不变量测试 | 用一条永恒成立的等式守住数据一致性 | 图参数量之和 == 组件参数量 |
| 原创原理图 | 只参考表达方式，自己画，避免版权与错字 | `principles/*.svg` |
| 分阶段破坏性变更 | 先给替代品，验收后再拆旧的 | P1–P3 保留旧输出，P4 删除 |

**这次计划的三个关键判断：**

1. 先数据、再页面、最后删旧输出——每一步都可回退、可验收。
2. 架构图必须数据驱动：图上每个块都能点回真实键名与参数量，这是与"AI 生成示意图"的根本区别。
3. 专业性来自"知道什么不知道"：头数、绝对分辨率等权重里没有的信息，明确标注来源与不确定性。
