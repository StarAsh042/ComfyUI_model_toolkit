# ComfyUI 模型工具包 (Model Toolkit)

[![CI](https://github.com/StarAsh042/ComfyUI_model_toolkit/actions/workflows/ci.yml/badge.svg?branch=dev)](https://github.com/StarAsh042/ComfyUI_model_toolkit/actions/workflows/ci.yml)

这是一个用于ComfyUI的模型工具包插件，提供了一套完整的模型分析和修改工具。
![交互式模型报告页（Architecture）](docs/images/report-overview.png)
## Todo List 📝


### 已完成 ✅
- [x] 实现基础模型分析功能，支持SD1.x/SD2.x/SDXL架构检测
- [x] 提供交互式模型报告页面：UNet 架构、原理解析、键名浏览、精度与元信息，浏览器本地渲染
- [x] 精确识别和统计EMA参数
- [x] 实现组件提取和替换功能
- [x] 添加未归类参数识别与统计（`Other` 参数）

### 待开发 🚀
- [ ] 添加对更多模型架构的支持（如Stable Cascade、SD3等）
- [ ] 优化内存使用，提高大型模型分析效率
- [ ] 添加LoRA和其他插件模型的支持
- [ ] 国际化支持：当前界面与输出为英文，后续增加 i18n 中文翻译
- [ ] 添加模型合并功能，支持多种合并算法
- [ ] 参数差异比较工具，支持同架构模型细粒度对比
- [ ] 增加参数剪枝功能，优化模型大小
- [ ] 添加模型量化工具，支持INT8/FP16转换
- [ ] DiT / LLM 架构特化视图（Flux、SD3 等）

## 功能

该插件提供了以下节点。当前节点界面、报告与图表均为英文，中文 i18n 翻译在后续版本提供。

### 🔍 Model Inspector

校验模型并预热报告缓存；分析结果在独立的交互式报告页面中查看。

- 输入：
  - `model_dir`：选择模型所在目录（`ComfyUI/models/` 下第一层子目录，如
    checkpoints / diffusion_models / loras…）。默认 `checkpoints`，没有
    「全部」选项；下方模型列表**严格只显示该目录内**的模型
  - `model_name`：选择要分析的模型文件
- 输出：**无**（2.0.0 起节点为零输出节点）
- 执行语义：解析模型路径、校验文件并预热报告缓存；结果在报告页查看
- 节点上的 `Open Interactive Report` 按钮会先重建报告，再在新标签页打开页面

报告页包含：

- Overview / Capabilities / Parameter reconciliation
- Architecture：自绘 SVG 的 UNet U 形结构图，支持阶段与块详情、跳连、
  文本/时间条件、缩放与导出 SVG、基础分辨率切换、宽屏模式
- Principles：5 张原创原理示意图与论文引用，数值与当前模型联动
- Keys：分页平面表，支持键名搜索、组件/精度/注意力过滤、排序与按 UNet 块前缀钻取；
  状态条显示当前视图与全部键的数量关系
- Precision / Metadata / Raw JSON
- 导出：JSON；`Export keys CSV · current view`（默认，包含搜索/组件/dtype/attention/
  块前缀与当前排序，跨分页取全集）；`Export keys CSV · all keys`（全部键，按当前排序）；
  Markdown（摘要，不含键名）

> Markdown 导出只包含概览、组件、精度与元信息，**不含键名章节**；1.x `Advanced`
> 报告里的全量键名清单由 `Export keys CSV · all keys` 承接。

> `.safetensors` 只读取文件头部（键名/形状/精度），**不加载权重**；`.ckpt/.pt` 因格式限制仍需全量加载。
> 模型文件被同名覆盖或重新下载后，节点会依据文件指纹自动重算，不会返回旧报告。
> 载入 1.2.0 / 1.3.0 工作流时会自动迁移控件位置（详见下方「升级到 2.0.0」）。

特点：
- 支持检测各种架构：SD1.x、SD2.x、SDXL及其变种
- 精确识别EMA参数和其他组件
- B/M单位的参数展示
- 交互式报告使用浏览器本地 ECharts 与自绘 SVG，不访问 CDN、不加载额外 Python 图表依赖

### 📤 Component Extractor

从模型中提取特定组件，产物写入 `ComfyUI/models/model_toolkit_extracted/`。

- 输入：
  - `model_name`：选择源模型文件
  - `component`：要提取的组件类型（UNET/VAE/CLIP）
  - `output_name`：输出文件名（可选）
- 输出：
  - `Extraction Result`：操作结果信息（含完整保存路径）
- 说明：提取产物只包含组件张量，不复制源模型 safetensors 的 `__metadata__`；
  输出文件不会带 metadata 头。

### 🔄 Component Replacer

用提取出的组件替换目标模型中的同名组件。

- 输入：
  - `model_name`：选择目标模型文件
  - `component`：要替换的组件类型（UNET/VAE/CLIP），决定哪些键允许被覆盖
  - `replacement_name`：从 `models/model_toolkit_extracted/` 选择替换文件（即组件提取器的产物）
  - `output_name`：输出文件名（可选）
- 输出：
  - `Replacement Result`：操作结果信息（替换 / 忽略键数）
- 说明：
  - **严格全量替换**：只有替换文件覆盖目标组件的全部键、且每个键形状一致时才保存。
    任一形状不匹配或目标键缺失都会在写文件前拒绝，不创建/覆盖输出文件或保存目录；
    错误信息使用固定模板并包含 `replaced/target_keys`、shape mismatch、uncovered 与
    missing 计数
  - 只替换属于所选 `component` 的键；替换文件里其它组件的额外键会被忽略并计数，
    不写入输出，也不影响目标模型的其它组件
  - 输出 metadata 只继承**目标模型** safetensors 头部的 `__metadata__`。键名转小写并
    去掉 `-`、`_`、`.` 后，只要包含以下任一子串（子串匹配，不是全匹配或通配符）即
    剔除：`hash`、`checksum`、`digest`、`sha1`、`sha2`、`sha224`、`sha256`、
    `sha384`、`sha512`、`sha3`、`shasum`、`md5`、`crc`、`blake`、`blake2`、
    `blake3`。例如 `sha2`、`sha-2`、`sha2-256`、`shasum`、`blake` 都会被剔除；
    这里刻意不使用裸 `sha` 作为 token，因此 `sha`、`shape`、`shard` 等安全键会保留，
    值不参与判断。替换源的 metadata 不复制，旧输出文件的 metadata 也不合并；
    目标无 metadata 或为 ckpt/pt 时输出不写 metadata 头
  - 替换源目录里也可以直接放完整 checkpoint，此时只会取其中的目标组件

## 支持的架构

内置 **10 份架构模板**（来自 sd-mecha，MIT）：

- **SD1.x**：`sd1-ldm`（ComfyUI 单文件）、`sd1-kohya`、`sd1-supermerger_blocks`
- **SDXL**：`sdxl-sgm`（ComfyUI 单文件）、`sdxl-kohya`、`sdxl-kohya_but_diffusers`、
  `sdxl-diffusers_unet_only`、`sdxl-supermerger_blocks`
- **Flux**：`flux-flux`；**SD3**：`sd3-comfyui`

其中 SD1.x / SDXL 的 6 份模板参与**自动识别**，需同时满足：必需模板键召回 ≥95%、
至少 32 个必需键命中、未忽略的额外键 ≤5%。命中后组件归属按模板成员关系精确判定
（SDXL 双文本编码器合并为 CLIP）。

`model_ema.*`、精确的训练/调度状态键、训练时写入的 baked embedding/LoRA 键，
以及与模板等价的裸 VAE 副本（`encoder.*` 等）有明确忽略规则，不计入额外键；
其他模板不认识的键仍计入额外键。报告字段 `template_coverage` 名称不变，
数值现在表示上述必需模板键召回率。

只含少量模板键的提取产物、UNet-only 的普通 SD1 裁切包，或缺少超过 5% 必需键的
裁切包可能不再匹配完整模板，会回退到现有架构家族/前缀规则；专用的
`sdxl-diffusers_unet_only` 模板仍支持。

Flux/SD3 模板含 T5XXL、supermerger 模板只有 BLOCKS，超出当前节点能表达的组件范围，
**数据已内置但暂不参与识别**。
未命中模板的模型（SD2.x 等）回退到键前缀规则。

模板细节与覆盖边界见 `components/architectures/README.md`；
可用 `python tools/validate_templates.py <模型路径>` 校验某个模型是否被模板识别。

### 架构识别与能力矩阵

报告会先识别架构家族，再判断"三组件统计"是否适用于它；不适用时直接给出英文提示
（`⚠️ Architecture note: …` / `⛔ Architecture note: …`），而不是给一个看起来正常、
实则漏算的数字：

| 家族 | 代表架构 | 报告提示 |
|---|---|---|
| SD 系 | SD1.x / SD2.x / SDXL | 无（统计完整） |
| MMDiT | Flux、SD3、AuraFlow、PixArt、HunyuanDiT | ⚠️ 含 T5，CLIP 统计不完整 |
| 级联 | Stable Cascade | ⚠️ 多阶段，单个 UNET ≠ 整模规模 |
| 视频 / 音视频 | SVD、LTXV、MiniMax H3 | ⛔ 3D VAE / 时序层，统计不适用 |

判据是"特征键是否存在"（参考 ComfyUI 0.42.4 的检测逻辑，GPL-3.0）。
其中 **SD 家族经过本机真实模型验证**，其余为文档级判据、尚未用真实模型验证。

## 支持的组件

节点按 **UNET / VAE / CLIP** 三类组件工作。模型通过上述自动识别门槛
（必需模板键召回 ≥95%、至少 32 个必需键命中、未忽略的额外键 ≤5%）时，
每个键的组件归属由模板成员关系直接判定；未通过的模型回退到键前缀规则。

## 依赖要求

`requirements.txt` 内容如下（本段与文件逐字一致）：

```
# torch 由 ComfyUI 自带，插件不声明：
# 声明会让 ComfyUI Manager 在安装时 pip 解析 torch，可能重装或降级宿主环境。
safetensors>=0.3.0
aiohttp>=3.9
```

- `safetensors`：只读模型头部（键名/形状/精度），并用于组件保存。
- `aiohttp`：**运行期依赖**，`model_toolkit/server.py` 用它注册 ComfyUI 的 HTTP 路由。
- `torch`：由 ComfyUI 宿主提供，插件不声明，避免 Manager 安装时重装/降级宿主 torch。
- `pyyaml`：只用于开发/CI 编译模板索引与测试解析，不是运行依赖。

2.0.0 起不再依赖 `numpy` / `matplotlib` / `Pillow`：图表与结构图由浏览器渲染。

## 开发与 CI

```bash
# Python 测试（真实 ComfyUI 环境；真实模型缺失时相关测试自动跳过）
python -m unittest discover -s tests -v

# 前端纯函数测试（Node 24）
node --test tests/frontend/*.test.mjs

# 静态检查
ruff check .

# 模板索引与 YAML 同步校验（CI 同样执行）
python tools/build_template_index.py --check
```

- **CI**（`.github/workflows/ci.yml`）：PR 与推送到 `main` / `dev` 时运行
  Python 3.11/3.12 + Windows 3.11、Node 24 与 Ruff；聚合 job `ci` 供分支保护引用。
  Runner 上没有 ComfyUI 与真实模型，测试通过 `tests/ci/folder_paths.py` 最小替身
  运行纯逻辑；需要真实模型的用例自动跳过（本机运行仍会执行）。
- **自动发布**（`.github/workflows/publish.yml`）：`pyproject.toml` 变更并推送到
  `main`（或手动触发）时发布到 Comfy Registry。需在仓库
  `Settings → Secrets and variables → Actions` 配置 `REGISTRY_ACCESS_TOKEN`。
- **发布基线守卫**：发布前只读查询 Comfy Registry 已发布版本，要求当前版本高于
  已发布最高版本；`2.0.0` 已发布后再次发布同一版本会以 `RIG-A VERSION_UNCHANGED`
  阻断。守卫不读 secret，也不以 Git tag 为基线。
- **版本一致性**：发布前同时更新 `pyproject.toml` 与 `__init__.py` 的
  `__version__`；`tests/test_packaging.py` 与发布工作流都会校验两者一致。
- 建议分支保护只要求聚合检查 `ci`，以后增删矩阵项无需改保护配置。

## 参与开发

> **给外部贡献者的重要说明**
>
> `main` 分支受保护，且仅维护者可直接推送。
> **请向 `dev` 分支提交 Pull Request；直接向 `main` 提交的 PR 大概率不会受理。**

推荐流程：

1. Fork 本仓库到你自己账号
2. 从 `dev` 分支切出特性分支（`git checkout -b feature/xxx`）
3. 提交改动并推送到你的 Fork
4. 向本仓库的 **`dev` 分支**开 Pull Request，等待 CI 通过与合并

分支流向：`feature/xxx` → `dev`（日常集成）→ `main`（发布版本）

## 安装方法

1. 将此仓库克隆或下载到ComfyUI的`custom_nodes`目录：
   ```bash
   cd custom_nodes
   git clone https://github.com/StarAsh042/ComfyUI_model_toolkit.git
   ```
2. 安装依赖（如果尚未安装）：
   ```bash
   pip install -r ComfyUI_model_toolkit/requirements.txt
   ```
3. 重启ComfyUI

## 使用示例

1. **分析模型架构**：
   - 添加 "🔍 Model Inspector" 节点
   - 选择模型目录（默认 `checkpoints`），再选择该目录内的模型文件
   - 点击节点上的 `Open Interactive Report`，报告会在新标签页打开
   - 也可以直接运行节点：校验模型并预热缓存，不会产生输出或图片

2. **提取模型组件**：
   - 添加 "📤 Component Extractor" 节点
   - 选择源模型文件
   - 选择要提取的组件（如VAE）
   - 设置输出文件名（可选）
   - 运行后会将组件保存到输出目录

3. **替换模型组件**：
   - 添加 "🔄 Component Replacer" 节点
   - 选择目标模型文件
   - 选择要替换的组件类型
   - 选择用于替换的组件文件（必须覆盖目标组件的全部键）
   - 设置输出文件名（可选）
   - 运行后会生成新的模型文件；不完整替换会被拒绝且不写盘

## 升级到 2.0.0

2.0.0 是破坏性版本，升级前请阅读：

- **节点不再输出文字与图片**：`Analysis Report` / `Visualization Chart` 输出已移除，
  旧的 `ShowText` / `SaveImage` 等下游节点会断开。需要重新添加
  Model Inspector 节点（或删除悬空连线），分析结果改在报告页查看。
- **两个控件被移除**：`show_details`、`show_charts` 删除；页面内已有
  架构视图、参数/占比/结构开关与 Markdown 导出。
- **旧工作流自动迁移**：1.2.0 的三值数组会保留 `model_name` 并补
  `model_dir=checkpoints`；1.3.0 的四值数组保留 `model_dir`、`model_name`；
  `widgets_values_named` 中的旧键会被删除。1.3.0 的「（全部）」仍映射为
  `checkpoints`。
- **依赖精简**：`numpy`、`matplotlib`、`Pillow` 不再声明，图表全部由浏览器渲染。

## 注意事项

- 支持`.safetensors`和`.ckpt`格式的模型文件
- 替换组件要求目标组件键 100% 覆盖且形状一致；不完整替换会在写盘前被拒绝，
  不再生成部分替换的模型
- 建议使用`.safetensors`格式保存输出文件；只有 safetensors 目标会保留 metadata
- 处理大型模型文件时需要足够的内存
- 所有操作都是只读的，不会修改原始文件
- 报告页需要由 ComfyUI 提供路由与静态资源；节点按钮会先重建报告再打开页面

## 故障排除

- 如果报告按钮不可用，请确认 ComfyUI 已重启且插件路由注册成功（查看启动日志）
- 如果报告页打不开，请检查 `/model_toolkit/report` 路由是否可访问
- 对于大型SDXL模型，首次分析可能需要较长时间和更多内存

## 许可

本项目以 **GNU General Public License v3.0** 发布，全文见 [LICENSE](LICENSE)。
Copyright (C) 2026 StarAsh042

第三方数据：

- `components/architectures/*.yaml` 来自 [sd-mecha](https://github.com/ljleb/sd-mecha)（MIT），
  原始许可见该目录下的 `LICENSE-sd-mecha`。MIT 与 GPL-3.0 兼容，可并入本项目一并分发。
