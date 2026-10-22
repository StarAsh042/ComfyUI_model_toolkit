# ComfyUI 模型工具包 (Model Toolkit)

这是一个用于ComfyUI的模型工具包插件，提供了一套完整的模型分析和修改工具。
![ComfyUI_model_toolkit](https://github.com/user-attachments/assets/c58d0731-56ef-46e8-a85a-ad1f92ad4b52)
## Todo List 📝


### 已完成 ✅
- [x] 实现基础模型分析功能，支持SD1.x/SD2.x/SDXL架构检测
- [x] 提供可视化模型结构图表，包括组件分布和参数统计
- [x] 精确识别和统计EMA参数
- [x] 实现组件提取和替换功能
- [x] 添加未归类参数识别与统计("其他"参数)

### 待开发 🚀
- [ ] 添加对更多模型架构的支持（如Stable Cascade、SD3等）
- [ ] 优化内存使用，提高大型模型分析效率
- [ ] 添加LoRA和其他插件模型的支持
- [ ] 国际化支持，提供英文界面
- [ ] 增加图表自定义主题
- [ ] 支持模型权重分层可视化（按Block层级展示参数分布）
- [ ] 实现交互式组件浏览器，支持展开/折叠不同层级
- [ ] 添加模型合并功能，支持多种合并算法
- [ ] 参数差异比较工具，支持同架构模型细粒度对比
- [ ] 增加参数剪枝功能，优化模型大小
- [ ] 增强UNET内部结构分析，支持注意力机制可视化
- [ ] 添加模型量化工具，支持INT8/FP16转换

## 功能

该插件提供了以下节点：

### 🔍 模型检查器 (ModelInspector)

分析模型文件并显示其架构和组件信息，提供可视化报告。

- 输入：
  - `model_name`：选择要分析的模型文件
  - `show_details`：选择"基础"或"高级"显示模式
  - `show_charts`：是否生成可视化图表
- 输出：
  - `分析报告`：包含模型架构、组件分布等详细信息
  - `可视化图表`：直观展示模型结构的图形化报表，包括参数分布、UNET结构等

> `.safetensors` 只读取文件头部（键名/形状/精度），**不加载权重**；`.ckpt/.pt` 因格式限制仍需全量加载。
> 模型文件被同名覆盖或重新下载后，节点会依据文件指纹自动重算，不会返回旧报告。

特点：
- 支持检测各种架构：SD1.x、SD2.x、SDXL及其变种
- 精确识别EMA参数和其他组件
- B/M单位的参数展示
- 美观的深色主题图表
- 精确的组件分类和分析

### 📤 组件提取器 (ComponentExtractor)

从模型中提取特定组件，产物写入 `ComfyUI/models/model_toolkit_extracted/`。

- 输入：
  - `model_name`：选择源模型文件
  - `component`：要提取的组件类型（UNET/VAE/CLIP）
  - `output_name`：输出文件名（可选）
- 输出：
  - `提取结果`：操作结果信息（含完整保存路径）

### 🔄 组件替换器 (ComponentReplacer)

用提取出的组件替换目标模型中的同名组件。

- 输入：
  - `model_name`：选择目标模型文件
  - `component`：要替换的组件类型（UNET/VAE/CLIP），决定哪些键允许被覆盖
  - `replacement_name`：从 `models/model_toolkit_extracted/` 选择替换文件（即组件提取器的产物）
  - `output_name`：输出文件名（可选）
- 输出：
  - `替换结果`：操作结果信息（替换 / 跳过键数）
- 说明：
  - 只替换属于所选 `component`、且键名与形状都匹配的键；其它组件的键不受影响
  - 替换源目录里也可以直接放完整 checkpoint，此时只会取其中的目标组件

### 🔬 UNET检查器 (UNetInspector)

专门分析模型中UNET组件的结构和参数分布。

- 输入：
  - `model_name`：选择要分析的模型文件
  - `show_details`：选择显示简略或详细信息
- 输出：
  - `分析结果`：UNET的详细结构信息

## 支持的架构

内置 **10 份架构模板**（来自 sd-mecha，MIT）：

- **SD1.x**：`sd1-ldm`（ComfyUI 单文件）、`sd1-kohya`、`sd1-supermerger_blocks`
- **SDXL**：`sdxl-sgm`（ComfyUI 单文件）、`sdxl-kohya`、`sdxl-kohya_but_diffusers`、
  `sdxl-diffusers_unet_only`、`sdxl-supermerger_blocks`
- **Flux**：`flux-flux`；**SD3**：`sd3-comfyui`

其中 SD1.x / SDXL 的 6 份模板参与**自动识别**：模型键被模板覆盖 ≥95% 时，
组件归属按模板成员关系精确判定（SDXL 双文本编码器合并为 CLIP）。
Flux/SD3 模板含 T5XXL、supermerger 模板只有 BLOCKS，超出当前节点能表达的组件范围，
**数据已内置但暂不参与识别**。
未命中模板的模型（SD2.x 等）回退到键前缀规则。

模板细节与覆盖边界见 `components/architectures/README.md`；
可用 `python tools/validate_templates.py <模型路径>` 校验某个模型是否被模板覆盖。

## 支持的组件

节点按 **UNET / VAE / CLIP** 三类组件工作。当模型被架构模板覆盖（覆盖率 ≥ 95%）时，
每个键的组件归属由模板成员关系直接判定；未覆盖的模型回退到键前缀规则。

## 依赖要求

该插件依赖以下Python库（torch 由 ComfyUI 自带，插件不声明）：
```
numpy>=1.23.0
matplotlib>=3.5.0
Pillow>=9.0.0
safetensors>=0.3.0
pyyaml>=5.4.0
```

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
   - 添加"🔍 模型检查器"节点
   - 选择要分析的模型文件
   - 选择显示详细程度
   - 启用图表显示选项
   - 运行后可查看模型的完整信息和可视化图表

2. **提取模型组件**：
   - 添加"📤 组件提取器"节点
   - 选择源模型文件
   - 选择要提取的组件（如VAE）
   - 设置输出文件名（可选）
   - 运行后会将组件保存到输出目录

3. **替换模型组件**：
   - 添加"🔄 组件替换器"节点
   - 选择目标模型文件
   - 选择要替换的组件类型
   - 选择用于替换的组件文件
   - 设置输出文件名（可选）
   - 运行后会生成新的模型文件

4. **分析UNET结构**：
   - 添加"🔬 UNET检查器"节点
   - 选择要分析的模型文件
   - 选择显示详细程度
   - 运行后可查看UNET的详细结构信息

## 注意事项

- 支持`.safetensors`和`.ckpt`格式的模型文件
- 替换组件时会自动验证兼容性
- 建议使用`.safetensors`格式保存输出文件
- 处理大型模型文件时需要足够的内存
- 所有操作都是只读的，不会修改原始文件
- 可视化图表需要matplotlib库支持

## 故障排除

- 如果图表无法正常显示，请确保已安装matplotlib和Pillow库
- 如果节点无法加载，请检查ComfyUI日志中的错误信息
- 对于大型SDXL模型，分析可能需要较长时间和更多内存

## 许可

本项目以 **GNU General Public License v3.0** 发布，全文见 [LICENSE](LICENSE)。
Copyright (C) 2026 StarAsh042

第三方数据：

- `components/architectures/*.yaml` 来自 [sd-mecha](https://github.com/ljleb/sd-mecha)（MIT），
  原始许可见该目录下的 `LICENSE-sd-mecha`。MIT 与 GPL-3.0 兼容，可并入本项目一并分发。
