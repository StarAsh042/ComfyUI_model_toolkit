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
  - `模型`：选择要分析的模型文件
  - `显示细节`：选择"基础"或"高级"显示模式
  - `显示图表`：是否生成可视化图表
- 输出：
  - `分析报告`：包含模型架构、组件分布等详细信息
  - `可视化图表`：直观展示模型结构的图形化报表，包括参数分布、UNET结构等

特点：
- 支持检测各种架构：SD1.x、SD2.x、SDXL及其变种
- 精确识别EMA参数和其他组件
- B/M单位的参数展示
- 美观的深色主题图表
- 精确的组件分类和分析

### 📤 组件提取器 (ComponentExtractor)

从模型中提取特定组件。

- 输入：
  - `model_name`：选择源模型文件
  - `component`：要提取的组件类型（UNET/VAE/CLIP）
  - `output_name`：输出文件名（可选）
- 输出：
  - `提取结果`：操作结果信息

### 🔄 组件替换器 (ComponentReplacer)

替换模型中的特定组件。

- 输入：
  - `model_name`：选择目标模型文件
  - `component`：要替换的组件类型（UNET/VAE/CLIP）
  - `replacement_name`：选择替换用的组件文件
  - `output_name`：输出文件名（可选）
- 输出：
  - `替换结果`：操作结果信息

### 🔬 UNET检查器 (UNetInspector)

专门分析模型中UNET组件的结构和参数分布。

- 输入：
  - `model_name`：选择要分析的模型文件
  - `show_details`：选择显示简略或详细信息
- 输出：
  - `分析结果`：UNET的详细结构信息

## 支持的架构

- SD1.x：Stable Diffusion 1.x系列模型
- SD2.x：Stable Diffusion 2.x系列模型
- SDXL：Stable Diffusion XL系列模型
- SD1.x-EMA：带有EMA权重的模型

## 支持的组件

- UNET：
  - UNET-v1-SD：SD 1.x的UNET
  - UNET-v2-SD：SD 2.x的UNET
  - UNET-XL-SD：SDXL的UNET
  - UNET-v1-EMA：带EMA的UNET
  - UNET-Flux-SD：Flux架构的UNET

- CLIP：
  - CLIP-v1-SD：SD 1.x的CLIP
  - CLIP-v2-SD：SD 2.x的CLIP
  - CLIP-XL-SD：SDXL的CLIP
  - CLIP-v2-WD：WD 1.4的CLIP

- VAE：
  - VAE-v1-SD：通用VAE组件

## 依赖要求

该插件依赖以下Python库：
```
torch>=1.12.0
numpy>=1.23.0
matplotlib>=3.5.0
Pillow>=9.0.0
safetensors>=0.3.0
```

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
