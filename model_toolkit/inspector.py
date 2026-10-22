"""模型检查器节点：编排分析流程与报告输出。"""
import logging
import os
from collections import defaultdict

import folder_paths
import torch

from .analysis import AnalysisMixin
from . import architectures
from .charts import ChartsMixin
from .precision import PrecisionMixin
from .core import ModelUtils

logger = logging.getLogger(__name__)


class CheckpointInspector(AnalysisMixin, PrecisionMixin, ChartsMixin):
    """提供完整的模型分析功能，包括：
    - 模型架构检测 (SD1.x/SD2.x/SDXL)
    - 参数统计和精度分析
    - 可视化报告生成
    - 组件结构分析
    """
    CATEGORY = "model_toolkit/分析"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "show_details": (["基础", "高级"], {"default": "基础"}),
                "show_charts": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("分析报告", "可视化图表")
    FUNCTION = "analyze_model"

    @classmethod
    def IS_CHANGED(cls, model_name, show_details, show_charts):
        """模型文件内容变化时让 ComfyUI 重跑（mtime+大小指纹）。

        节点只在文件名相同的输入下才会命中缓存，而文件名不变、内容被替换
        （重新下载 / 覆盖同名模型）时旧实现会一直返回旧报告（T10）。
        """
        return ModelUtils.model_fingerprint(folder_paths.get_full_path("checkpoints", model_name))

    def analyze_model(self, model_name, show_details, show_charts):
        """分析模型结构(主方法)"""
        try:
            model_path = folder_paths.get_full_path("checkpoints", model_name)
            # 只读头部即可完成分析（P4）：不把权重读进内存
            model_data = ModelUtils.load_model_index(model_path)

            if not model_data:
                return ("❌ 无法加载模型文件", torch.zeros((1, 1, 3)))

            result = []
            result.append(f"📄 模型文件: {model_name}")
            result.append(f"📊 模型大小: {ModelUtils.format_size(os.path.getsize(model_path))}")
            total_params = sum(t.numel() for t in model_data.values())
            
            precision_stats = self.analyze_model_precision(model_data)
            
            result.append(f"🔢 参数总量: {self.format_params(total_params)} [{precision_stats['primary_precision']}]")
            
            arch_result = self.analyze_architecture(model_data)
            if isinstance(arch_result, tuple):
                arch_info = arch_result[0]
            else:
                arch_info = arch_result
            
            result.append(f"\n🏗️ 模型架构: {arch_info['type']}")
            if arch_info.get('extra_info'):
                for info in arch_info['extra_info']:
                    result.append(f"  • {info}")

            # 架构能力提示：MMDiT/视频/音频架构下，三组件统计并不完整
            arch_hint = architectures.report_hint(architectures.identify(model_data.keys()))
            if arch_hint:
                result.append(arch_hint)
            
            groups = ModelUtils.group_model_keys(model_data)
            
            comp_params = defaultdict(lambda: (0, 0.0))
            for comp_name, keys in groups.items():
                params = sum(model_data[k].numel() for k in keys if k in model_data)
                comp_params[comp_name] = (params, params / total_params * 100)
            
            for k in ['UNET', 'CLIP', 'VAE']:
                if k not in comp_params:
                    comp_params[k] = (0, 0.0)
            
            # 计算其他参数
            other_params = comp_params.get('其他', (0, 0.0))[0]
            if other_params > 0:
                other_percent = other_params / total_params * 100
                comp_params['其他'] = (other_params, other_percent)
            
            # 计算EMA参数（model_ema.* 前缀/命名模式）
            ema_keys = []
            ema_patterns = ['model_ema', 'ema.', '.ema', '_ema', 'ema_', 'EMA']
            for pattern in ema_patterns:
                ema_keys.extend([k for k in model_data.keys() if pattern in k])

            # 移除重复项
            ema_keys = list(set(ema_keys))
            ema_params = sum(model_data[k].numel() for k in ema_keys if k in model_data)
            
            # 调整UNET参数，排除EMA部分
            if ema_params > 0:
                # 计算重叠部分：同时属于UNET和EMA的参数
                unet_keys = groups.get('UNET', [])
                overlap_keys = [k for k in ema_keys if k in unet_keys]
                overlap_params = sum(model_data[k].numel() for k in overlap_keys if k in model_data)
                
                # 从UNET中减去重叠部分
                if 'UNET' in comp_params:
                    unet_params, unet_percent = comp_params['UNET']
                    adj_unet_params = unet_params - overlap_params
                    adj_unet_percent = adj_unet_params / total_params * 100
                    comp_params['UNET'] = (adj_unet_params, adj_unet_percent)
                
                # 添加EMA作为独立组件
                comp_params['EMA'] = (ema_params, ema_params / total_params * 100)
                
                # 更新文本报告
                result.append(f"\n⚡ EMA参数:")
                result.append(f"- EMA参数量: {self.format_params(ema_params)} ({ema_params/total_params*100:.1f}%)")
                result.append(f"- EMA键数量: {len(ema_keys)}")
            
            # 获取未归类键名
            classified_keys = set()
            for group_name, keys in groups.items():
                classified_keys.update(keys)
            
            unknown_keys = [k for k in model_data.keys() if k not in classified_keys and k not in ema_keys]
            unknown_params = sum(model_data[k].numel() for k in unknown_keys if k in model_data)
            
            # 如果有未归类参数，添加到结果
            if unknown_params > 0:
                comp_params['未知'] = (unknown_params, unknown_params/total_params*100)
                result.append(f"\n🔍 未归类参数:")
                result.append(f"- 未归类参数量: {self.format_params(unknown_params)} ({unknown_params/total_params*100:.1f}%)")
                result.append(f"- 未归类键数量: {len(unknown_keys)}")
            
            # 如果有其他参数，添加到结果
            if other_params > 0:
                result.append(f"\n📦 其他参数:")
                result.append(f"- 其他参数量: {self.format_params(other_params)} ({other_percent:.1f}%)")
                result.append(f"- 其他键数量: {len(groups.get('其他', []))}")
            
            # 元数据提取：safetensors 读头部 __metadata__，其余格式回退到键扫描
            metadata = ModelUtils.read_safetensors_metadata(model_path)
            if not metadata:
                for k in model_data.keys():
                    if k.startswith('_metadata') or '_metadata' in k:
                        metadata[k] = model_data[k]
            
            if metadata:
                result.append(f"\n📌 模型元信息:")
                for key, value in metadata.items():
                    if hasattr(value, 'tolist'):
                        try:
                            value = value.tolist()
                        except:
                            value = str(value)
                    result.append(f"- {key}: {value}")
            else:
                result.append(f"\n📌 模型未包含元信息")

            # 详细分析（基础/高级模式）
            if show_details == "基础":
                # UNET分析
                if 'UNET' in groups and groups['UNET']:
                    unet_results = self.analyze_unet_structure(model_data, groups['UNET'], total_params)
                    result.extend(unet_results)
                
                # VAE分析
                if 'VAE' in groups and groups['VAE']:
                    vae_results = self.analyze_vae_structure(model_data, groups['VAE'], total_params)
                    result.extend(vae_results)
                
                # CLIP分析
                if 'CLIP' in groups and groups['CLIP']:
                    clip_results = self.analyze_clip_structure(model_data, groups['CLIP'], total_params, arch_info['type'])
                    result.extend(clip_results)

            elif show_details == "高级":
                result.append("\n🔍 模型所有键名:")
                all_keys = set(model_data.keys())
                
                for comp_name, keys in groups.items():
                    if keys:
                        result.append(f"\n=== {comp_name} 键名 ===")
                        for key in sorted(keys):
                            if key in model_data:
                                result.append(f"  {key}")
                                all_keys.discard(key)

                if all_keys:
                    result.append("\n=== 未分类键名 ===")
                    for key in sorted(all_keys):
                        result.append(f"  {key}")

            # 创建分析数据字典
            analysis_data = {
                'model_name': model_name,
                'model_size': ModelUtils.format_size(os.path.getsize(model_path)),
                'total_params': total_params,
                'total_params_formatted': self.format_params(total_params),
                'primary_precision': precision_stats['primary_precision'],
                'precision_distributions': precision_stats.get('distributions', {}),
                'components': {}, 
                'architecture': arch_info
            }
            
            # 填充组件数据
            for comp_name, (params, percentage) in comp_params.items():
                analysis_data['components'][comp_name] = params
            
            # UNET细分图数据计算
            unet_parts = {}
            if 'UNET' in groups and groups['UNET']:
                # 计算各块的参数量占比 - 排除EMA参数
                input_keys = [k for k in groups['UNET'] if any(p in k for p in ['input_blocks', 'down_blocks']) and k not in ema_keys]
                middle_keys = [k for k in groups['UNET'] if any(p in k for p in ['middle_block', 'mid_block', 'bottleneck']) and k not in ema_keys]
                output_keys = [k for k in groups['UNET'] if any(p in k for p in ['output_blocks', 'up_blocks']) and k not in ema_keys]
                
                input_params = sum(model_data[k].numel() for k in input_keys if k in model_data)
                middle_params = sum(model_data[k].numel() for k in middle_keys if k in model_data)
                output_params = sum(model_data[k].numel() for k in output_keys if k in model_data)
                
                total_unet_params = input_params + middle_params + output_params
                if total_unet_params > 0:
                    unet_parts = {
                        '输入块': input_params / total_unet_params,
                        '中间块': middle_params / total_unet_params,
                        '输出块': output_params / total_unet_params
                    }
                    analysis_data['unet_parts'] = unet_parts
            
            # 生成图表
            image_tensor = torch.zeros((1, 1, 3))
            if show_charts:
                try:
                    chart_data = self.generate_visual_report(analysis_data, model_name, arch_info['type'])
                    if chart_data:
                        image_tensor = self.convert_image(chart_data)
                except Exception as e:
                    logger.error(f"生成图表失败: {str(e)}")
            
            return ("\n".join(result), image_tensor)
            
        except Exception as e:
            logger.error(f"分析失败: {str(e)}")
            return (f"❌ 分析失败: {str(e)}", torch.zeros((1, 1, 3)))


    def update_progress(self, progress, desc):
        """更新进度提示"""
        if hasattr(self, "progress_bar"):
            self.progress_bar.update(progress, desc)
        else:
            logger.info(f"[进度 {progress}%] {desc}")

    def generate_text_report(self, analysis_result, ema_params=0, ema_keys=()):
        """生成文本格式的详细分析报告"""
        report = []
        total_params = analysis_result['total_params']
        # 基础信息
        report.append(f"📄 模型文件: {analysis_result['model_name']}")
        report.append(f"📊 模型大小: {analysis_result['model_size']}")
        report.append(f"🔢 参数总量: {analysis_result['total_params']}")
        
        # 架构信息
        report.append(f"\n🏗️ 模型架构: {analysis_result['architecture']['type']}")
        for info in analysis_result['architecture'].get('extra_info', []):
            report.append(f"  • {info}")
        
        # 组件分析
        for comp in ['UNET', 'VAE', 'CLIP']:
            if comp in analysis_result['components']:
                report.append(f"\n{comp}结构:")
                report.append(f"- 参数量: {analysis_result['components'][comp]['params']}")
                report.append(f"- 占比: {analysis_result['components'][comp]['percentage']}%")
        
        # 精度分析
        report.append("\n🔧 精度分析:")
        report.append(f"- 主要精度: {analysis_result['precision']['primary']}")
        for dist in analysis_result['precision']['distribution']:
            report.append(f"- {dist}: {analysis_result['precision']['distribution'][dist]}%")
        
        # 添加EMA参数统计到文本报告
        if ema_params > 0:
            report.append(f"\n⚡ EMA参数:")
            report.append(f"- 参数量: {self.format_params(ema_params)} ({ema_params/total_params*100:.1f}%)")
            report.append(f"- 键数量: {len(ema_keys)}")
        
        return "\n".join(report) 