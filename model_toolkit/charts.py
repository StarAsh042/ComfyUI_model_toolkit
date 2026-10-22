"""检查器的可视化渲染（matplotlib / PIL）。"""
import base64
import io
import logging
import math

import matplotlib
matplotlib.use('Agg')  # 解决线程警告
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# 图表专用 rc 设置：只在渲染期间应用，不写入全局 matplotlib 状态
CHART_RC = {
    'font.sans-serif': ['SimHei', 'Microsoft YaHei'],  # 优先使用系统字体
    'axes.unicode_minus': False,
}

# 图表输出尺寸与背景色（与 plt.savefig(facecolor=...) 时期保持一致）
IMAGE_SIZE = (2048, 1280)
FIGURE_BACKGROUND = '#1E1E1E'

logger = logging.getLogger(__package__.split(".")[0])


class ChartsMixin:
    def generate_visual_report(self, analysis_data, model_name, arch_type):
        """生成可视化报告。

        图表样式只在本方法内生效：进入前快照全局 rcParams，退出时恢复，
        避免 dark_background 与中文字体设置污染同进程的其它节点（技术债）。
        """
        previous_rc = plt.rcParams.copy()
        try:
            return self._render_visual_report(analysis_data, model_name, arch_type)
        finally:
            plt.rcParams.update(previous_rc)

    def _render_visual_report(self, analysis_data, model_name, arch_type):
        """实际渲染；全局 rcParams 由 generate_visual_report 负责恢复。"""
        try:
            # 设置图表风格（仅本次渲染）
            plt.style.use('dark_background')
            plt.rcParams.update(CHART_RC)
            # 使用稍大一些的画布
            fig = plt.figure(figsize=(20, 12), dpi=100)
            
            # 添加标题
            plt.suptitle(
                f"{model_name} 结构分析 - {arch_type}\n"
                f"总参数量: {analysis_data['total_params_formatted']} | "
                f"主要精度: {analysis_data['primary_precision']}",
                color='white', fontsize=20, y=0.98
            )
            
            # 使用3x3网格布局
            grid = plt.GridSpec(3, 3, hspace=0.4, wspace=0.3)
            
            # 左上：组件比例圆环图
            ax_pie = plt.subplot(grid[0:2, 0])
            self.draw_component_pie(ax_pie, analysis_data)
            
            # 右侧：详细参数条形图
            ax_bars = plt.subplot(grid[:, 1:3])
            self.draw_param_bars(ax_bars, analysis_data, arch_type)
            
            # 左下：UNET结构
            ax_unet = plt.subplot(grid[2, 0])
            self.draw_unet_structure(ax_unet, analysis_data)
            
            # 添加水印
            plt.figtext(0.02, 0.01, "由 ComfyUI Model Toolkit 生成", color='gray', alpha=0.5)
            
            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # 直接从 Agg 画布取像素：跳过 PNG 编解码与 base64 往返（P3）
            fig.patch.set_facecolor(FIGURE_BACKGROUND)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            plt.close(fig)

            # 缩放到固定输出尺寸
            return Image.fromarray(rgba).resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        except Exception as e:
            logger.error(f"生成图表出错: {str(e)}")
            return None

    def draw_component_pie(self, ax, data):
        """绘制组件构成比例饼图"""
        components = data.get('components', {})
        if not components:
            ax.text(0.5, 0.5, '无组件数据', ha='center', va='center', color='white')
            return
        
        # 获取组件参数量和比例，过滤掉未知和占比很小的组件
        names = []
        values = []
        explode = []
        
        # 基础组件优先显示
        base_components = ['UNET', 'VAE', 'CLIP', 'EMA']
        for comp in base_components:
            if comp in components and components[comp] > 0:
                names.append(comp)
                values.append(components[comp])
                explode.append(0.05 if comp == 'UNET' else 0)
        
        # 添加其他组件
        for comp, value in components.items():
            if comp not in base_components and value > 0:
                if comp == '未知' and value/sum(components.values()) < 0.01:
                    continue  # 跳过占比极小的未知组件
                names.append(comp)
                values.append(value)
                explode.append(0)
        
        # 颜色映射
        colors = {
            'UNET': '#4C72B0', 
            'VAE': '#55A868', 
            'CLIP': '#C44E52', 
            'EMA': '#DD8452',
            '未知': '#937860',
            '其他': '#8172B0'
        }
        pie_colors = [colors.get(name, '#cccccc') for name in names]
        
        # 使用环形图替代饼图，并移除内部标签
        wedges, texts = ax.pie(
            values, 
            autopct=None,  # 移除内部百分比标签
            explode=explode, 
            startangle=90, 
            colors=pie_colors,
            wedgeprops={'width': 0.5, 'edgecolor': '#1E1E1E', 'linewidth': 0.5}
        )
        
        # 添加图例到左侧
        ax.legend(
            wedges, 
            [f"{name} ({self.format_params(val)}, {val/sum(values)*100:.1f}%)" for name, val in zip(names, values)],
            loc="center right", 
            bbox_to_anchor=(0, 0.5),  # 图例位置改到左侧
            fontsize=9,
            framealpha=0.5,
            labelcolor='white'
        )
        
        ax.set_title("组件参数分布", color='white', fontsize=14)

    def draw_unet_structure(self, ax, data):
        """修改后的UNET结构图使用B单位坐标系"""
        unet_parts = data.get('unet_parts', {})
        if not unet_parts:
            ax.text(0.5, 0.5, '无UNET结构数据', ha='center', va='center', color='white')
            return
        
        # 计算绝对参数量
        total_unet_params = data['components'].get('UNET', 0)
        abs_parts = {}
        for part, ratio in unet_parts.items():
            abs_parts[part] = ratio * total_unet_params
        
        # 排序并创建标签
        labels = []
        values = []
        for part, params in sorted(abs_parts.items(), key=lambda x: -x[1]):
            labels.append(part)
            values.append(params)
        
        # 创建水平条形图
        bars = ax.barh(labels, values, color=['#2E5D87', '#3B8EA5', '#4BB3D3'])
        
        # 添加参数量标签 - 使用动态单位（保持不变）
        ax.bar_label(bars, [self.format_params(v) for v in values], 
                    padding=3, color='white', fontsize=10)
        
        # 设置X轴为参数量（使用0.5B为最小间隔单位）
        max_value = max(values) * 1.1
        max_value_in_b = max_value / 1e9
        
        # 确定最大刻度值（向上取整到0.5B的倍数）
        max_tick_value_in_b = math.ceil(max_value_in_b * 2) / 2
        
        # 生成0.5B间隔的刻度
        tick_values_in_b = [i * 0.5 for i in range(int(max_tick_value_in_b * 2) + 1)]
        x_ticks = [v * 1e9 for v in tick_values_in_b]
        
        ax.set_xticks(x_ticks)
        
        # 使用B单位坐标轴刻度(0.5B, 1B, 1.5B等)
        x_tick_labels = []
        for value_in_b in tick_values_in_b:
            if value_in_b == int(value_in_b):
                # 整数B值
                x_tick_labels.append(f"{int(value_in_b)}B")
            else:
                # 0.5B的倍数
                x_tick_labels.append(f"{value_in_b:.1f}B")
        
        ax.set_xticklabels(x_tick_labels, color='white')
        ax.set_xlim(0, max_value)
        
        ax.set_title("UNET内部结构", color='white', fontsize=14)

    def draw_param_bars(self, ax, data, arch_type):
        """修改后的柱状图使用整数单位"""
        # 基础组件 + 更多可能的组件
        components = ['UNET', 'VAE', 'CLIP']
        
        # 如果存在EMA，添加到组件列表
        if data['components'].get('EMA', 0) > 0:
            components.append('EMA')
        
        # 添加未知分类
        if data['components'].get('未知', 0) > 0:
            components.append('未知')
        
        # 添加其他分类
        if data['components'].get('其他', 0) > 0:
            components.append('其他')
        
        params = [data['components'].get(c, 0) for c in components]
        
        # 过滤掉数值为0的组件
        valid_indices = [i for i, p in enumerate(params) if p > 0]
        valid_components = [components[i] for i in valid_indices]
        valid_params = [params[i] for i in valid_indices]
        
        # 颜色映射
        colors = {
            'UNET': '#4C72B0', 
            'VAE': '#55A868', 
            'CLIP': '#C44E52', 
            'EMA': '#DD8452',
            '未知': '#937860',
            '其他': '#8172B0'
        }
        valid_colors = [colors.get(comp, '#cccccc') for comp in valid_components]
        
        # 创建简洁标签
        bars = ax.bar(valid_components, valid_params, color=valid_colors)
        
        # 添加参数数值标签 - 使用动态单位（保持不变）
        ax.bar_label(bars, [self.format_params(p) for p in valid_params], 
                    color='white', fontsize=10)
        
        # 设置Y轴为参数量（使用0.5B为最小间隔单位）
        max_value = max(valid_params) * 1.1 if valid_params else 1
        max_value_in_b = max_value / 1e9
        
        # 确定最大刻度值（向上取整到0.5B的倍数）
        max_tick_value_in_b = math.ceil(max_value_in_b * 2) / 2
        
        # 生成0.5B间隔的刻度
        tick_values_in_b = [i * 0.5 for i in range(int(max_tick_value_in_b * 2) + 1)]
        y_ticks = [v * 1e9 for v in tick_values_in_b]
        
        ax.set_yticks(y_ticks)
        
        # 使用B单位坐标轴刻度(0.5B, 1B, 1.5B等)
        y_tick_labels = []
        for value_in_b in tick_values_in_b:
            if value_in_b == int(value_in_b):
                # 整数B值
                y_tick_labels.append(f"{int(value_in_b)}B")
            else:
                # 0.5B的倍数
                y_tick_labels.append(f"{value_in_b:.1f}B")
        
        ax.set_yticklabels(y_tick_labels, color='white')
        ax.set_ylim(0, max_value)
        
        # 美化图表
        ax.set_ylabel("参数量", color='white')
        ax.set_title(f"{arch_type} 模型组件参数分布", color='white', fontsize=16)
        ax.tick_params(axis='x', labelcolor='white', labelrotation=0)
        ax.grid(True, axis='y', alpha=0.3)

    def convert_image(self, image):
        """把渲染结果转成 ComfyUI 的 IMAGE 张量。

        支持 PIL.Image（P3 之后的正常路径）与旧的 base64 data URL（兼容旧调用）。
        """
        try:
            if image is None:
                return torch.zeros((1, 1, 3))

            if isinstance(image, str):
                data = image.split(",", 1)[-1]
                image = Image.open(io.BytesIO(base64.b64decode(data)))

            if tuple(image.size) != IMAGE_SIZE:
                image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

            return torch.from_numpy(np.array(image)).float().div_(255.0).unsqueeze(0)
        except Exception as e:
            logger.error(f"图像转换失败: {str(e)}")
            return torch.zeros((1, 512, 512, 3))
