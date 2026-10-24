"""模型检查器节点：校验模型并预热报告缓存。

2.0.0 起节点不再输出文字/图片，报告页是唯一结果面；节点执行只做
路径解析、报告构建（预热缓存）与错误校验。
"""
import folder_paths

from .core import ModelUtils
from .folders import DEFAULT_DIR, list_model_dirs, resolve_model_path
from .report import build_report


class CheckpointInspector:
    CATEGORY = "model_toolkit/Analysis"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 顶部目录选择：models/ 下第一层子目录。模型列表由前端
                # js/model_dir_filter.js 按此值联动——INPUT_TYPES 只在
                # 启动时求值一次，服务端无法做控件间联动。
                "model_dir": (list_model_dirs(), {"default": DEFAULT_DIR}),
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "prepare_report"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, model_name, model_dir=DEFAULT_DIR):
        """模型文件内容变化时让 ComfyUI 重跑（mtime+大小指纹）。"""
        return ModelUtils.model_fingerprint(resolve_model_path(model_dir, model_name))

    def prepare_report(self, model_name, model_dir=DEFAULT_DIR):
        """校验模型并预热报告缓存；页面由节点按钮打开。"""
        build_report(model_dir, model_name)
        return {}
