"""图表模块的进程级副作用测试（技术债：不得污染全局 matplotlib 状态）。"""
import unittest

import _env  # noqa: F401  设置 sys.path

import matplotlib.pyplot as plt

from ComfyUI_model_toolkit.model_toolkit.charts import ChartsMixin


class _Probe(ChartsMixin):
    """用桩方法验证 rcParams 快照/恢复机制，避免依赖真实模型。"""

    def _render_visual_report(self, analysis_data, model_name, arch_type):
        plt.style.use('dark_background')
        plt.rcParams['axes.unicode_minus'] = False
        return "rendered"


class ChartStyleIsolationTest(unittest.TestCase):
    def test_render_does_not_leak_global_rcparams(self):
        before = dict(plt.rcParams)
        result = _Probe().generate_visual_report({}, "probe.safetensors", "SD1.x")
        after = dict(plt.rcParams)

        self.assertEqual(result, "rendered")
        self.assertEqual(before, after, "渲染后全局 rcParams 必须恢复原值")

    def test_render_exception_also_restores(self):
        class _Boom(ChartsMixin):
            def _render_visual_report(self, *args):
                plt.style.use('dark_background')
                raise RuntimeError("boom")

        before = dict(plt.rcParams)
        with self.assertRaises(RuntimeError):
            _Boom().generate_visual_report({}, "probe.safetensors", "SD1.x")

        self.assertEqual(before, dict(plt.rcParams))


if __name__ == "__main__":
    unittest.main()
