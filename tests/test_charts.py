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


class ConvertImageTest(unittest.TestCase):
    """P3 之后 generate_visual_report 返回 PIL.Image，convert_image 需兼容旧 base64 入参。"""

    def _image(self):
        from PIL import Image

        return Image.new("RGBA", (640, 400), (30, 30, 30, 255))

    def test_accepts_pil_image(self):
        tensor = ChartsMixin().convert_image(self._image())

        self.assertEqual(tuple(tensor.shape), (1, 1280, 2048, 4))
        # 第 4 通道是 alpha（=1.0），颜色值看 RGB 通道
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 30 / 255, places=5)

    def test_accepts_legacy_data_url(self):
        import base64
        import io

        buf = io.BytesIO()
        self._image().save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        tensor = ChartsMixin().convert_image(data_url)

        self.assertEqual(tuple(tensor.shape), (1, 1280, 2048, 4))

    def test_none_returns_placeholder(self):
        tensor = ChartsMixin().convert_image(None)
        self.assertEqual(tuple(tensor.shape), (1, 1, 3))


if __name__ == "__main__":
    unittest.main()
