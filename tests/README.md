# 测试与黄金样本

环境：ComfyUI 自带 Python（3.11.6）+ torch 2.14.0+cpu。
ComfyUI 根目录默认 `D:\AIGC\ComfyUI`，可用环境变量 `COMFYUI_DIR` 覆盖。

## 单元与契约测试（G1 / G2 / G6）

```powershell
cd D:\GitHub\ComfyUI_model_toolkit
& "D:\AIGC\ComfyUI\.ext\python.exe" -m unittest discover -s tests -v
```

## 黄金样本（G9，需真实模型，约 15 秒）

```powershell
& "D:\AIGC\ComfyUI\.ext\python.exe" tests\golden_check.py
```

覆盖四个场景：基础报告、高级报告 + 图表、UNET 简略、替换器错误路径。
比对文本 SHA256、图像张量 shape/dtype/完整字节 SHA256。

## 约定

- `test_behavior.test_group_model_keys_misclassifies_clip_encoder` 锁定的是已知缺陷
  （CODE-REVIEW B1：CLIP 文本编码器键落进 VAE 组）。重构期间不得改变；修 bug 时同步更新。
- `golden_check.py --write` 仅用于重建基线，重构期间禁止使用。
