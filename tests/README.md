# 测试与黄金样本

环境：ComfyUI 自带 Python（3.11.6）+ torch 2.14.0+cpu。
ComfyUI 根目录默认 `D:\AIGC\ComfyUI`，可用环境变量 `COMFYUI_DIR` 覆盖。

## 单元与契约测试（G1 / G2 / G6）

```powershell
cd D:\AIGC\ComfyUI\custom_nodes\ComfyUI_model_toolkit
& "D:\AIGC\ComfyUI\.ext\python.exe" -m unittest discover -s tests -v
```

## 前端纯函数测试（Node 24+）

```powershell
cd D:\AIGC\ComfyUI\custom_nodes\ComfyUI_model_toolkit
node --test tests/frontend/widget_migration.test.mjs
```

`tests/fixtures/workflow_1.2.0_model_inspector.json` 是 1.2.0 保存的工作流，
使用与黄金样本相同的本机模型，用于在 ComfyUI 中手动验证控件位置迁移：
加载后 `model_dir` 应为 `checkpoints`，`show_details` 应为 `Basic`，模型选择不变。

## 黄金样本（G9，需真实模型，约 15 秒）

```powershell
& "D:\AIGC\ComfyUI\.ext\python.exe" tests\golden_check.py
```

覆盖三个场景：`Basic` 报告、`Advanced` 报告 + 图表、替换器错误路径。
比对文本 SHA256、图像张量 shape/dtype/完整字节 SHA256。

## 约定

- `test_behavior.GroupModelKeysTest` 与 `RealCheckpointGroupingTest` 锁定 B1 修复后的分类：
  CLIP 文本编码器键必须归 CLIP；真实模型分组数为 UNET 686 / CLIP 197 / VAE 248 / EMA 2，
  未分类键归 `Other`。再改分类逻辑时，先改测试再改实现。
- 黄金样本基线于 2026-10-22 因 B1/B3/B4/B2 修复重建过；
  2026-10-24 因界面/输出英文化、默认目录改为 `checkpoints` 再次重建，
  文本与图表哈希变化均属预期（原因见 CHANGELOG）。
- `golden_check.py --write` 仅用于重建基线；重建前必须确认行为变化是预期的，并在提交信息里写明原因。
