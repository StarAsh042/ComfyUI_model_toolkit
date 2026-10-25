"""组件提取器 / 替换器的输出路径安全校验。

两个节点共用同一套规则，避免一边加固、另一边仍可越界：
- 显式 output_name 只能是保存目录下的单层文件名。绝对路径
  （盘符、UNC、根路径）、``/`` 与 ``\\`` 分隔符（含 ``../``、``..\\``
  穿越与子目录）、``.`` / ``..``、空字节都会被拒绝，不做静默清洗。
- 空 output_name 沿用原有默认命名。默认名由模型名派生，可能自带子目录，
  因此只做最终 realpath 包含性校验，既不改变命名，也不允许写出保存目录。
- 校验先于任何保存调用；越界时抛出 :class:`UnsafeOutputPathError`。
"""
import os

# 可识别的模型扩展名（比较时统一转小写）：已有扩展名不再重复追加
MODEL_EXTENSIONS = (".safetensors", ".sft", ".ckpt", ".pt")


class UnsafeOutputPathError(ValueError):
    """output_name 无法安全地落在保存目录内。"""


def _reject(reason):
    raise UnsafeOutputPathError(reason)


def validate_output_name(output_name):
    """显式 output_name 必须是保存目录下的单层文件名。"""
    if not isinstance(output_name, str) or not output_name:
        _reject("output_name must be a non-empty file name")
    if "\x00" in output_name:
        _reject("output_name must not contain a null byte")
    if os.path.isabs(output_name):
        _reject("output_name must not be an absolute path")
    if os.path.splitdrive(output_name)[0]:
        _reject("output_name must not include a drive")
    if os.name == "nt" and ":" in output_name:
        # Windows 下冒号只能用于盘符/备用数据流，不是合法普通文件名
        _reject("output_name must not contain ':'")
    if "/" in output_name or "\\" in output_name:
        _reject("output_name must not contain path separators or subdirectories")
    if output_name in (".", ".."):
        _reject("output_name must not be a directory reference")


def _validate_default_name(default_name):
    """默认名可以含子目录，但拒绝空值、空字节、绝对路径与盘符。"""
    if not isinstance(default_name, str) or not default_name:
        _reject("default output name is not a valid file name")
    if "\x00" in default_name:
        _reject("default output name must not contain a null byte")
    if os.path.isabs(default_name) or os.path.splitdrive(default_name)[0]:
        _reject("default output name must not be an absolute path")
    if os.name == "nt" and ":" in default_name:
        _reject("default output name must not contain ':'")


def _ensure_contained(save_dir, file_name):
    """返回拼接路径；realpath 越出 save_dir 时拒绝。"""
    if not isinstance(save_dir, (str, os.PathLike)) or not os.fspath(save_dir):
        _reject("save directory is not configured")

    candidate = os.path.join(save_dir, file_name)
    base = os.path.normcase(os.path.realpath(save_dir))
    target = os.path.normcase(os.path.realpath(candidate))
    try:
        contained = os.path.commonpath([base, target]) == base
    except ValueError:  # 不同盘符
        contained = False
    if not contained or target == base:
        _reject("output_name resolves outside the save directory")
    return candidate


def resolve_output_path(save_dir, output_name, default_name):
    """校验并返回组件保存路径。

    output_name 非空时按用户输入严格校验；为空时使用 default_name
    （沿用原默认命名）。扩展名规则与旧实现一致：缺少模型扩展名时补
    ``.safetensors``；已有扩展名的判断大小写不敏感，且保留用户输入的
    原始大小写（``foo.SFT`` 不会变成 ``foo.SFT.safetensors``）。
    任何越界输入都在调用保存前抛出异常。
    """
    if output_name:
        validate_output_name(output_name)
        file_name = output_name
    else:
        _validate_default_name(default_name)
        file_name = default_name

    if not file_name.lower().endswith(MODEL_EXTENSIONS):
        file_name += ".safetensors"
    return _ensure_contained(save_dir, file_name)
