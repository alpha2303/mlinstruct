from typing import Any

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve the torch.device to train on.

    Args:
        device (Optional[Union[str, torch.device]]): An explicit device. Takes
            precedence over accelerator detection when given.

    Returns:
        torch.device: The explicit device if given, else the current
            accelerator if one is available, else CPU.
    """
    if device is not None:
        return torch.device(device)

    accelerator = torch.accelerator.current_accelerator()
    if accelerator is not None:
        return torch.device(accelerator)

    return torch.device("cpu")


def move_to_device(obj: Any, device: torch.device, non_blocking: bool = True) -> Any:
    """Recursively move any Tensors nested within obj to device.

    Args:
        obj (Any): A Tensor, or a tuple/list/dict possibly nesting Tensors.
            Anything else is returned unchanged.
        device (torch.device): The target device.
        non_blocking (bool): Passed through to Tensor.to.

    Returns:
        Any: The same structure as obj, with contained Tensors moved to device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, tuple):
        return tuple(move_to_device(item, device, non_blocking) for item in obj)
    if isinstance(obj, list):
        return [move_to_device(item, device, non_blocking) for item in obj]
    if isinstance(obj, dict):
        return {key: move_to_device(value, device, non_blocking) for key, value in obj.items()}
    return obj
