"""Residual hooks shared by the held-out Step 3 protocols.

A direction estimated at image-placeholder tokens is distinct from the site
where that direction is injected. The main experiments inject at generated-text
states: the final attended prefill token and every cached decode token.
"""

from __future__ import annotations


def register_text_trajectory_hook(layer, attention, image_mask, direction, scale):
    """Register a tuple-preserving hook and return its prefill/decode counters."""
    import torch

    state = {"prefill": 0, "decode": 0}
    attention_cpu = attention.detach().cpu().bool()
    image_cpu = image_mask.detach().cpu().bool()
    direction_cpu = direction.detach().cpu().float()

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        patched = hidden.clone()
        addition = float(scale) * direction_cpu.to(hidden.device, hidden.dtype)
        if hidden.shape[:2] == attention_cpu.shape:
            if state["prefill"]:
                raise RuntimeError("Received more than one full prefill")
            mask = attention_cpu.to(hidden.device)
            positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0).expand_as(mask)
            last = positions.masked_fill(~mask, -1).amax(dim=1)
            if (last < 0).any():
                raise ValueError("Every batch row must have an attended prefill token")
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            if image_cpu.to(hidden.device)[rows, last].any():
                raise ValueError("The final prefill intervention site must be a text token")
            patched[rows, last] += addition
            state["prefill"] = 1
        elif hidden.shape[1] == 1:
            if not state["prefill"]:
                raise RuntimeError("Decode arrived before prefill")
            patched[:, 0] += addition
            state["decode"] += 1
        else:
            raise RuntimeError(f"Unexpected layer shape: {tuple(hidden.shape)}")
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    return layer.register_forward_hook(hook), state


def register_prefill_capture(layer, attention):
    """Retain only the full prefill residual, ignoring later cached decode states.

    The original behavior-check notebook overwrote this buffer at every decode
    step. That made its image-mask pooling broadcast a final decode vector; this
    helper intentionally corrects that bug and marks the rerun separately in its
    saved artifact.
    """
    captured = {}
    expected_shape = tuple(attention.shape)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if tuple(hidden.shape[:2]) == expected_shape:
            if "h" in captured:
                raise RuntimeError("Received more than one full prefill")
            captured["h"] = hidden.detach()
        return output

    return layer.register_forward_hook(hook), captured
