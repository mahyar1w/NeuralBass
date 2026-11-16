#!/usr/bin/env python3
"""
Exports a PyTorch ResidualGRUBassNet (1 -> 48 GRU -> 1 Dense) model or state_dict
to RTNeural's JSON schema.

Includes both:
1. Native RTNeural `ModelT::parseJson` schema (`in_shape` + `layers` with transposed
   matrices and swapped [z, r, n] gates).
2. PyTorch `RTNeural::torch_helpers` keys (`gru.weight_ih_l0`, `gru.weight_hh_l0`,
   `gru.bias_ih_l0`, `gru.bias_hh_l0`, `dense.weight`, `dense.bias`).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class ResidualGRUBassNet(nn.Module):
        """
        Stateful GRU neural timbre transfer network:
        1 -> 48 GRU -> 1 Dense with residual skip connection y = x + net(x).
        Matches C++ compile-time type:
        RTNeural::ModelT<float, 1, 1, RTNeural::GRULayerT<float, 1, 48>, RTNeural::DenseT<float, 48, 1>>
        """
        def __init__(self, hidden_size: int = 48):
            super().__init__()
            self.hidden_size = hidden_size
            self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
            self.dense = nn.Linear(in_features=hidden_size, out_features=1, bias=True)

        def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
            x = x.contiguous()
            if h0 is not None:
                h0 = h0.contiguous()
            gru_out, hn = self.gru(x, h0)
            residual = 0.35 * torch.tanh(self.dense(gru_out))
            return x + residual, hn


def _swap_rz_gates_2d(matrix_in_by_3h: np.ndarray, hidden_size: int) -> np.ndarray:
    """
    Swaps PyTorch [r, z, n] gate order along axis 1 into RTNeural/Keras [z, r, n] order.
    matrix_in_by_3h has shape (rows, 3 * hidden_size).
    """
    r_gate = matrix_in_by_3h[:, 0:hidden_size].copy()
    z_gate = matrix_in_by_3h[:, hidden_size:2 * hidden_size].copy()
    n_gate = matrix_in_by_3h[:, 2 * hidden_size:3 * hidden_size].copy()
    return np.concatenate([z_gate, r_gate, n_gate], axis=1)


def _swap_rz_gates_1d(vec_3h: np.ndarray, hidden_size: int) -> np.ndarray:
    """
    Swaps PyTorch [r, z, n] gate order in a 1D bias vector of length 3 * hidden_size
    into RTNeural/Keras [z, r, n] order.
    """
    r_gate = vec_3h[0:hidden_size].copy()
    z_gate = vec_3h[hidden_size:2 * hidden_size].copy()
    n_gate = vec_3h[2 * hidden_size:3 * hidden_size].copy()
    return np.concatenate([z_gate, r_gate, n_gate], axis=0)


def build_rtneural_json_dict(
    w_ih: np.ndarray,
    w_hh: np.ndarray,
    b_ih: np.ndarray,
    b_hh: np.ndarray,
    dense_w: np.ndarray,
    dense_b: np.ndarray,
    hidden_size: int = 48,
) -> Dict[str, Any]:
    """
    Builds a complete RTNeural JSON dictionary from numpy arrays in PyTorch layout:
      w_ih:    (3*H, 1)
      w_hh:    (3*H, H)
      b_ih:    (3*H,)
      b_hh:    (3*H,)
      dense_w: (1, H)
      dense_b: (1,)
    """
    assert w_ih.shape == (3 * hidden_size, 1), f"Unexpected w_ih shape: {w_ih.shape}"
    assert w_hh.shape == (3 * hidden_size, hidden_size), f"Unexpected w_hh shape: {w_hh.shape}"
    assert b_ih.shape == (3 * hidden_size,), f"Unexpected b_ih shape: {b_ih.shape}"
    assert b_hh.shape == (3 * hidden_size,), f"Unexpected b_hh shape: {b_hh.shape}"
    assert dense_w.shape == (1, hidden_size), f"Unexpected dense_w shape: {dense_w.shape}"
    assert dense_b.shape == (1,), f"Unexpected dense_b shape: {dense_b.shape}"

    # Convert PyTorch (3H, In) [r, z, n] -> RTNeural (In, 3H) [z, r, n]
    kernel_weights = _swap_rz_gates_2d(w_ih.T, hidden_size)
    recurrent_weights = _swap_rz_gates_2d(w_hh.T, hidden_size)
    bias_ih_rt = _swap_rz_gates_1d(b_ih, hidden_size)
    bias_hh_rt = _swap_rz_gates_1d(b_hh, hidden_size)
    gru_biases = np.stack([bias_ih_rt, bias_hh_rt], axis=0)

    # Dense weights in RTNeural parseJson format have shape (in_features, out_features) = (48, 1)
    dense_kernel_rt = dense_w.T

    return {
        "in_shape": [None, 1],
        "layers": [
            {
                "type": "gru",
                "shape": [None, None, hidden_size],
                "activation": "tanh",
                "weights": [
                    kernel_weights.astype(np.float32).tolist(),
                    recurrent_weights.astype(np.float32).tolist(),
                    gru_biases.astype(np.float32).tolist(),
                ],
            },
            {
                "type": "dense",
                "shape": [None, 1],
                "activation": "",
                "weights": [
                    dense_kernel_rt.astype(np.float32).tolist(),
                    dense_b.astype(np.float32).tolist(),
                ],
            },
        ],
        # Also include PyTorch state_dict keys for RTNeural::torch_helpers compatibility
        "gru.weight_ih_l0": w_ih.astype(np.float32).tolist(),
        "gru.weight_hh_l0": w_hh.astype(np.float32).tolist(),
        "gru.bias_ih_l0": b_ih.astype(np.float32).tolist(),
        "gru.bias_hh_l0": b_hh.astype(np.float32).tolist(),
        "dense.weight": dense_w.astype(np.float32).tolist(),
        "dense.bias": dense_b.astype(np.float32).tolist(),
    }


def export_state_dict_to_json(state_dict: Dict[str, Any], output_path: Path, hidden_size: int = 48) -> None:
    def to_np(val):
        if HAS_TORCH and isinstance(val, torch.Tensor):
            return val.detach().cpu().numpy().astype(np.float32)
        return np.asarray(val, dtype=np.float32)

    w_ih = to_np(state_dict["gru.weight_ih_l0"])
    w_hh = to_np(state_dict["gru.weight_hh_l0"])
    b_ih = to_np(state_dict["gru.bias_ih_l0"])
    b_hh = to_np(state_dict["gru.bias_hh_l0"])
    dense_w = to_np(state_dict["dense.weight"])
    dense_b = to_np(state_dict["dense.bias"])

    payload = build_rtneural_json_dict(w_ih, w_hh, b_ih, b_hh, dense_w, dense_b, hidden_size=hidden_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[export_rtneural] Exported RTNeural JSON weights -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch GRU Bass checkpoint to RTNeural JSON")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt/.pth checkpoint")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models" / "default_di_bass.json",
        help="Destination JSON path",
    )
    args = parser.parse_args()

    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to load .pt checkpoints.")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    export_state_dict_to_json(state_dict, args.output)


if __name__ == "__main__":
    main()
