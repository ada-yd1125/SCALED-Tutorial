"""Reusable 3D pollution step from ``tools/pollution_code.py``."""

from __future__ import annotations

import torch


def heaviside(x: torch.Tensor) -> torch.Tensor:
    return (x > 0).to(x.dtype)


def upwind_step(
    concentration: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    solid: torch.Tensor,
    dt: float,
    source: torch.Tensor,
    max_iter: int = 20,
    conv_tol: float = 1e-6,
) -> torch.Tensor:
    u_face = 0.5 * (torch.roll(u, 1, 2) + u) * (
        1 - heaviside(torch.maximum(torch.roll(solid, 1, 2), solid))
    )
    v_face = 0.5 * (torch.roll(v, 1, 1) + v) * (
        1 - heaviside(torch.maximum(torch.roll(solid, 1, 1), solid))
    )
    w_face = 0.5 * (torch.roll(w, 1, 0) + w) * (
        1 - heaviside(torch.maximum(torch.roll(solid, 1, 0), solid))
    )

    u_imh, u_iph = u_face, torch.roll(u_face, -1, 2)
    v_jmh, v_jph = v_face, torch.roll(v_face, -1, 1)
    w_kmh, w_kph = w_face, torch.roll(w_face, -1, 0)
    hu_imh, hu_iph = heaviside(u_imh), heaviside(u_iph)
    hv_jmh, hv_jph = heaviside(v_jmh), heaviside(v_jph)
    hw_kmh, hw_kph = heaviside(w_kmh), heaviside(w_kph)

    a_im1 = -hu_imh * u_imh
    a_ip1 = (1 - hu_iph) * u_iph
    a_jm1 = -hv_jmh * v_jmh
    a_jp1 = (1 - hv_jph) * v_jph
    a_km1 = -hw_kmh * w_kmh
    a_kp1 = (1 - hw_kph) * w_kph
    a_center = (
        1.0 / dt
        + hu_iph * u_iph
        - (1 - hu_imh) * u_imh
        + hv_jph * v_jph
        - (1 - hv_jmh) * v_jmh
        + hw_kph * w_kph
        - (1 - hw_kmh) * w_kmh
    )
    diagonal = (
        1.0 / dt
        + torch.maximum(torch.abs(u_imh), torch.abs(u_iph))
        + torch.maximum(torch.abs(v_jmh), torch.abs(v_jph))
        + torch.maximum(torch.abs(w_kmh), torch.abs(w_kph))
    )

    updated = concentration.clone()
    for _ in range(max_iter):
        old = updated
        rhs = (
            concentration / dt
            + source
            + (diagonal - a_center) * old
            - a_im1 * torch.roll(old, 1, 2)
            - a_ip1 * torch.roll(old, -1, 2)
            - a_jm1 * torch.roll(old, 1, 1)
            - a_jp1 * torch.roll(old, -1, 1)
            - a_km1 * torch.roll(old, 1, 0)
            - a_kp1 * torch.roll(old, -1, 0)
        )
        updated = rhs / diagonal
        scale = torch.clamp(old.abs().max(), min=1e-12)
        if ((updated - old).abs().max() / scale).item() < conv_tol:
            break
    return updated
