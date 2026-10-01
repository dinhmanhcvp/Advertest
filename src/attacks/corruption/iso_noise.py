"""Group A: Realistic camera ISO sensor noise (Poisson-Gaussian model).

Simulates the noise characteristics of high-ISO capture on real cameras.
At low severity the effect matches ISO 800 (slight grain); at high severity
it matches ISO 12800+ (heavy luminance and colour noise).

Unlike the existing ``gaussian_noise`` attack (which uses a single sigma),
this models the *signal-dependent* nature of real sensor noise: brighter
regions get more Poisson shot noise, and a baseline Gaussian read noise
is added uniformly. This better represents egocentric cameras that auto-
select high ISO in dim environments.

Reference:
    - Foi et al., "Practical Poissonian-Gaussian Noise Modeling and Fitting
      for Single-Image Raw-Data", IEEE TIP 2008
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from src.attacks import ATTACKS
from src.attacks.base import AttackContext, AttackParams, BaseAttack
from src.core.types import AttackGroup, CostClass, Sample


class ISONoiseParams(AttackParams):
    """Per-severity noise scale matching real ISO levels.

    ``poisson_scale`` controls signal-dependent shot noise intensity.
    ``gaussian_sigma`` controls read noise (signal-independent).
    """

    poisson_scale_per_severity: tuple[float, ...] = (0.02, 0.05, 0.10, 0.18, 0.30)
    gaussian_sigma_per_severity: tuple[float, ...] = (0.01, 0.02, 0.04, 0.07, 0.12)


@ATTACKS.register
class ISONoise(BaseAttack):
    """Poisson-Gaussian sensor noise simulating high-ISO egocentric capture."""

    name: ClassVar[str] = "iso_noise"
    group: ClassVar[AttackGroup] = "A"
    cost_class: ClassVar[CostClass] = "cheap"
    owner: ClassVar[str] = "egocentric"
    reference: ClassVar[str] = (
        "Foi et al., Practical Poissonian-Gaussian Noise Modeling, IEEE TIP 2008"
    )
    params_model: ClassVar[type[AttackParams]] = ISONoiseParams

    def apply(self, sample: Sample, severity: int, ctx: AttackContext) -> Sample:
        params: ISONoiseParams = self.params  # type: ignore[assignment]
        poisson_scale = self.level(severity, params.poisson_scale_per_severity)
        gaussian_sigma = self.level(severity, params.gaussian_sigma_per_severity)

        image = sample.image.copy()

        # Signal-dependent Poisson shot noise: variance proportional to signal.
        # We scale the image to a photon count, apply Poisson, then scale back.
        if poisson_scale > 0:
            # Avoid division by zero and negative values.
            safe_image = np.clip(image, 1e-6, 1.0)
            photon_count = safe_image / poisson_scale
            noisy_photons = ctx.rng.poisson(photon_count).astype(np.float32)
            image = noisy_photons * poisson_scale

        # Signal-independent Gaussian read noise.
        if gaussian_sigma > 0:
            read_noise = ctx.rng.normal(0, gaussian_sigma, image.shape).astype(
                np.float32
            )
            image = image + read_noise

        return sample.with_image(image)
