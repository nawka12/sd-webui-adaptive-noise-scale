# sd-webui-adaptive-noise-scale

A standalone extension for [stable-diffusion-webui-reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) that ports the Adaptive Noise Scale calibration algorithm from [adept-sampler](https://github.com/nawka12/adept-sampler) to work transparently with reForge's built-in k-diffusion samplers.

## How it works

During sampling, the extension measures how quickly the denoised output changes relative to the sigma schedule (the "excess" ratio). After collecting enough samples in the texture phase (σ between 0.5 and 5.0), it computes a correction factor and **restarts sampling from scratch** with the corrected noise scale applied from step 0.

```
excess  = (‖d_i - d_{i-1}‖ / ‖d_{i-1} - d_{i-2}‖) / (σ_i / σ_{i-1})

correction = clamp(1 / median_excess^power, floor, ceiling)
```

Two-pass execution:
- **Pass 1 (calibration):** runs the sampler at scale 1.0 until the warmup window completes, then aborts.
- **Pass 2 (production):** restarts from the original latent with the computed correction applied.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| Warmup steps | 5 | Texture-phase steps to collect before computing correction |
| Correction power | 0.5 | Exponent for correction (0.5 = gentle, 1.0 = linear, >1 = aggressive) |
| Dampen floor | 0.80 | Minimum correction multiplier |
| Boost ceiling | 1.15 | Maximum correction multiplier |
| Phase-binned correction | On | Separate correction factors for structural / texture / cleanup phases |

## Supported samplers

Works with all standard reForge k-diffusion samplers: Euler a, DPM++ SDE, DPM++ 2M SDE, DPM++ 3M SDE, DPM++ 2S a, DPM2 a, DPM fast, DPM adaptive, Euler A2, and others that inject noise.

**Not supported (with warnings):**
- **DPM++ 2M, LMS** — fully deterministic, no noise injection.
- **Euler Dy, Euler SMEA Dy, Euler Negative, Euler Negative Dy** — SMEA family bypasses the noise interception point.

> **Note:** Euler, Heun, and DPM2 only inject noise when `s_churn > 0` (default is 0), so the extension is active but has no effect at default settings.

## Installation

Clone into your extensions folder:

```bash
cd stable-diffusion-webui-reForge/extensions
git clone https://github.com/nawka12/sd-webui-adaptive-noise-scale
```

Then restart the WebUI.

## Disclaimer

> **This extension has been developed and tested exclusively on [stable-diffusion-webui-reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge).**
> It has not been tested on other WebUI forks such as AUTOMATIC1111, SD WebUI Forge, Forge Neo, ComfyUI, or any other variant. Compatibility with those environments is not guaranteed and no support is provided for them.

## Credits

Algorithm ported from [adept-sampler](https://github.com/nawka12/adept-sampler).

## License

MIT
