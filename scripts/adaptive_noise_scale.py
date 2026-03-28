"""
Adaptive Noise Scale — standalone reForge extension.

Ports the adaptive noise calibration algorithm from cussam-sampler to work as a
transparent hook on the default k-diffusion samplers already in reForge.

Algorithm (identical to cussam-sampler's implementation):
  During a warmup window of the "texture phase" (0.5 < sigma < 5.0) we measure how
  quickly the denoised output is changing relative to the sigma schedule:

      excess  =  (||d_i - d_{i-1}|| / ||d_{i-1} - d_{i-2}||)
               / (sigma_i / sigma_{i-1})

  When excess > 1 noise slows convergence; when excess < 1 more noise could help.
  After collecting `warmup` texture-phase samples, a correction factor is derived:

      correction = clamp(1.0 / median_excess^power, floor, ceiling)

  The sampler is then RESTARTED from the original latent with the correction applied
  as a multiplicative scale on all subsequent noise injections.  Without restart the
  calibration is meaningless — the latent is already formed by incorrectly-scaled
  noise during the warmup window.

Two-pass execution (implemented by wrapping sampler.func):
  Pass 1 — calibration:  run the sampler normally (scale = 1.0) until the warmup
    trigger fires.  A _CalibrationAbort exception is raised from the callback to
    abort cleanly.  If no texture-phase data was collected (very short runs or
    purely deterministic schedules) the first pass result is returned as-is.
  Pass 2 — production:  restart from x_initial (saved at the start of pass 1)
    with scale_multiplier = correction applied from step 0.

Noise interception:
  ┌─ Standard k-diffusion samplers ──────────────────────────────────────────┐
  │  Noise from default_noise_sampler / direct randn_like calls is routed    │
  │  through the TorchHijack instance that sd_samplers_common sets on        │
  │  sampling.torch.  We patch TorchHijack.randn_like (A1111 backend) and    │
  │  inject a randn shim (ldm_patched backend) in process_before_every_step  │
  │  on the first callback.  The patches read scale_multiplier dynamically.  │
  ├─ BrownianTree samplers (DPM++ SDE / 2M SDE / 3M SDE) ─────────────────  │
  │  BrownianTree bypasses TorchHijack, so sampler.func is wrapped to scale  │
  │  the noise_sampler kwarg that KDiffusionSampler passes in.               │
  └───────────────────────────────────────────────────────────────────────────┘

Samplers NOT supported (warning emitted):
  • Deterministic: DPM++ 2M, LMS — no noise injection at all.
  • SMEA family (Euler Dy, etc.) — import torch directly, bypass TorchHijack.
"""

import functools
from functools import partial
import torch
import gradio as gr
from modules import scripts, script_callbacks
from modules.shared import opts

# Resolve which k-diffusion sampling module the current backend uses.
# reForge exposes opts.sd_sampling to select the backend; Forge Neo does not
# have this option and uses k_diffusion.sampling directly.
_sd_sampling = getattr(opts, 'sd_sampling', None)
if _sd_sampling == "A1111":
    from k_diff.k_diffusion import sampling as _k_sampling
elif _sd_sampling == "ldm patched (Comfy)":
    from ldm_patched.k_diffusion import sampling as _k_sampling
else:
    # Forge Neo (and compatible forks): k_diffusion is on sys.path directly.
    try:
        import k_diffusion.sampling as _k_sampling
    except ImportError:
        _k_sampling = None  # no interception possible

# ---------------------------------------------------------------------------
# Sampler compatibility tables
# ---------------------------------------------------------------------------

_SAMPLERS_NO_NOISE = frozenset({
    'sample_dpmpp_2m',   # DPM++ 2M  — fully deterministic multistep ODE
    'sample_lms',        # LMS       — linear multistep, no noise
})

# SMEA samplers import torch directly (bypassing TorchHijack) and read
# s_noise from opts — no clean per-step interception point.
_SAMPLERS_SMEA = frozenset({
    'sample_euler_dy',
    'sample_euler_smea_dy',
    'sample_euler_negative',
    'sample_euler_dy_negative',
})

_SAMPLER_LABELS = {
    'sample_euler_ancestral': 'Euler a',
    'sample_euler_a2':        'Euler A2',
    'sample_euler':           'Euler',
    'sample_heun':            'Heun',
    'sample_dpm_2':           'DPM2',
    'sample_dpm_2_ancestral': 'DPM2 a',
    'sample_dpm_fast':        'DPM fast',
    'sample_dpm_adaptive':    'DPM adaptive',
    'sample_dpmpp_2s_ancestral': 'DPM++ 2S a',
    'sample_dpmpp_sde':       'DPM++ SDE',
    'sample_dpmpp_2m':        'DPM++ 2M',
    'sample_dpmpp_2m_sde':    'DPM++ 2M SDE',
    'sample_dpmpp_3m_sde':    'DPM++ 3M SDE',
    'sample_lms':             'LMS',
    'sample_euler_dy':        'Euler Dy',
    'sample_euler_smea_dy':   'Euler SMEA Dy',
    'sample_euler_negative':  'Euler Negative',
    'sample_euler_dy_negative': 'Euler Negative Dy',
}

# Euler/Heun/DPM2 only inject noise when s_churn > 0 (default = 0).
_SAMPLERS_STOCHASTIC_OPTIONAL = frozenset({
    'sample_euler', 'sample_heun', 'sample_dpm_2',
})

# ---------------------------------------------------------------------------
# Exception used to abort the calibration pass
# ---------------------------------------------------------------------------

class _CalibrationAbort(Exception):
    """Raised inside the sampler callback to abort the calibration pass."""


# ---------------------------------------------------------------------------
# Algorithm helpers
# ---------------------------------------------------------------------------

def _compute_correction(excess_samples, power, floor_val, ceiling_val):
    """
    correction = clamp(1 / median_excess^power, floor, ceiling)
    Returns (correction, median_excess).
    """
    if not excess_samples:
        return 1.0, 0.0
    sorted_s = sorted(excess_samples)
    median = sorted_s[len(sorted_s) // 2]
    correction = 1.0 / (median ** power) if median > 0 else 1.0
    correction = max(floor_val, min(ceiling_val, correction))
    return correction, median


def _compute_binned_corrections(bins, global_corr, power, floor_val, ceiling_val, min_samples=3):
    """Per-phase corrections; bins with < min_samples fall back to global."""
    result = {}
    for name, samples in bins.items():
        if len(samples) >= min_samples:
            c, _ = _compute_correction(samples, power, floor_val, ceiling_val)
            result[name] = c
        else:
            result[name] = global_corr
    return result


def _get_phase_correction(sigma_val, bin_corrections, global_correction):
    """Return the correction for the current sigma value's phase."""
    if bin_corrections is None:
        return global_correction
    if sigma_val > 5.0:
        return bin_corrections.get('structural', global_correction)
    elif sigma_val > 0.5:
        return bin_corrections.get('texture', global_correction)
    else:
        return bin_corrections.get('cleanup', global_correction)


# ---------------------------------------------------------------------------
# Main script class
# ---------------------------------------------------------------------------

class AdaptiveNoiseScaleScript(scripts.Script):
    """
    Always-on script implementing two-pass adaptive noise calibration on top
    of the default reForge k-diffusion samplers.
    """

    sorting_priority = 14

    def __init__(self):
        super().__init__()
        self._state: dict = {}

    # --- UI --------------------------------------------------------------- #

    def title(self):
        return "Adaptive Noise Scale"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Adaptive Noise Scale", open=False):
            enabled = gr.Checkbox(label="Enable Adaptive Noise Scale", value=False)
            with gr.Row():
                warmup = gr.Slider(
                    label="Warmup steps", minimum=1, maximum=20, value=5, step=1,
                    info="Texture-phase steps to collect before computing correction")
                power = gr.Slider(
                    label="Correction power", minimum=0.1, maximum=2.0, value=0.5, step=0.05,
                    info="Exponent: 0.5 = sqrt (gentle), 1.0 = linear, >1 = aggressive")
            with gr.Row():
                floor_val = gr.Slider(
                    label="Dampen floor", minimum=0.5, maximum=1.0, value=0.80, step=0.01,
                    info="Minimum correction multiplier")
                ceiling_val = gr.Slider(
                    label="Boost ceiling", minimum=1.0, maximum=2.0, value=1.15, step=0.01,
                    info="Maximum correction multiplier")
            use_binned = gr.Checkbox(
                label="Phase-binned correction", value=True,
                info="Separate correction factors for structural / texture / cleanup phases")

        self.infotext_fields = [
            (enabled,    "ANS Enabled"),
            (warmup,     "ANS Warmup"),
            (power,      "ANS Power"),
            (floor_val,  "ANS Floor"),
            (ceiling_val,"ANS Ceiling"),
            (use_binned, "ANS Binned"),
        ]

        return [enabled, warmup, power, floor_val, ceiling_val, use_binned]

    # --- Hooks ------------------------------------------------------------ #

    def process_before_every_sampling(self, p, *args, **kwargs):
        self._restore_patches()
        self._state = {}

        if not args or not args[0]:
            return

        enabled     = bool(args[0])
        warmup      = int(args[1])   if len(args) > 1 else 5
        power       = float(args[2]) if len(args) > 2 else 0.5
        floor_val   = float(args[3]) if len(args) > 3 else 0.80
        ceiling_val = float(args[4]) if len(args) > 4 else 1.15
        use_binned  = bool(args[5])  if len(args) > 5 else True

        # XYZ grid overrides
        xyz = getattr(p, '_ans_xyz', {})
        if xyz:
            if 'enabled'    in xyz: enabled     = xyz['enabled']
            if 'warmup'     in xyz: warmup      = xyz['warmup']
            if 'power'      in xyz: power       = xyz['power']
            if 'floor'      in xyz: floor_val   = xyz['floor']
            if 'ceiling'    in xyz: ceiling_val = xyz['ceiling']
            if 'use_binned' in xyz: use_binned  = xyz['use_binned']

        if not enabled:
            return

        sampler = getattr(p, 'sampler', None)
        if sampler is None:
            return

        funcname = getattr(sampler, 'funcname', None)
        if funcname is None or callable(funcname):
            print("[ANS] Cannot determine sampler type — ANS will not be applied.")
            return

        label = _SAMPLER_LABELS.get(funcname, funcname)

        # ── Compatibility checks ──────────────────────────────────────────
        if funcname in _SAMPLERS_NO_NOISE:
            print(
                f"[ANS] ⚠ Warning: '{label}' is deterministic — no noise injection.  "
                "Adaptive Noise Scale will have no effect."
            )
            return

        if funcname in _SAMPLERS_SMEA:
            print(
                f"[ANS] ⚠ Warning: '{label}' (SMEA family) imports torch directly and "
                "bypasses the TorchHijack interception point.  ANS cannot intercept noise."
            )
            return

        if funcname in _SAMPLERS_STOCHASTIC_OPTIONAL:
            print(
                f"[ANS] Note: '{label}' injects noise only when s_churn > 0.  "
                "ANS is active but has no effect at the default s_churn = 0."
            )

        # ── Build mutable state dict (captured by closures) ──────────────
        state: dict = {
            'warmup':      warmup,
            'power':       power,
            'floor':       floor_val,
            'ceiling':     ceiling_val,
            'use_binned':  use_binned,
            # per-step tracking
            'prev_denoised':    None,
            'prev_change_norm': None,
            'prev_sigma':       None,
            # calibration results
            'excess_samples':   [],
            'excess_bins':      {'structural': [], 'texture': [], 'cleanup': []},
            'adaptive_corr':    None,
            'bin_corrections':  None,
            'scale_multiplier': 1.0,
            'calibrated':       False,
            # patch housekeeping
            'sampling_patched': False,
            'sampler_ref':      sampler,
            'original_func':    None,
        }
        self._state = state

        # ── Wrap sampler.func for two-pass restart ────────────────────────
        self._wrap_sampler_func(sampler, state, p)

        print(
            f"[ANS] Active on '{label}'.  "
            f"warmup={warmup}, power={power:.2f}, "
            f"floor={floor_val:.2f}, ceiling={ceiling_val:.2f}, "
            f"binned={use_binned}"
        )

    def process_before_every_step(self, p, *args, **kwargs):
        """
        Called every sampler step (via callback_state).
        Handles TorchHijack patching (first call), excess tracking, and
        per-step scale_multiplier updates during the production pass.
        """
        state = self._state
        if not state:
            return

        d = kwargs.get('d')
        if not d:
            return

        sigma    = d.get('sigma')
        denoised = d.get('denoised')
        if sigma is None or denoised is None:
            return

        # Patch TorchHijack on first step (it is available now).
        if not state['sampling_patched']:
            self._patch_sampling_torch(state)
            state['sampling_patched'] = True

        sigma_val = sigma.item() if torch.is_tensor(sigma) else float(sigma)

        # ── Excess ratio tracking ─────────────────────────────────────────
        prev_denoised    = state['prev_denoised']
        prev_sigma       = state['prev_sigma']
        prev_change_norm = state['prev_change_norm']

        if prev_denoised is not None and prev_sigma is not None:
            prev_sigma_val = (
                prev_sigma.item() if torch.is_tensor(prev_sigma) else float(prev_sigma)
            )

            change_norm = (
                torch.norm((denoised - prev_denoised).flatten(1), dim=1).mean().item()
            )

            if (prev_change_norm is not None
                    and sigma_val > 0
                    and prev_sigma_val > 0):

                change_ratio = change_norm / (prev_change_norm + 1e-8)
                sigma_ratio  = sigma_val   / (prev_sigma_val  + 1e-8)
                excess       = change_ratio / (sigma_ratio + 1e-8)

                if state['use_binned']:
                    if prev_sigma_val > 5.0:
                        state['excess_bins']['structural'].append(excess)
                    elif prev_sigma_val > 0.5:
                        state['excess_bins']['texture'].append(excess)
                    else:
                        state['excess_bins']['cleanup'].append(excess)

                # Warmup trigger: texture phase, calibration not yet done.
                if not state['calibrated'] and 0.5 < prev_sigma_val < 5.0:
                    state['excess_samples'].append(excess)
                    if len(state['excess_samples']) >= state['warmup']:
                        self._calibrate(state)

            state['prev_change_norm'] = change_norm
        else:
            state['prev_change_norm'] = None

        # ── Update scale for the noise injection that follows ─────────────
        if state['adaptive_corr'] is not None:
            if state['use_binned'] and state['bin_corrections'] is not None:
                state['scale_multiplier'] = _get_phase_correction(
                    sigma_val, state['bin_corrections'], state['adaptive_corr']
                )
            else:
                state['scale_multiplier'] = state['adaptive_corr']

        state['prev_denoised'] = denoised
        state['prev_sigma']    = sigma

    def postprocess(self, p, processed, *args):
        self._restore_patches()

    # --- Internal helpers ------------------------------------------------- #

    def _calibrate(self, state: dict) -> None:
        corr, median = _compute_correction(
            state['excess_samples'], state['power'], state['floor'], state['ceiling'],
        )
        state['adaptive_corr'] = corr
        state['calibrated']    = True

        if state['use_binned']:
            bin_corr = _compute_binned_corrections(
                state['excess_bins'], corr,
                state['power'], state['floor'], state['ceiling'],
            )
            state['bin_corrections'] = bin_corr
            bin_info = {k: f"{v:.3f}" for k, v in bin_corr.items()}
            print(
                f"[ANS] Calibrated — global={corr:.3f} "
                f"(median_excess={median:.3f}), phases={bin_info}"
            )
        else:
            print(
                f"[ANS] Calibrated — correction={corr:.3f} "
                f"(median_excess={median:.3f})"
            )

    def _patch_sampling_torch(self, state: dict) -> None:
        """
        Patch TorchHijack (at sampling.torch) so randn_like and randn calls
        inside the k-diffusion sampling module are scaled by scale_multiplier.
        Called once per sampling run on the first step callback.
        """
        if _k_sampling is None:
            return
        hijack = getattr(_k_sampling, 'torch', None)
        if hijack is None:
            return

        state['_hijack']          = hijack
        orig_rl                   = hijack.randn_like  # bound method
        state['_orig_randn_like'] = orig_rl
        state['_had_randn_attr']  = 'randn' in hijack.__dict__
        state['_orig_randn']      = hijack.__dict__.get('randn')

        _state_ref = state

        def scaled_randn_like(x):
            noise = orig_rl(x)
            s = _state_ref.get('scale_multiplier', 1.0)
            return noise * s if s != 1.0 else noise

        import torch as _rt

        def scaled_randn(*a, **kw):
            noise = _rt.randn(*a, **kw)
            s = _state_ref.get('scale_multiplier', 1.0)
            return noise * s if s != 1.0 else noise

        hijack.randn_like = scaled_randn_like
        hijack.randn      = scaled_randn

    def _wrap_sampler_func(self, sampler, state: dict, p) -> None:
        """
        Replace sampler.func with a wrapper that runs two passes:

          Pass 1 — calibration: scale = 1.0, aborted as soon as the warmup
            trigger fires in process_before_every_step.
          Pass 2 — production: restart from x_initial with the computed
            correction applied from step 0 onward.

        For BrownianTree samplers (noise_sampler kwarg), the noise_sampler is
        wrapped inside the closure to apply scale_multiplier dynamically.
        Non-BrownianTree noise goes through TorchHijack (patched separately).
        """
        original_func       = sampler.func
        state['original_func'] = original_func
        _state_ref          = state
        _p_ref              = p
        _self_ref           = self

        # reForge's ScriptRunner calls process_before_every_step automatically
        # on every step during the production pass; Forge Neo has no such method
        # so we must inject the per-step call ourselves into the callback.
        _runner_has_step_cb = hasattr(getattr(_p_ref, 'scripts', None),
                                      'process_before_every_step')

        def wrapped_func(*f_args, **f_kwargs):
            # f_args[0] = model, f_args[1] = x (sigma-scaled initial latent)
            x = f_args[1]
            x_initial = x.clone()

            # ── BrownianTree interception ─────────────────────────────────
            # If a noise_sampler was passed (BrownianTree), wrap it so it
            # reads scale_multiplier dynamically.  Leave out noise_sampler
            # entirely if it wasn't provided — the sampler will call
            # default_noise_sampler(x) which routes through TorchHijack.
            raw_ns = f_kwargs.get('noise_sampler')  # None for non-Brownian
            if raw_ns is not None:
                _ns = raw_ns

                def _scaled_ns(sigma, sigma_next):
                    noise = _ns(sigma, sigma_next)
                    s = _state_ref.get('scale_multiplier', 1.0)
                    return noise * s if s != 1.0 else noise

            # ── Calibration pass callback ─────────────────────────────────
            # Call the script's step logic directly rather than routing through
            # ScriptRunner — keeps the calibration pass self-contained and
            # works on both reForge (has process_before_every_step) and
            # Forge Neo (does not).
            def _cal_cb(d):
                _self_ref.process_before_every_step(p=_p_ref, d=d)
                if _state_ref.get('calibrated'):
                    raise _CalibrationAbort()

            # ── Pass 1: calibration ───────────────────────────────────────
            _state_ref['scale_multiplier'] = 1.0

            cal_kw = dict(f_kwargs)
            cal_kw['callback'] = _cal_cb
            if raw_ns is not None:
                cal_kw['noise_sampler'] = _scaled_ns  # scale = 1.0 during cal

            calibration_result  = None
            calibration_aborted = False
            try:
                calibration_result = original_func(*f_args, **cal_kw)
            except _CalibrationAbort:
                calibration_aborted = True
            # All other exceptions (InterruptedException, etc.) propagate.

            if not calibration_aborted:
                # No texture phase reached — return the completed result.
                return calibration_result

            # ── Pass 2: production ────────────────────────────────────────
            # Set initial scale (process_before_every_step will refine per
            # sigma phase on each step if binned correction is enabled).
            _state_ref['scale_multiplier'] = _state_ref.get('adaptive_corr', 1.0)

            # Reset per-step tracking for the fresh run.
            _state_ref['prev_denoised']    = None
            _state_ref['prev_change_norm'] = None
            _state_ref['prev_sigma']       = None

            # Rebuild f_args with x_initial in place of x.
            f_args_restart = (f_args[0], x_initial.to(x.device)) + f_args[2:]

            prod_kw = dict(f_kwargs)
            if raw_ns is not None:
                prod_kw['noise_sampler'] = _scaled_ns  # closure reads scale dynamically

            if not _runner_has_step_cb:
                # Forge Neo: ScriptRunner won't call process_before_every_step,
                # so wrap the original callback to inject our per-step tracking.
                _orig_cb = f_kwargs.get('callback')

                def _prod_cb(d):
                    _self_ref.process_before_every_step(p=_p_ref, d=d)
                    if _orig_cb is not None:
                        _orig_cb(d)

                prod_kw['callback'] = _prod_cb
            # else: reForge — ScriptRunner drives process_before_every_step
            # through the original callback automatically.

            return original_func(*f_args_restart, **prod_kw)

        # Preserve the original function's signature so that
        # inspect.signature(sampler.func).parameters returns the right params.
        # KDiffusionSampler.sample uses this to decide which kwargs to pass
        # (e.g. 'sigmas', 'sigma_min', 'n').  Without this, those kwargs are
        # silently dropped and the real sampler fails with a missing-argument error.
        functools.update_wrapper(wrapped_func, original_func)

        sampler.func = wrapped_func

    def _restore_patches(self) -> None:
        state = self._state
        if not state:
            return

        # Restore TorchHijack.
        hijack = state.get('_hijack')
        if hijack is not None:
            orig_rl = state.get('_orig_randn_like')
            if orig_rl is not None:
                hijack.randn_like = orig_rl
            if state.get('_had_randn_attr'):
                hijack.randn = state['_orig_randn']
            else:
                try:
                    del hijack.randn
                except AttributeError:
                    pass

        # Restore sampler.func.
        sampler = state.get('sampler_ref')
        orig_fn = state.get('original_func')
        if sampler is not None and orig_fn is not None:
            sampler.func = orig_fn

        self._state = {}


# ---------------------------------------------------------------------------
# XYZ grid integration
# ---------------------------------------------------------------------------

def _ans_set_value(p, x, xs, *, field):
    """Setter called by the XYZ grid for each cell."""
    if not hasattr(p, '_ans_xyz'):
        p._ans_xyz = {}
    try:
        if field in ('enabled', 'use_binned'):
            x = str(x).strip().lower() == 'true'
        elif field == 'warmup':
            x = int(x)
        elif field in ('power', 'floor', 'ceiling'):
            x = float(x)
        p._ans_xyz[field] = x
    except (ValueError, TypeError) as e:
        print(f"[ANS] XYZ Grid: invalid value '{x}' for '{field}': {e}")


def _make_ans_axis_options():
    xyz_grid = None
    for sd in scripts.scripts_data:
        if sd.script_class.__module__ == 'xyz_grid.py':
            xyz_grid = sd.module
            break
    if xyz_grid is None:
        return

    axis = [
        xyz_grid.AxisOption(
            "(ANS) Enabled",
            str,
            partial(_ans_set_value, field='enabled'),
            choices=lambda: ["True", "False"],
        ),
        xyz_grid.AxisOption(
            "(ANS) Warmup Steps",
            int,
            partial(_ans_set_value, field='warmup'),
        ),
        xyz_grid.AxisOption(
            "(ANS) Correction Power",
            float,
            partial(_ans_set_value, field='power'),
        ),
        xyz_grid.AxisOption(
            "(ANS) Dampen Floor",
            float,
            partial(_ans_set_value, field='floor'),
        ),
        xyz_grid.AxisOption(
            "(ANS) Boost Ceiling",
            float,
            partial(_ans_set_value, field='ceiling'),
        ),
        xyz_grid.AxisOption(
            "(ANS) Phase-Binned Correction",
            str,
            partial(_ans_set_value, field='use_binned'),
            choices=lambda: ["True", "False"],
        ),
    ]

    if not any(getattr(o, 'label', '').startswith('(ANS)') for o in xyz_grid.axis_options):
        xyz_grid.axis_options.extend(axis)


def _on_before_ui():
    try:
        _make_ans_axis_options()
    except Exception as e:
        print(f"[ANS] XYZ grid registration failed: {e}")


script_callbacks.on_before_ui(_on_before_ui)
