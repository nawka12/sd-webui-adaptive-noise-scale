"""Tests for _resolve_funcname — no WebUI imports required."""
import sys
import types

# ---------------------------------------------------------------------------
# Inline the function under test so we don't need the full WebUI environment.
# Keep this in sync with the implementation in scripts/adaptive_noise_scale.py.
# ---------------------------------------------------------------------------

def _resolve_funcname(sampler):
    fn = getattr(sampler, 'funcname', None)
    if fn is None:
        return None
    if callable(fn):
        return getattr(fn, '__name__', None)
    return fn


class _Sampler:
    def __init__(self, funcname):
        self.funcname = funcname


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_string_funcname_returned_as_is():
    s = _Sampler('sample_euler_ancestral')
    assert _resolve_funcname(s) == 'sample_euler_ancestral'


def test_callable_funcname_returns_dunder_name():
    def sample_er_sde(): pass
    s = _Sampler(sample_er_sde)
    assert _resolve_funcname(s) == 'sample_er_sde'


def test_callable_without_name_returns_none():
    # Callable with no __name__ (edge case: class instance with __call__)
    class NoName:
        def __call__(self): pass
    s = _Sampler(NoName())
    assert _resolve_funcname(s) is None


def test_missing_funcname_attribute_returns_none():
    class NoFuncname: pass
    assert _resolve_funcname(NoFuncname()) is None


def test_none_funcname_returns_none():
    s = _Sampler(None)
    assert _resolve_funcname(s) is None


def test_bound_method_funcname():
    """AlterSampler ODE samplers store bound methods (e.g. self.sample_ode_bosh3)."""
    class Fake:
        def sample_ode_bosh3(self): pass
    fake = Fake()
    s = _Sampler(fake.sample_ode_bosh3)
    assert _resolve_funcname(s) == 'sample_ode_bosh3'
