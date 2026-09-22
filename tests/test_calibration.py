import numpy as np
import pytest

from src.heston import PARAM_RANGES, HESTON_PARAM_NAMES
from src.data_gen import surface_for_params

pytest.importorskip("torch")
calibration = pytest.importorskip("src.model_calibration")

def require(name):
    if not hasattr(calibration, name):
        pytest.skip(f"{name} not implemented yet")


MID = np.array([(lo + hi) / 2 for lo, hi in PARAM_RANGES.values()])
RANGES = np.array([hi - lo for lo, hi in PARAM_RANGES.values()])
TRUE = np.array([1.5, 0.06, 0.6, -0.7, 0.05])


@pytest.fixture(scope="module")
def surrogate():
    from src.train_surrogate import load_model
    try:
        return load_model()
    except FileNotFoundError:
        pytest.skip("no trained model, run src.train_surrogate first")


@pytest.fixture(scope="module")
def target():
    return surface_for_params(*TRUE)


def test_synthetic_target_shape_and_range():
    params, surface = calibration.synthetic_target(random_seed=1)
    assert set(params) == set(HESTON_PARAM_NAMES)
    for name, value in params.items():
        lo, hi = PARAM_RANGES[name]
        assert lo <= value <= hi
    surface = np.asarray(surface).ravel()
    assert surface.shape == (63,)
    assert not np.isnan(surface).any()


def test_synthetic_target_is_reproducible():
    a = calibration.synthetic_target(random_seed=7)
    b = calibration.synthetic_target(random_seed=7)
    assert a[0] == b[0]
    np.testing.assert_allclose(np.asarray(a[1]).ravel(), np.asarray(b[1]).ravel())


def test_surrogate_recovers_params(surrogate, target):
    require("surrogate_calibrate")
    model, y_scaler = surrogate
    params, elapsed, _ = calibration.surrogate_calibrate(target, model, y_scaler)
    err = np.abs(params - TRUE) / RANGES
    assert err.mean() < 0.15
    assert elapsed < 5.0


def test_surrogate_stays_in_bounds(surrogate, target):
    require("surrogate_calibrate")
    model, y_scaler = surrogate
    params, _, _ = calibration.surrogate_calibrate(target, model, y_scaler)
    for p, name in zip(params, HESTON_PARAM_NAMES):
        lo, hi = PARAM_RANGES[name]
        assert lo <= p <= hi, f"{name} out of range"


def test_surrogate_beats_its_starting_point(surrogate, target):
    require("surrogate_calibrate")
    model, y_scaler = surrogate
    params, _, _ = calibration.surrogate_calibrate(target, model, y_scaler)
    start_err = np.abs(MID - TRUE) / RANGES
    end_err = np.abs(params - TRUE) / RANGES
    assert end_err.mean() < start_err.mean()


@pytest.mark.slow
def test_classical_recovers_params(target):
    require("classical_calibrate")
    params, elapsed, _ = calibration.classical_calibrate(target, max_iter=400)
    err = np.abs(params - TRUE) / RANGES
    assert err.mean() < 0.15
    assert elapsed > 0


@pytest.mark.slow
def test_surrogate_is_faster(surrogate, target):
    require("classical_calibrate")
    require("surrogate_calibrate")
    model, y_scaler = surrogate
    _, classical_time, _ = calibration.classical_calibrate(target, max_iter=200)
    _, surrogate_time, _ = calibration.surrogate_calibrate(target, model, y_scaler)
    assert surrogate_time < classical_time / 10
