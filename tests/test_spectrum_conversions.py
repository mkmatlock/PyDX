"""Exercise the binary scientific dependencies without an external database."""

import numpy as np
import pandas as pd
import pytest

from pydx.analysis import make_matchms_spectrum, make_oms_spectrum


@pytest.fixture
def spectrum():
    return pd.Series({
        'Spectrum': pd.DataFrame({'mz': [100.0, 150.0], 'intensity': [10.0, 20.0]}),
        'RetentionTime': np.float64(12.5),
        'MSn': np.int64(2),
        'Precursor': {'precursor_mz': np.float64(200.0)},
    })


def test_pyopenms_conversion(spectrum):
    result = make_oms_spectrum(spectrum)
    mz, intensity = result.get_peaks()
    np.testing.assert_allclose(mz, spectrum.Spectrum.mz)
    np.testing.assert_allclose(intensity, spectrum.Spectrum.intensity)
    assert result.getRT() == 12.5
    assert result.getMSLevel() == 2
    assert result.getPrecursors()[0].getMZ() == 200.0


def test_matchms_conversion(spectrum):
    result = make_matchms_spectrum(spectrum)
    np.testing.assert_allclose(result.peaks.mz, spectrum.Spectrum.mz)
    np.testing.assert_allclose(result.peaks.intensities, spectrum.Spectrum.intensity)
    assert result.get('precursor_mz') == 200.0
