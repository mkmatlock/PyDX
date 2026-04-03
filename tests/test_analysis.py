TEST_DATABASE_FILE = "../DrugContaminants/raw/241203_LM_PWID_untargeted_alignment.cdResult"

def test_convert_to_pyopenms():
    import numpy as np
    from pydx import PyDX
    from pydx.analysis import make_oms_spectrum
    idxa = PyDX(TEST_DATABASE_FILE) # Replace this with a smaller test database in the future
    
    spectra = idxa.get_spectra_by_id(446272)
    feature = idxa.get_features_by_id(spectra.FeatureID.iloc[0])
    
    oms_spectrum = make_oms_spectrum(spectra.Spectrum.iloc[0], feature.iloc[0], spectra.MSn.iloc[0])
    mz, intensity = oms_spectrum.get_peaks()
    
    np.testing.assert_allclose(np.sort(mz), np.sort(spectra.Spectrum.iloc[0]['mz']))
    np.testing.assert_allclose(np.sort(intensity), np.sort(spectra.Spectrum.iloc[0]['intensity']))
    
def test_convert_to_matchms_spectrum():
    import numpy as np
    from pydx import PyDX
    from pydx.analysis import make_matchms_spectrum
    idxa = PyDX(TEST_DATABASE_FILE) # Replace this with a smaller test database in the future
    
    spectra = idxa.get_spectra_by_id(446272)
    feature = idxa.get_features_by_id(spectra.FeatureID.iloc[0])
    matchms_spectrum = make_matchms_spectrum(spectra.Spectrum.iloc[0], feature.iloc[0], spectra.MSn.iloc[0])
    print(matchms_spectrum)
    
    np.testing.assert_allclose(np.sort(matchms_spectrum.peaks.mz), np.sort(spectra.Spectrum.iloc[0]['mz']))
    np.testing.assert_allclose(np.sort(matchms_spectrum.peaks.intensities), np.sort(spectra.Spectrum.iloc[0]['intensity']))