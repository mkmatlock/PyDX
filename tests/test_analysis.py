TEST_DATABASE_FILE = "../DrugContaminants/raw/241203_LM_PWID_untargeted_alignment.cdResult"

def test_make_oms_spectrum_converts_to_pyopenms():
    from pydx import PyDX
    from pydx.analysis import make_oms_spectrum
    idxa = PyDX(TEST_DATABASE_FILE) # Replace this with a smaller test database in the future
    
    spectra = idxa.get_spectra_by_id(446272)
    feature = idxa.get_features_by_id(spectra.FeatureID.iloc[0])
    
    oms_spectrum = make_oms_spectrum(spectra.Spectrum.iloc[0], feature.iloc[0])