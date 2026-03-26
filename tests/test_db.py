TEST_DATABASE_FILE = "../DrugContaminants/raw/241203_LM_PWID_untargeted_alignment.cdResult"

def test_fetch_chemspider_annotations_for_feature_set():
    from pydx import PyDX
    idxa = PyDX(TEST_DATABASE_FILE) # Replace this with a smaller test database in the future
    
    feature_IDs = list(range(1,20))
    
    result = idxa.get_chemspider_search_results_for_feature(feature_IDs)    
    assert result.columns.tolist() == ['FeatureID', 'ChemSpiderID', 'Name', 'Formula', 'MolecularWeight', 'InChi', 'InChiKey', 'Structure', 'DeltaMass', 'Score', 'Status']
    
def test_get_mzcloud_search_results_for_feature_set():
    from pydx import PyDX
    idxa = PyDX(TEST_DATABASE_FILE) # Replace this with a smaller test database in the future
    
    feature_IDs = list(range(1,20))
    
    result = idxa.get_mzcloud_search_results_for_feature(feature_IDs)
    assert result.columns.tolist() == ['FeatureID', 'MzCloudID', 'KeggID', 'Name', 'Formula', 'MolecularWeight', 'Structure', 'DeltaMass', 'Score', 'Confidence', 'Status']
    
def test_get_features_decodes_all_fields():
    from pydx import PyDX
    idxa = PyDX(TEST_DATABASE_FILE) # Replace this with a smaller test database in the future
    
    feature = idxa.get_feature_by_name('Benzylfentanyl').iloc[0]
    assert feature['Area'].shape == (107,)
    assert feature['AreaFlags'].shape == (107,)
    assert feature['PeakRating'].shape == (107,)
    assert feature['PeakRatingFlags'].shape == (107,)
    assert feature['GapFillStatus'].shape == (107,)
    assert feature['GapStatus'].shape == (107,)
    assert isinstance(feature['MolStructure'], str)
    
def test_get_spectra_for_feature_decodes_spectra():
    import pandas as pd
    from pydx import PyDX
    idxa = PyDX(TEST_DATABASE_FILE) # Replace this with a smaller test database in the future
    
    fid = idxa.get_feature_by_name('Benzylfentanyl').iloc[0].ID
    spectra = idxa.get_compound_spectra(feature_ids=[fid]).Spectrum
    for i in range(len(spectra)):
        assert isinstance(spectra.iloc[i], pd.DataFrame)
        assert spectra.iloc[i].columns.tolist() == ["mz", "intensity", "Z", "resolution", "signalNoiseRatio"]
    
def test_get_retention_time_corrections_returns_decoded_curves():
    import numpy as np
    from pydx import PyDX
    idxa = PyDX(TEST_DATABASE_FILE) # Replace this with a smaller test database in the future
    
    corrections = idxa.corrected_retention_times
    for i in range(len(corrections)):
        oRT = corrections.iloc[i]['OriginalRT']
        cRT = corrections.iloc[i]['CorrectedRT']
        
        assert isinstance(oRT, np.ndarray)
        assert isinstance(cRT, np.ndarray)
        
        assert oRT.shape == cRT.shape