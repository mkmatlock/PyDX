TEST_DATABASE_FILE = "../DrugContaminants/raw/241203_LM_PWID_untargeted_alignment.cdResult"

def test_fetch_chemspider_annotations_for_feature_set():
    from pydx import PyDX
    idxa = PyDX(TEST_DATABASE_FILE) # Replace this with a smaller test database in the future
    
    feature_IDs = list(range(1,20))
    
    result = idxa.get_chemspider_hits_for_feature(feature_IDs)    
    assert result.columns.tolist() == ['FeatureID', 'ChemSpiderID', 'DeltaMass', 'Score', 'Structure', 'Name', 'Formula', 'MolecularWeight', 'InChi', 'InChiKey']