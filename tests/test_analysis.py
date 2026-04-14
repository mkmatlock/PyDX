TEST_DATABASE_FILE = "/mnt/d/Data/PWID/20241203/241203_LM_PWID_untargeted_alignment.cdResult"

def test_reduce2d():
    import numpy as np
    from pydx.analysis import reduce2d
    
    input_array = np.arange(100).reshape((10,10))
    print(input_array)
    groups = np.array([6, 6, 6, 1, 1, 1, 1, 10, 2, 2])
    expected_output = np.array([[66, 69, 62, 67], [96, 99, 92, 97], [26, 29, 22, 27], [76, 79, 72, 77]])
    
    output_array = reduce2d(input_array, groups, op=np.max)
    np.testing.assert_equal(output_array, expected_output)

def test_get_isotope_pattern():
    import numpy as np
    from pydx.analysis import generate_isotope_spectrum, match_peaks
    
    spectrum = generate_isotope_spectrum("C22H28N2O", "H", 1)
    
    empirical_spectrum = np.array([337.22717, 338.23056, 339.12015, 339.23389, 340.23715, 340.26910, 341.09842, 341.15878, 342.26358])
    mapping = match_peaks(spectrum.mz, empirical_spectrum, tolerance=5)

    np.testing.assert_equal(mapping, [0, 1, 3, 4, -1, -1])

def test_convert_to_pyopenms():
    import numpy as np
    from pydx import PyDX
    from pydx.analysis import make_oms_spectrum
    idxa = PyDX(TEST_DATABASE_FILE) # Replace this with a smaller test database in the future
    
    spectra = idxa.get_spectra_by_id(446272)
    
    oms_spectrum = make_oms_spectrum(spectra.iloc[0])
    mz, intensity = oms_spectrum.get_peaks()
    
    np.testing.assert_allclose(np.sort(mz), np.sort(spectra.Spectrum.iloc[0]['mz']))
    np.testing.assert_allclose(np.sort(intensity), np.sort(spectra.Spectrum.iloc[0]['intensity']))

def test_convert_to_matchms_spectrum():
    import numpy as np
    from pydx import PyDX
    from pydx.analysis import make_matchms_spectrum
    idxa = PyDX(TEST_DATABASE_FILE) # Replace this with a smaller test database in the future
    
    spectra = idxa.get_spectra_by_id(446272)
    matchms_spectrum = make_matchms_spectrum(spectra.iloc[0])
    print(matchms_spectrum)
    
    np.testing.assert_allclose(np.sort(matchms_spectrum.peaks.mz), np.sort(spectra.Spectrum.iloc[0]['mz']))
    np.testing.assert_allclose(np.sort(matchms_spectrum.peaks.intensities), np.sort(spectra.Spectrum.iloc[0]['intensity']))