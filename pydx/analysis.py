import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt

def match_peaks(mz_1, mz_2, tolerance):
    """Match peaks between two spectra given their m/z values and a tolerance in ppm.
    
    Arguments:
        mz_1: A vector of m/z values for the first spectrum
        mz_2: A vector of m/z values for the second spectrum
        tolerance: The maximum allowed difference in m/z values for two peaks to be considered a match, in ppm
    Returns:
        A numpy array of indicies in mz_2 corresponding to the best match for each peak in mz_1, or -1 if no match is found within the tolerance.
    """
    matches = np.full(mz_1.shape, -1, dtype=int)
    for i, mz in enumerate(mz_1):
        ppm_diff = np.abs(mz_2 - mz) / mz * 1e6
        best_match_idx = np.argmin(ppm_diff)
        if ppm_diff[best_match_idx] <= tolerance:
            matches[i] = best_match_idx
    return matches

def generate_isotope_spectrum(formula, adduct, charge):
    """Generate an isotope spectrum for a given empirical formula using pyOpenMS.
    
    Arguments:
        formula: A string representing the empirical formula, e.g. "C10H15N2O"
        
    Returns:
        A dataframe with columns mz, intensity representing the isotope spectrum.
    """
    from pyopenms import EmpiricalFormula, CoarseIsotopePatternGenerator
    formula = EmpiricalFormula(formula)+EmpiricalFormula(adduct)
    print(formula.getMonoWeight())
    
    isotopes = formula.getIsotopeDistribution(CoarseIsotopePatternGenerator(6))

    L = isotopes.size()
    mz = np.zeros(L)
    intensity = np.zeros(L)
    for i, iso in enumerate(isotopes):
        mz[i] = iso.getMZ() / charge
        intensity[i] = iso.getIntensity()
    return pd.DataFrame({'mz': mz, 'intensity': intensity})

def plot_all_peak_areas(features, sample_filter=None, combine=False, include_gap_status=False):
    """
        Plot the peak areas for a set of features across. Optionally filter by sample type and plot gap status and gap fill method as well.
        Arguments:
            features: A subset of the dataframe returned by PyDX.features
            combine: If True, combine features with the same name by summing their peak areas
            include_gap_status: If True, plot gap status and gap fill method as colored bars behind the peak area bars. Note that this option is incompatible with combine=True.
    """
    if sample_filter is None:
        sample_filter = np.ones(len(features.iloc[0].Area))
    
    gap_color_map = {1: 'blue', 2:'yellow', 3:'red'}
    gap_status_color_map = {0: 'gray', 1: 'blue', 2:'red', 4:'red', 8:'red', 16:'red', 32:'red', 64:'blue', 128:'blue', 256:'yellow', 512:'yellow', 1024:'yellow'}
    
    all_peak_areas = features.apply(lambda row: row.Area[sample_filter].tolist(), axis=1, result_type='expand')
    if combine:
        all_peak_areas.insert(0, 'Name', features.Name)
        all_peak_areas = all_peak_areas.groupby('Name').sum()
    total_features = len(all_peak_areas)
    
    if include_gap_status and not combine: # options are mutually incompatible
        gap_status = features.apply(lambda row: row.GapStatus[sample_filter].tolist(), axis=1, result_type='expand')
        gap_fill_method = features.apply(lambda row: row.GapFillStatus[sample_filter].tolist(), axis=1, result_type='expand')
    
    _, ax = plt.subplots(1, total_features, figsize=(3 * total_features, 2))
    if total_features == 1:
        ax = [ax]
    x = np.arange(len(all_peak_areas.columns))
    max_area = all_peak_areas.max(axis=None)
    
    for i, ix in enumerate(all_peak_areas.index):
        ax[i].bar(x, np.log10(all_peak_areas.loc[ix]), width=1, color='blue')
        if include_gap_status and not combine:
             ax[i].bar(x, [1] * len(gap_status.loc[ix]), width=1, color=[gap_color_map[status] for status in gap_status.loc[ix].tolist()])
             ax[i].bar(x, [0.5] * len(gap_fill_method.loc[ix]), width=1, color=[gap_status_color_map[status] for status in gap_fill_method.loc[ix].tolist()])
        ax[i].set_xticks([])
        if combine:
            ax[i].set_title(f"{ix}", fontsize=10)
        else:
            ax[i].set_title(f"{features.loc[ix].Name} ({ix})", fontsize=10)
        ax[i].set_ylim(0, np.log10(max_area) * 1.2)
    
    plt.tight_layout()
    plt.show()

def reduce2d(X, groups, op):
    G = np.unique(groups)
    N = len(G)
    s = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            s[i,j] = op(X[groups==G[i]][:,groups==G[j]])
    return s

def plot_spectrum(ax, spectrum, mz_range=None, top_n=3, label_fmt="{mz:.4f}", min_mz_label_separation=0.5, hide_x_label=False):
    mz = spectrum['mz'].to_numpy()
    intensity = spectrum['intensity'].to_numpy()
    
    ax.stem(mz, intensity, markerfmt=" ", basefmt=" ")
    order = np.argsort(intensity)[::-1]
    top_idx = order[: min(top_n, len(order))]
    top_idx = top_idx[np.argsort(mz[top_idx])]
    
    labeled_mz = []
    for i in top_idx:
        mz_i = mz[i]
        inten_i = intensity[i]

        # simple declutter: skip if too close in m/z to an already-labeled peak
        if any(abs(mz_i - m) < min_mz_label_separation for m in labeled_mz):
            continue
        labeled_mz.append(mz_i)

        ax.annotate(
            label_fmt.format(mz=mz_i, intensity=inten_i),
            xy=(mz_i, inten_i),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=0,
        )

    if mz_range:
        ax.set_xlim(*mz_range)
    ax.set_ylim(0, max(intensity) * 1.33)
    if not hide_x_label:
        ax.set_xlabel("m/z")
    if hide_x_label:
        ax.set_xticklabels([])
        ax.set_xlabel(None)
    ax.set_ylabel("Intensity")

"""
    Plot all spectra in a dataframe returned by PyDX.get_spectra_by_id or PyDX.get_compound_spectra
    
    Arguments:
        spectra_df (pd.DataFrame): A dataframe as returned by PyDX.get_spectra_by_id or PyDX.get_compound_spectra
        names (List[str]]): Optional list of names to use as titles for each spectrum. If None, defaults to "Feature {FeatureID} Spectrum {SpectrumID}"
        precursor_mz (List[float]): Optional m/z value to indicate the precursor ion on the plot
        neutral_loss (Boolean): If True, plot (precursor_mz - mz) values (requires precursor_mz to be provided)
        columns: Number of columns to use in the plot grid
        **kwargs: Additional keyword arguments to pass to plot_spectrum
        
    Returns: 
        Tuple(Figure, Axes[][]): a matplotlib plot of the spectra
"""
def plot_all_spectra(spectra_df, names=None, precursor_mz=None, neutral_loss=False, columns=1, **kwargs):  
    if names is None:
        names = [f"Feature {j} Spectrum {i} " for i,j in zip(spectra_df.FeatureID, spectra_df.SpectrumID)]
    if type(names) is str:
        names = [names] * len(spectra_df)
    
    if neutral_loss:
        min_mz = min(min(precursor_mz - spectrum['mz']) for spectrum in spectra_df.Spectrum)
        min_mz = 0 if min_mz >= 0 else min_mz*1.1
        max_mz = max(max(precursor_mz - spectrum['mz']) for spectrum in spectra_df.Spectrum)
        max_mz = 0 if max_mz <= 0 else max_mz*1.1
    else:
        min_mz = 0
        max_mz = max(max(spectrum['mz']) for spectrum in spectra_df.Spectrum)*1.1
    
    fig, axes = plt.subplots(len(spectra_df)//columns + (len(spectra_df) % columns > 0), columns, figsize=(4*columns, 2*(len(spectra_df)//columns + (len(spectra_df) % columns > 0))), dpi=100, sharex=True)
    axes = np.array(axes).reshape(-1)  # Flatten in case of multiple rows
    for i, (spectrum, name) in enumerate(zip(spectra_df.Spectrum, names)):
        if neutral_loss:
            neutral_loss = spectrum['mz'] = precursor_mz[i] - spectrum['mz']
            precursor_loc = neutral_loss.abs().argmin()
            spectrum = spectrum.copy()
            spectrum['mz'] = precursor_mz[i] - spectrum['mz']
            spectrum.drop(index=spectrum.index[precursor_loc], inplace=True)  # Remove precursor peak from plot
            
        plot_spectrum(axes[i], spectrum, mz_range=(min_mz, max_mz), hide_x_label=(i<(len(spectra_df)-1)), **kwargs)
        axes[i].set_title(name)
    return fig, axes

def convert_native_type(value):
    """Convert a value from the native type used in the IDX database to a standard Python type for use in pyOpenMS and matchms.
    
    Arguments:
        value: A value from the PyDX dataframe, which may be of a native type such as numpy.float32 or numpy.int64
        
    Returns:
        The input value converted to a standard Python type such as float or int
    """
    if isinstance(value, np.generic):
        return value.item()
    return value

def make_oms_spectrum(spectrum):
    """
    Build a pyOpenMS MSSpectrum from an IDX spectrum
    
    Arguments:
        spectrum: A row from the dataframe returned by PyDX.get_spectra_by_id or PyDX.get_compound_spectra
        
    Returns:
        A pyOpenMS MSSpectrum object with the m/z and intensity values from the input spectrum
    """
    import pyopenms as oms
    
    mz = np.asarray(spectrum.Spectrum.mz, dtype=float)
    intensity = np.asarray(spectrum.Spectrum.intensity, dtype=float)

    if mz.shape != intensity.shape:
        raise ValueError("mz and intensity must have the same shape")

    spec = oms.MSSpectrum()
    spec.setRT(convert_native_type(spectrum.RetentionTime))
    spec.setMSLevel(convert_native_type(spectrum.MSn))
    if spectrum.MSn > 1:
        p = oms.Precursor()
        p.setMZ(convert_native_type(spectrum.Precursor['precursor_mz']))
        spec.setPrecursors([p])
    spec.set_peaks((mz, intensity))
    spec.sortByPosition()  # important before alignment
    return spec

def make_matchms_spectrum(spectrum):
    """
    Build a matchms Spectrum from an IDX spectrum
    
    Arguments:
        spectrum: A row from the dataframe returned by PyDX.get_spectra_by_id or PyDX.get_compound_spectra
        
    Returns:
        A matchms Spectrum object with the m/z and intensity values from the input spectrum, and a metadata field "precursor_mz" if the spectrum is MS2 or higher.
    """
    
    from matchms import Spectrum
    
    mz = np.asarray(spectrum.Spectrum.mz, dtype=float)
    intensity = np.asarray(spectrum.Spectrum.intensity, dtype=float)

    if mz.shape != intensity.shape:
        raise ValueError("mz and intensity must have the same shape")

    metadata = {'precursor_mz': spectrum.Precursor['precursor_mz']} if spectrum.MSn > 1 else {}
    return Spectrum(mz=mz, intensities=intensity, metadata=metadata)

def retention_time_interpolator(rt_in, rt_out):
    """Compute the parameters of a linear regression model to convert retention times from rt_in to rt_out.
    
    Arguments:
        rt_in: A vector of original retention times
        rt_out: A vector of corrected retention times"""
        
    return lambda x: np.interp(x, rt_in, rt_out)

def probablistic_subset_likelihood(p_1, p_2):
    """Compute the likelihood that the distribution of feature f_1 in a set of samples is a subset of the distribution of feature f_2 in the same set.
    p_1: A vector of probabilities for feature 1 across the sample set
    p_2: A vector of probabilities for feature 2 across the sample set
    Returns a value between 0 and 1, where 1 indicates that the samples containing feature 1 are a subset of those containing feature 2, and 0 indicates that there is no overlap between the two features.
    
    This uses probabilistic implication:
        
        p(f_1 -> f_2) = p(not f_1 and f_2) = 1 - p(f_1 and not f_2) = 1 - p(f_1)(1 - p(f_2))
    
        p(f_i) = p(z_i = 1 | A_i)
        
        z_i is an indicator variable for whether feature f_i is present in a sample
    
    We compute the likelihood that, given two vectors of indicator variables z_1 and z_2 with joint distributions p_1 and p_2,
        p(z_1 /cdot z_2 = z_1) = prod(1 - p_1 * (1-p_2))
    """
    return np.prod(1 - p_1 * (1-p_2))

def compute_peak_likelihood(peak_area, gap_status, gap_fill_method):
    """Compute peak likelihoods for use in the probabilistic subset likelihood function.
    
    Models a mixture distribution based on gap status and gap fill methods and the peak area. 
    If peak area is zero, then the likelihood is zero. If peak area is nonzero, then likelihood
    is calculated by logistic regression on the log(peak areas), where peaks are labeled as positive
    if they have a gap status of "No Gap" or a gap fill method of "Unknown", "Original ion used", 
    "Filled by re-detected peak", or "Filled by matching ion".
    
    Arguments:
        peak_area: A vector of peak areas across the sample set
        gap_status: A vector of gap status codes across the sample set
        gap_fill_method: A vector of gap fill method codes across the sample set
        
    Returns:
        A vector of likelihoods between 0 and 1 inclusive for each sample.
    """
    positive_gap_fill_codes = [0, 1, 64, 128]
    feature_targets = np.isin( gap_fill_method, positive_gap_fill_codes) | (gap_status == 1)
 
    nonzero = peak_area > 0
    nonzero_targets = feature_targets[nonzero]
    nonzero_areas = np.log10(peak_area[nonzero])
    
    if nonzero_targets.sum() == 0:
        return np.zeros(feature_targets.shape)
    if nonzero_targets.sum() == len(nonzero_targets):
        return np.ones(feature_targets.shape)

    model = sklearn.linear_model.LogisticRegression()
    model.fit(nonzero_areas.reshape(-1, 1), nonzero_targets)
    lr_output = model.predict_proba(nonzero_areas.reshape(-1, 1))[:, 1]
    probabilities = np.zeros(feature_targets.shape)
    probabilities[nonzero] = lr_output
    probabilities[~nonzero] = 0
    return probabilities

def compute_all_peak_likelihoods(peak_area, gap_fill_method, gap_status):
    D = np.zeros(peak_area.shape)
    for i, index in enumerate(peak_area.index):
        A = peak_area.loc[index].values
        GFM = gap_fill_method.loc[index].values
        GS = gap_status.loc[index].values
        D[i, :] = compute_peak_likelihood(A, GS, GFM)
    return pd.DataFrame(D, index=peak_area.index, columns=peak_area.columns)