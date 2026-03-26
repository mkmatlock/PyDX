import numpy as np
import pandas as pd
import sklearn
import pyopenms as oms
from matplotlib import pyplot as plt

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
    for i, (spectrum, name) in enumerate(zip(spectra_df, names)):
        if neutral_loss:
            neutral_loss = spectrum['mz'] = precursor_mz[i] - spectrum['mz']
            precursor_loc = neutral_loss.abs().argmin()
            spectrum = spectrum.copy()
            spectrum['mz'] = precursor_mz[i] - spectrum['mz']
            spectrum.drop(index=spectrum.index[precursor_loc], inplace=True)  # Remove precursor peak from plot
            
        plot_spectrum(axes[i], spectrum, mz_range=(min_mz, max_mz), hide_x_label=(i<(len(spectra_df)-1)), **kwargs)
        axes[i].set_title(name)
    return fig, axes

def make_spectrum(mz, intensity, ms_level=2):
    """
    Build a pyOpenMS MSSpectrum from arrays/lists of m/z and intensity.
    """
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    if mz.shape != intensity.shape:
        raise ValueError("mz and intensity must have the same shape")

    spec = oms.MSSpectrum()
    spec.setMSLevel(ms_level)
    spec.set_peaks((mz, intensity))
    spec.sortByPosition()  # important before alignment
    return spec


def aligned_cosine_similarity(spec1, spec2, tolerance=0.01, unit="Da"):
    """
    Cosine similarity using pyOpenMS SpectrumAlignment.

    Parameters
    ----------
    spec1, spec2 : oms.MSSpectrum
    tolerance : float
        m/z matching tolerance
    unit : {"Da", "ppm"}
        Tolerance unit

    Returns
    -------
    score : float
        Cosine similarity in [0, 1]
    matches : list[tuple[int, int]]
        Matched peak index pairs
    """
    aligner = oms.SpectrumAlignment()
    params = aligner.getParameters()

    # Parameter names may vary slightly by version, but this is the usual pattern.
    params.setValue("tolerance", tolerance)
    params.setValue("is_relative_tolerance", unit.lower() == "ppm")
    aligner.setParameters(params)

    alignment = []
    aligner.getSpectrumAlignment(alignment, spec1, spec2)

    mz1, i1 = spec1.get_peaks()
    mz2, i2 = spec2.get_peaks()

    matched_1 = np.zeros(len(i1), dtype=float)
    matched_2 = np.zeros(len(i2), dtype=float)

    for idx1, idx2 in alignment:
        matched_1[idx1] = i1[idx1]
        matched_2[idx2] = i2[idx2]

    # Build union vectors: matched peaks plus unmatched peaks as zeros
    # This gives a cosine on the aligned representation.
    v1 = []
    v2 = []

    used1 = set()
    used2 = set()

    for idx1, idx2 in alignment:
        v1.append(i1[idx1])
        v2.append(i2[idx2])
        used1.add(idx1)
        used2.add(idx2)

    for idx1 in range(len(i1)):
        if idx1 not in used1:
            v1.append(i1[idx1])
            v2.append(0.0)

    for idx2 in range(len(i2)):
        if idx2 not in used2:
            v1.append(0.0)
            v2.append(i2[idx2])

    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    score = 0.0 if denom == 0 else float(np.dot(v1, v2) / denom)

    return score, alignment


def pairwise_similarity_matrix(spectra, tolerance=0.01, unit="Da"):
    """
    Compute an NxN similarity matrix for a list of MSSpectrum objects.
    """
    n = len(spectra)
    sim = np.eye(n, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            score, _ = aligned_cosine_similarity(
                spectra[i], spectra[j], tolerance=tolerance, unit=unit
            )
            sim[i, j] = score
            sim[j, i] = score

    return sim

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