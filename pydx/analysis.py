import numpy as np
import sklearn

def probablistic_subset_likelihood(p_1, p_2):
    """Compute the likelihood that the distribution of feature f_1 in a set of samples is a subset of the distribution of feature f_2 in the same set.
    p_1: A vector of probabilities for feature 1 across the sample set
    p_2: A vector of probabilities for feature 2 across the sample set
    Returns a value between 0 and 1, where 1 indicates that the samples containing feature 1 are a subset of those containing feature 2, and 0 indicates that there is no overlap between the two features.
    
    This uses probabilistic implication:
        
        p(f_1 -> f_2) = p(not f_1 and f_2) = 1 - p(f_1 and not f_2) = 1 - p(f_1)(1 - p(f_2))
    
        p(f_i) = p(z_i = 1 | A_i)
        
        z_i is an indicator variable for whether feature f_i is present in a sample
    
    The metric is then computed as:
    
        sum( p(f_1 -> f_2) ) / sum( p(f_1) )
    """
    return np.sum(p_1 * (1-p_2), axis=1) / np.sum(p_1)

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