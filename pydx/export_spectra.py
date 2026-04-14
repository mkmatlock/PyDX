import sys

import click
import h5py
import pandas as pd
import numpy as np
from pydx import PyDX
from pydx.analysis import retention_time_interpolator
from pydx.parsers import activation_type_codes
from rdkit import Chem


polarity_codes = {
    1: "Positive",
    2: "Negative"
}

export_fields = ['id', 'file_id', 'feature_id', 'spectrum_id', 'spectrum', 'level', 'fragmentation_method', 'precursor_mz', 'RT', 'charge', 'adduct', 'polarity', 'exact_mass', 'formula', 'smiles']

def init_hdf5_datasets(h5file, example_batch, compression="gzip"):
    """
    Create one resizable dataset per column based on an example batch.

    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file handle.
    example_batch : dict[str, np.ndarray]
        Mapping from column name to batch array.
        Each array must have shape (N, ...) where N is batch size.
    compression : str | None
        Compression algorithm, e.g. 'gzip', or None.
    """
    for name, arr in example_batch.items():
        arr = np.asarray(arr)

        if arr.ndim < 1:
            raise ValueError(f"Column {name!r} must have at least 1 dimension")

        # Dataset starts with 0 rows, but fixed trailing dimensions
        shape = (0,) + arr.shape[1:]
        maxshape = (None,) + arr.shape[1:]

        h5file.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=arr.dtype,
            chunks=True,           # required for resizing
            compression=compression,
        )
        
def append_batch(h5file, batch):
    """
    Append one batch of column arrays to existing datasets.

    Parameters
    ----------
    h5file : h5py.File
        Open HDF5 file handle with datasets already created.
    batch : dict[str, np.ndarray]
        Mapping from column name to batch array.
    """
    # Check consistent batch size
    batch_sizes = {name: np.asarray(arr).shape[0] for name, arr in batch.items()}
    if len(set(batch_sizes.values())) != 1:
        raise ValueError(f"Inconsistent batch sizes: {batch_sizes}")

    n_new = next(iter(batch_sizes.values()))

    for name, arr in batch.items():
        arr = np.asarray(arr)
        dset = h5file[name]

        # Validate trailing dimensions
        if arr.shape[1:] != dset.shape[1:]:
            raise ValueError(
                f"Shape mismatch for column {name!r}: "
                f"batch has trailing shape {arr.shape[1:]}, "
                f"dataset expects {dset.shape[1:]}"
            )

        old_size = dset.shape[0]
        new_size = old_size + n_new

        dset.resize((new_size,) + dset.shape[1:])
        dset[old_size:new_size, ...] = arr

def reformat_formula(formula):
    if pd.isnull(formula):
        return ""
    return "".join(formula.split())

def reformat_spectrum(spectra, num_peaks):
    formatted_spectra = np.zeros((len(spectra), 2, num_peaks), dtype=np.float32)
    for i, spectrum in enumerate(spectra):
        spectrum.sort_values(by="intensity", inplace=True, ascending=False)
        total_peaks = min(len(spectrum), num_peaks)
        spectrum = spectrum.iloc[0:total_peaks]
        sorted_spectrum = spectrum.sort_values(by="mz", ascending=True)
        
        formatted_spectra[i, 0, 0:total_peaks] = sorted_spectrum.mz
        formatted_spectra[i, 1, 0:total_peaks] = sorted_spectrum.intensity
    return formatted_spectra
    
def batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def structure_to_smi(feature_row, delta_mass):
    struct = feature_row.MolStructure
    if pd.isnull(struct):
        return ""
    if delta_mass < abs(feature_row.AnnotationDeltaMassInPPM):
        return ""
    rdmol = Chem.MolFromMolBlock(struct, sanitize=False, removeHs=False)
    if rdmol is None:
        return ""
    return Chem.MolToSmiles(rdmol)
    
def generate_batches(idxa, msn_level, apply_rt_correction, delta_mass, num_peaks):
    unicode_dt = h5py.string_dtype(encoding='utf-8')    

    file_id_to_study_file_id_map = {}
    for _, row in idxa.inputs.iterrows():
        file_id_to_study_file_id_map[row.FileID] = row.StudyFileID
    
    if apply_rt_correction:
        rt_corrections = {}
        for _, row in idxa.corrected_retention_times.iterrows():
            rt_corrections[row.FileID] = retention_time_interpolator(row.OriginalRT, row.CorrectedRT)
    
    spectrum_ids = idxa.filter_spectrum_ids(msn_level=msn_level)
    print(f"Found {len(spectrum_ids)} matching spectra...", file=sys.stderr)
    totalN = 0
    
    with click.progressbar(batched(spectrum_ids, 100), length=(len(spectrum_ids) + 99) // 100, label="Exporting spectra") as bar:
        for batch in bar:
            spectra = idxa.get_spectra_by_id(batch)
            spectrum_ids = spectra.SpectrumID.to_numpy().astype(int)
            
            formatted_spectra = reformat_spectrum(spectra.Spectrum.to_list(), num_peaks).astype(np.float32)
            level = spectra.MSn.to_numpy().astype(int)
            fragmentation_method = spectra.ActivationType.apply(lambda x: activation_type_codes.get(x, 'Unknown')).to_numpy().astype(unicode_dt)
            precursor_mz = spectra.Precursor.apply(lambda x: (0 if x is None else x['precursor_mz'])).to_numpy().astype(np.float32)
            charge = spectra.Precursor.apply(lambda x: (0 if x is None else x['charge'])).to_numpy().astype(int)
            
            if apply_rt_correction:
                RT = spectra.apply(lambda row: rt_corrections[row.FileID](row.RetentionTime), axis=1).to_numpy().astype(np.float32)
            else:
                RT = spectra.RetentionTime.to_numpy().astype(np.float32)
            polarity = spectra.Polarity.apply(lambda x: polarity_codes[x]).to_numpy().astype(unicode_dt)
            
            batch_result = {'spectrum_id': spectrum_ids, 'spectrum': formatted_spectra, 'ms_level': level, 'fragmentation_method': fragmentation_method, 'precursor_mz': precursor_mz, 'charge': charge, 'RT': RT, 'polarity': polarity}
            totalN += len(spectrum_ids)
            yield batch_result
            
    print(f"Finished exporting {totalN} spectra.", file=sys.stderr)
            
@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--msn-level', '-l', default=None, type=int, help='MS level to export (default: all)')
@click.option('--num-peaks', '-n', default=128, type=int, help='Number of peaks to include in the exported spectrum (default: 128)')
@click.option('--delta-mass', '-d', default=5, type=float, help='PPM mass tolerance for structure annotation data (default: 5)')
@click.option('--apply-rt-correction', '-r', is_flag=True, help='Apply retention time correction to the exported data')
@click.option('--output', '-o', default='output.hdf5', help='Output file name')
def export(input_file, msn_level, num_peaks, delta_mass, apply_rt_correction, output):
    idxa = PyDX(input_file)
    
    with h5py.File(output, 'w') as hf:
        batch_iterator = generate_batches(idxa, msn_level, apply_rt_correction, delta_mass, num_peaks)
        
        batch = next(batch_iterator)
        init_hdf5_datasets(hf, batch)
        append_batch(hf, batch)
        
        for batch in batch_iterator:
            append_batch(hf, batch)


if __name__ == '__main__':
    export()