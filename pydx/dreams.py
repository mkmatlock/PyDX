from dreams.utils.data import MSData
from dreams.api import dreams_embeddings
import numpy as np
import h5py
import click

@click.command()
@click.argument('spectra_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
def embed(spectra_file, output_file):
    msd = MSData.from_hdf5(spectra_file)
    embeddings = dreams_embeddings(msd)
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('id', data=embeddings.index.values)
        hf.create_dataset('feature_id', data=embeddings['FeatureID'].values)
        hf.create_dataset('spectrum_id', data=embeddings['SpectrumID'].values)
        hf.create_dataset('embedding', data=np.stack(embeddings['Embedding'].values))
        
if __name__ == '__main__':
    embed()