from dreams.utils.data import MSData
from dreams.api import dreams_embeddings
import numpy as np
import h5py
import click

@click.command()
@click.argument('spectra_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
def embed(spectra_file, output_file):
    print("Loading spectra data...")
    msd = MSData.from_hdf5(spectra_file)
    
    print("Generating embeddings...")
    embeddings = dreams_embeddings(msd)
    
    print("Saving embeddings to HDF5...")
    with h5py.File(output_file, 'w') as hf:
        hf.create_dataset('spectrum_id', data=msd['spectrum_id'].values)
        hf.create_dataset('embedding', data=np.stack(embeddings['Embedding'].values))

if __name__ == '__main__':
    embed()