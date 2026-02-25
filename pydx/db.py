import sqlite3
import pandas as pd
from .parsers import decode_gap_fill_status, decode_spectrum
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine, inspect
from typing import List, Tuple, Union

sample_type_codes = {
    0: 'Sample',
    1: 'Unknown',
    2: 'Blank',
    3: 'Quality Control',
    4: 'Identification Only'
}

class LazyBlob(object):
    def __init__(self, data, decoder):
        self._data = data
        self._decoder = decoder
        self._decoded = False

    @property
    def data(self):
        if not self._decoded:
            self._data = self._decoder(self._data)
            self._decoded = True
        return self._data

class PyDX(object):
    def __init__(self, path):
        self.path = path
        self.engine = create_engine(f"sqlite:///{self.path}")
        Base = automap_base()
        Base.prepare(autoload_with=self.engine)
    
    @property
    def inputs(self):
        if not hasattr(self, '_workflow_data'):
            self._workflow_data = pd.read_sql_table('WorkflowInputFiles', con=self.engine)
        return self._workflow_data
        
    @property
    def samples(self):
        if not hasattr(self, '_samples_data'):
            self._samples_data = pd.read_sql_table('StudyInformation', con=self.engine)
        return self._samples_data
        
    @property
    def features(self):
        if not hasattr(self, '_features_data'):
            self._features_data = pd.read_sql_table('ConsolidatedUnknownCompoundItems', con=self.engine, index_col='ID')
        return self._features_data
        
    @property
    def spectra(self):
        if not hasattr(self, '_spectra_data'):
            self._spectra_data = pd.read_sql_table('MassSpectrumItems', con=self.engine)
        return self._spectra_data
    
    @property
    def chromatograms(self):
        if not hasattr(self, '_chromatograms_data'):
            self._chromatograms_data = pd.read_sql_table('ChromatogramPeakItems', con=self.engine)
        return self._chromatograms_data
    
    @property
    def corrected_retention_times(self):
        if not hasattr(self, '_corrected_retention_times_data'):
            self._corrected_retention_times_data = pd.read_sql_table('FileAlignmentCorrectionItems', con=self.engine)
        return self._corrected_retention_times_data
    
    
    """The following properties return tables that link the features to their best ChemSpider hits"""
    @property
    def chemspider_annotations(self):
        if not hasattr(self, '_chemspider_annotations_data'):
            self._chemspider_annotations_data = pd.read_sql_table('ConsolidatedUnknownCompoundItemsChemSpiderResultItems', con=self.engine)
        return self._chemspider_annotations_data
    
    @property
    def chemspider_hits(self):
        if not hasattr(self, '_chemspider_hits_data'):
            self._chemspider_hits_data = pd.read_sql_table('ChemSpiderResultItems', con=self.engine)
        return self._chemspider_hits_data
    
    def get_chemspider_hits_for_feature(self, feature_ids):
        join = f"""SELECT J.ConsolidatedUnknownCompoundItemsID AS FeatureID, C.ChemSpiderID AS ChemSpiderID, J.DeltaMassInPPM AS DeltaMass, J.MzLogicScore AS Score, C.MolStructure AS Structure, C.Name AS Name, C.Formula AS Formula, C.MolecularWeight AS MolecularWeight, C.InChi AS InChi, C.InChiKey AS InChiKey
                        FROM ConsolidatedUnknownCompoundItemsChemSpiderResultItems J JOIN ChemSpiderResultItems C 
                        ON J.ChemSpiderResultItemsChemSpiderID = C.ChemSpiderID
                        WHERE J.ConsolidatedUnknownCompoundItemsID IN {tuple(feature_ids)}"""
        return pd.read_sql_query(join, con=self.engine)
    
    """The following properties return tables that link the features to their best mzCloud hits"""
    @property
    def mzcloud_annotations(self):
        if not hasattr(self, '_mzcloud_annotations_data'):
            self._mzcloud_annotations_data = pd.read_sql_table('ConsolidatedUnknownCompoundItemsMzCloudSearchResultItems', con=self.engine)
        return self._mzcloud_annotations_data    

    @property
    def mzcloud_hits(self):
        if not hasattr(self, '_mzcloud_hits_data'):
            self._mzcloud_hits_data = pd.read_sql_table('MzCloudHitItems', con=self.engine)
        return self._mzcloud_hits_data
    
    
    """The following properties return tables that link the features to their best hit ions and associated spectra"""
    @property
    def best_hit_ion_annotations(self):
        if not hasattr(self, '_best_hit_ion_annotations_data'):
            self._best_hit_ion_annotations_data = pd.read_sql_table('ConsolidatedUnknownCompoundItemsBestHitIonInstanceItems', con=self.engine)
        return self._best_hit_ion_annotations_data
    
    @property
    def best_hit_ions(self):
        if not hasattr(self, '_best_hit_ions_data'):
            self._best_hit_ions_data = pd.read_sql_table('BestHitIonInstanceItems', con=self.engine)
        return self._best_hit_ions_data
    
    @property
    def spectra_ion_annotations(self):
        if not hasattr(self, '_mass_spectra_ion_annotations_data'):
            self._mass_spectra_ion_annotations_data = pd.read_sql_table('BestHitIonInstanceItemsMassSpectrumItems', con=self.engine)
        return self._mass_spectra_ion_annotations_data
    
    
    def get_compound_spectra(self, feature_ids):
        Q = f"""SELECT BHI.ConsolidatedUnknownCompoundItemsID AS FeatureID, MS.WorkflowID as WorkflowID, MS.ID AS SpectrumID, MS.MSOrder AS Order, MS.Polarity as Polarity, MS.RetentionTime as RetentionTime, MS.Ionization as Ionization, MS.MassAnalyzer as MassAnalyzer, MS.ScanPolarity as ScanPolarity, MS.Spectrum as Spectrum
                FROM ConsolidatedUnknownCompoundItemsBestHitIonInstanceItems BHI 
                        JOIN BestHitIonInstanceItemsMassSpectrumItems BHIMS 
                            ON BHI.BestHitIonInstanceItemsWorkflowID = BHIMS.BestHitIonInstanceItemsWorkflowID 
                                AND BHI.BestHitIonInstanceItemsID = BHIMS.BestHitIonInstanceItemsID
                        JOIN MassSpectrumItems MS
                            ON BHIMS.MassSpectrumItemsWorkflowID = MS.WorkflowID
                                AND BHIMS.MassSpectrumItemsID = MS.ID
                WHERE BHI.ConsolidatedUnknownCompoundItemsID IN {tuple(feature_ids)}"""
        return pd.read_sql_query(Q, con=self.engine)
                
    
    def show_table_schema(self, table_name):
        print(f"Table: {table_name}")
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name, schema='main')
        for column in columns:
            print(f"   Column: {column['name']}, Type: {column['type']}")