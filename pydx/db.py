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
        if len(feature_ids) == 1:
            where_clause = f"J.ConsolidatedUnknownCompoundItemsID = {feature_ids[0]}"
        elif len(feature_ids) > 1:
            where_clause = f"J.ConsolidatedUnknownCompoundItemsID IN {tuple(feature_ids)}"
        else:
            raise ValueError("feature_ids must contain at least one ID")
        join = f"""SELECT J.ConsolidatedUnknownCompoundItemsID AS FeatureID, C.ChemSpiderID AS ChemSpiderID, J.DeltaMassInPPM AS DeltaMass, J.MzLogicScore AS Score, C.MolStructure AS Structure, C.Name AS Name, C.Formula AS Formula, C.MolecularWeight AS MolecularWeight, C.InChi AS InChi, C.InChiKey AS InChiKey
                        FROM ConsolidatedUnknownCompoundItemsChemSpiderResultItems J JOIN ChemSpiderResultItems C 
                        ON J.ChemSpiderResultItemsChemSpiderID = C.ChemSpiderID
                        WHERE {where_clause}"""
        return pd.read_sql_query(join, con=self.engine)
    
    """The following properties return tables that link the features to their best mzCloud hits"""
    @property
    def mzcloud_annotations(self):
        if not hasattr(self, '_mzcloud_annotations_data'):
            self._mzcloud_annotations_data = pd.read_sql_table('ConsolidatedUnknownCompoundItemsMzCloudHitItems', con=self.engine)
        return self._mzcloud_annotations_data    
    
    @property
    def mzcloud_search_result_annotations(self):
        if not hasattr(self, '_mzcloud_search_result_annotations_data'):
            self._mzcloud_search_result_annotations_data = pd.read_sql_table('ConsolidatedUnknownCompoundItemsMzCloudSearchResultItems', con=self.engine)
        return self._mzcloud_search_result_annotations_data
    
    @property
    def mzcloud_search_results(self):
        if not hasattr(self, '_mzcloud_search_results_data'):
            self._mzcloud_search_results_data = pd.read_sql_table('MzCloudSearchResultItems', con=self.engine)
        return self._mzcloud_search_results_data

    @property
    def mzcloud_hits(self):
        if not hasattr(self, '_mzcloud_hits_data'):
            self._mzcloud_hits_data = pd.read_sql_table('MzCloudHitItems', con=self.engine)
        return self._mzcloud_hits_data
        
    def get_mzcloud_search_results_for_feature(self, feature_ids):
        if len(feature_ids) == 1:
            where_clause = f"J.ConsolidatedUnknownCompoundItemsID = {feature_ids[0]}"
        elif len(feature_ids) > 1:
            where_clause = f"J.ConsolidatedUnknownCompoundItemsID IN {tuple(feature_ids)}"
        else:
            raise ValueError("feature_ids must contain at least one ID")
        join = f"""SELECT J.ConsolidatedUnknownCompoundItemsID AS FeatureID, M.MzCloudId AS MzCloudID, M.KeggId as KeggId, M.Name AS Name, M.Formula AS Formula, M.Mass AS MolecularWeight, M.MolStructure as Structure, J.DeltaMassInPPM AS DeltaMass, J.MzLibraryMatchFactor AS Score, J.Confidence AS Confidence, J.CompoundMatchStatus AS Status
                        FROM ConsolidatedUnknownCompoundItemsMzCloudSearchResultItems J JOIN MzCloudSearchResultItems M 
                        ON J.MzCloudSearchResultItemsID = M.ID
                        WHERE {where_clause}"""
        return pd.read_sql_query(join, con=self.engine)
    
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
        if len(feature_ids) == 1:
                where_clause = f"BHI.ConsolidatedUnknownCompoundItemsID = {feature_ids[0]}"
        elif len(feature_ids) > 1:
            where_clause = f"BHI.ConsolidatedUnknownCompoundItemsID IN {tuple(feature_ids)}"
        else:
            raise ValueError("feature_ids must contain at least one ID")
        Q = f"""SELECT BHI.ConsolidatedUnknownCompoundItemsID AS FeatureID, MS.WorkflowID AS WorkflowID, MS.ID AS SpectrumID, MS.FileID AS FileID, MS.MSOrder AS MSn, MS.Polarity AS Polarity, MS.RetentionTime AS RetentionTime, MS.ResolutionAtMass200 AS Resolution, MS.ActivationType AS ActivationType, MS.ScanType AS ScanType, MS.Ionization AS Ionization, MS.MassAnalyzer AS MassAnalyzer, MS.Spectrum AS Spectrum
                FROM ConsolidatedUnknownCompoundItemsBestHitIonInstanceItems BHI 
                        JOIN BestHitIonInstanceItemsMassSpectrumItems BHIMS 
                            ON BHI.BestHitIonInstanceItemsWorkflowID = BHIMS.BestHitIonInstanceItemsWorkflowID 
                                AND BHI.BestHitIonInstanceItemsID = BHIMS.BestHitIonInstanceItemsID
                        JOIN MassSpectrumItems MS
                            ON BHIMS.MassSpectrumItemsWorkflowID = MS.WorkflowID
                                AND BHIMS.MassSpectrumItemsID = MS.ID
                WHERE {where_clause}"""
        return pd.read_sql_query(Q, con=idxa.engine)
                
    
    def show_table_schema(self, table_name):
        print(f"Table: {table_name}")
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name, schema='main')
        for column in columns:
            print(f"   Column: {column['name']}, Type: {column['type']}")