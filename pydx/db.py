import sqlite3
import pandas as pd
from .parsers import decode_gap_fill_status, decode_gap_status, decode_mol_structure, decode_peak_areas, decode_peak_ratings, decode_retention_times, decode_spectrum, decode_spectrum_to_xml
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine, inspect
from typing import List, Tuple, Union
from collections.abc import Iterable

class PyDX(object):
    
    def __init__(self, path):
        self.path = path
        self.engine = create_engine(f"sqlite:///{self.path}")
        Base = automap_base()
        Base.prepare(autoload_with=self.engine)
    
    
    """The following properties return tables with sample data"""
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
        
        
    """The following properties access feature data and decode peak data vectors and molecular structures"""
    def _decode_feature_vectors(self, features_df):
        features_df[['Area', 'AreaFlags']] = features_df[['Area']].apply(lambda row: decode_peak_areas(row['Area']), axis=1, result_type='expand')
        features_df[['PeakRating', 'PeakRatingFlags']] = features_df[['PeakRating']].apply(lambda row: decode_peak_ratings(row['PeakRating']), axis=1, result_type='expand')
        features_df['GapFillStatus'] = features_df['GapFillStatus'].apply(decode_gap_fill_status)
        features_df['GapStatus'] = features_df['GapStatus'].apply(decode_gap_status)
        features_df['MolStructure'] = features_df['MolStructure'].apply(decode_mol_structure)
        return features_df
    
    @property
    def features(self):
        if not hasattr(self, '_features_data'):
            self._features_data = pd.read_sql_table('ConsolidatedUnknownCompoundItems', con=self.engine, index_col='ID')
            self._decode_feature_vectors(self._features_data)
        return self._features_data 
        
    def get_feature_by_name(self, name):
        tab = pd.read_sql_query(f"SELECT * FROM ConsolidatedUnknownCompoundItems WHERE Name = '{name}'", con=self.engine)
        self._decode_feature_vectors(tab)
        return tab
    
    def get_features_by_id(self, feature_ids):
        if not isinstance(feature_ids, Iterable):
            feature_ids = [feature_ids]
        df = None
        for batch in self.iterate_features(feature_ids):
            if df is None:
                df = batch
            else:
                df = pd.concat([df, batch], ignore_index=True)
        return df
    
    def iterate_features(self, feature_ids=None, batch_size=1000):
        if feature_ids is None:
            feature_ids = pd.read_sql_query("SELECT ID FROM ConsolidatedUnknownCompoundItems", con=self.engine)['ID'].tolist()
        for i in range(0, len(feature_ids), batch_size):
            batch_ids = feature_ids[i:i+batch_size]
            if len(batch_ids) == 1:
                where_clause = f"ID = {batch_ids[0]}"
            else:
                where_clause = f"ID IN {tuple(batch_ids)}"
            tab = pd.read_sql_query(f"SELECT * FROM ConsolidatedUnknownCompoundItems WHERE {where_clause}", con=self.engine)
            self._decode_feature_vectors(tab)
            yield tab
            
    def count_features(self):
        result = pd.read_sql_query("SELECT COUNT(*) AS count FROM ConsolidatedUnknownCompoundItems", con=self.engine)
        return result['count'].iloc[0]

    @property
    def feature_ids(self):
        if not hasattr(self, '_features_ids'):
            result = pd.read_sql_query("SELECT ID FROM ConsolidatedUnknownCompoundItems", con=self.engine)
            self._features_ids = result['ID'].tolist()
        return self._features_ids

    """The following properties return chromatogram data and retention time correction curves"""
    @property
    def chromatograms(self):
        if not hasattr(self, '_chromatograms_data'):
            self._chromatograms_data = pd.read_sql_table('ChromatogramPeakItems', con=self.engine)
        return self._chromatograms_data
    
    @property
    def corrected_retention_times(self):
        if not hasattr(self, '_corrected_retention_times_data'):
            self._corrected_retention_times_data = pd.read_sql_table('FileAlignmentCorrectionItems', con=self.engine)
            self._corrected_retention_times_data['OriginalRT'] = self._corrected_retention_times_data['OriginalRT'].apply(lambda blb: decode_retention_times(blb))
            self._corrected_retention_times_data['CorrectedRT'] = self._corrected_retention_times_data['CorrectedRT'].apply(lambda blb: decode_retention_times(blb))
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
            self._chemspider_hits_data['MolStructure'] = self._chemspider_hits_data['MolStructure'].apply(decode_mol_structure)
        return self._chemspider_hits_data
    
    def get_chemspider_search_results_for_feature(self, feature_ids):
        if len(feature_ids) == 1:
            where_clause = f"J.ConsolidatedUnknownCompoundItemsID = {feature_ids[0]}"
        elif len(feature_ids) > 1:
            where_clause = f"J.ConsolidatedUnknownCompoundItemsID IN {tuple(feature_ids)}"
        else:
            raise ValueError("feature_ids must contain at least one ID")
        join = f"""SELECT J.ConsolidatedUnknownCompoundItemsID AS FeatureID, C.ChemSpiderID AS ChemSpiderID, C.Name AS Name, C.Formula AS Formula, C.MolecularWeight AS MolecularWeight, C.InChi AS InChi, C.InChiKey AS InChiKey, C.MolStructure AS Structure, J.DeltaMassInPPM AS DeltaMass, J.MzLogicScore AS Score, J.CompoundMatchStatus AS Status
                        FROM ConsolidatedUnknownCompoundItemsChemSpiderResultItems J JOIN ChemSpiderResultItems C 
                        ON J.ChemSpiderResultItemsChemSpiderID = C.ChemSpiderID
                        WHERE {where_clause}"""
        tab = pd.read_sql_query(join, con=self.engine)
        tab['Structure'] = tab['Structure'].apply(decode_mol_structure)
        return tab
    
    
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
            self._mzcloud_search_results_data['MolStructure'] = self._mzcloud_search_results_data['MolStructure'].apply(decode_mol_structure)
        return self._mzcloud_search_results_data

    @property
    def mzcloud_hits(self):
        if not hasattr(self, '_mzcloud_hits_data'):
            self._mzcloud_hits_data = pd.read_sql_table('MzCloudHitItems', con=self.engine)
            self._mzcloud_hits_data['MolStructure'] = self._mzcloud_hits_data['MolStructure'].apply(decode_mol_structure)
        return self._mzcloud_hits_data
        
    def get_mzcloud_search_results_for_feature(self, feature_ids):
        if len(feature_ids) == 1:
            where_clause = f"J.ConsolidatedUnknownCompoundItemsID = {feature_ids[0]}"
        elif len(feature_ids) > 1:
            where_clause = f"J.ConsolidatedUnknownCompoundItemsID IN {tuple(feature_ids)}"
        else:
            raise ValueError("feature_ids must contain at least one ID")
        join = f"""SELECT J.ConsolidatedUnknownCompoundItemsID AS FeatureID, M.MzCloudId AS MzCloudID, M.KeggId as KeggID, M.Name AS Name, M.Formula AS Formula, M.Mass AS MolecularWeight, M.MolStructure as Structure, J.DeltaMassInPPM AS DeltaMass, J.MzLibraryMatchFactor AS Score, J.Confidence AS Confidence, J.CompoundMatchStatus AS Status
                        FROM ConsolidatedUnknownCompoundItemsMzCloudSearchResultItems J JOIN MzCloudSearchResultItems M 
                        ON J.MzCloudSearchResultItemsID = M.ID
                        WHERE {where_clause}"""
        tab = pd.read_sql_query(join, con=self.engine)
        tab['Structure'] = tab['Structure'].apply(decode_mol_structure)
        return tab
    
    """The following properties return tables that link the features to their best hit ions and associated spectra"""
    @property
    def spectrum_ids(self):
        if not hasattr(self, '_spectrum_ids'):
            result = pd.read_sql_query("SELECT ID FROM MassSpectrumItems", con=self.engine)
            self._spectrum_ids = result['ID'].tolist()
        return self._spectrum_ids
    
    def filter_spectrum_ids(self, msn_level=None, activation_type=None, polarity=None):
        where_clauses = []
        if msn_level is not None:
            where_clauses.append(f"MSOrder = {msn_level}")
        if activation_type is not None:
            where_clauses.append(f"ActivationType = '{activation_type}'")
        if polarity is not None:
            where_clauses.append(f"Polarity = '{polarity}'")
        if len(where_clauses) > 0:
            where_statement = " AND ".join(where_clauses)
            result = pd.read_sql_query(f"SELECT ID FROM MassSpectrumItems WHERE {where_statement}", con=self.engine)
        else:
            result = pd.read_sql_query("SELECT ID FROM MassSpectrumItems", con=self.engine)
        return result['ID'].tolist()
        
    def _decode_spectra(self, spectra_df):
        ndf = spectra_df.apply(lambda row: decode_spectrum(row.Spectrum), axis=1, result_type='expand')
        spectra_df['Metadata'] = ndf[0]
        spectra_df['Precursor'] = ndf[1]
        spectra_df['Spectrum'] = ndf[2]
        
    @property
    def spectra(self):
        if not hasattr(self, '_spectra_data'):
            self._spectra_data = pd.read_sql_table('MassSpectrumItems', con=self.engine)
            self._decode_spectra(self._spectra_data)
        return self._spectra_data
    
    def feature_to_spectra(self, feature_ids=None, spectrum_ids=None):
        where_clauses = []
        if feature_ids is not None:
            where_clauses.append(f"BHI.ConsolidatedUnknownCompoundItemsID IN ({', '.join([str(fid) for fid in feature_ids])})")
        if specturm_ids is not None:
            where_clauses.append(f"BHIMS.MassSpectrumItemsID IN ({', '.join([str(sid) for sid in spectrum_ids])})")
        if len(where_clauses) == 0:
            where_clause = ""
        else:
            where_clause = "WHERE " + " AND ".join(where_clauses)

        Q = f"""SELECT BHI.ConsolidatedUnknownCompoundItemsID AS FeatureID, BHIMS.MassSpectrumItemsID AS SpectrumID
                FROM ConsolidatedUnknownCompoundItemsBestHitIonInstanceItems BHI 
                    JOIN BestHitIonInstanceItemsMassSpectrumItems BHIMS 
                        ON BHI.BestHitIonInstanceItemsWorkflowID = BHIMS.BestHitIonInstanceItemsWorkflowID 
                            AND BHI.BestHitIonInstanceItemsID = BHIMS.BestHitIonInstanceItemsID
                {where_clause}"""
        tab = pd.read_sql_query(Q, con=self.engine)
        return tab
    
    def get_spectra_by_id(self, spectrum_ids, asxml=False):
        if type(spectrum_ids) == int: 
            spectrum_ids = [spectrum_ids]
        if len(spectrum_ids) == 1:
            where_clause = f"MS.ID = {spectrum_ids[0]}"
        elif len(spectrum_ids) > 1:
            where_clause = f"MS.ID IN ({', '.join([str(sid) for sid in spectrum_ids])})"
        else:
            raise ValueError("spectrum_ids must contain at least one ID")
        
        Q = f"""SELECT MS.WorkflowID AS WorkflowID, MS.ID AS SpectrumID, MS.FileID AS FileID, MS.MSOrder AS MSn, MS.Polarity AS Polarity, MS.RetentionTime AS RetentionTime, MS.ResolutionAtMass200 AS Resolution, MS.ActivationType AS ActivationType, MS.ScanType AS ScanType, MS.Ionization AS Ionization, MS.MassAnalyzer AS MassAnalyzer, MS.Spectrum AS Spectrum
                    FROM MassSpectrumItems MS WHERE {where_clause}"""
        
        tab = pd.read_sql_query(Q, con=self.engine)
        if asxml:
            tab['Spectrum'] = tab['Spectrum'].apply(decode_spectrum_to_xml)
        else:
            self._decode_spectra(tab)
        return tab
    
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
    
    def get_compound_spectra(self, feature_ids, asxml=False):
        if len(feature_ids) == 1:
                where_clause = f"BHI.ConsolidatedUnknownCompoundItemsID = {feature_ids[0]}"
        elif len(feature_ids) > 1:
            where_clause = f"BHI.ConsolidatedUnknownCompoundItemsID IN ({', '.join([str(fid) for fid in feature_ids])})"
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
        tab = pd.read_sql_query(Q, con=self.engine)
        if asxml:
            tab['Spectrum'] = tab['Spectrum'].apply(decode_spectrum_to_xml)
        else:
            self._decode_spectra(tab)
        return tab
    
    
    """Utility function to print the schema of a table for debugging purposes"""
    def show_table_schema(self, table_name):
        print(f"Table: {table_name}")
        inspector = inspect(self.engine)
        columns = inspector.get_columns(table_name, schema='main')
        for column in columns:
            print(f"   Column: {column['name']}, Type: {column['type']}")