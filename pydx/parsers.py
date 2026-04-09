import base64
import struct
import zipfile
import io
import pandas as pd
import gzip
import numpy as np
from xml.etree import ElementTree


sample_type_codes = {
    0: 'Sample',
    1: 'Unknown',
    2: 'Blank',
    3: 'Quality Control',
    4: 'Identification Only'
}

gap_fill_status_codes = {
    0:	    "Unknown status",
    1:	    "Original ion used",
    2:	    "Unable to fill",
    4:	    "Filled by arbitrary value",
    8:	    "Filled by trace area",
    16:	    "Filled by simulated peak",
    32:	    "Filled by spectrum noise",
    64:	    "Filled by matching ion",
    128:	"Filled by re-detected peak",
    256:	"Imputed by low area value",
    512:	"Imputed by group median",
    1024:	"Imputed by Random Forest",
    2048:	"Skipped"
}

gap_status_codes = {
    1: "No gap",
    2: "Missing ions",
    3: "Full gap"
}

activation_type_codes = { # This is currently a guess, need to verify
    32: "HCD",
    1: "CID"
}

match_status_codes = {
    1: "Unknown",
    2: "No Match",
    3: "Partial Match",
    4: "Full Match",
    5: "Invalid Mass"
}

polarity_codes = {
    1: "Positive",
    2: "Negative"
}

def decode_mol_structure(blb):
    if pd.isnull(blb):
        return None
    return gzip.decompress(blb).decode('utf-8')

def decode_peak_ratings(blb):
    if pd.isnull(blb):
        return None, None
    assert len(blb) % 9 == 0, "Byte length must be a multiple of 9"
    
    ratings_bytes = bytes(b for i, b in enumerate(blb) if i % 9 != 8)
    ratings = list(struct.unpack('<' + 'd' * (len(ratings_bytes) // 8), ratings_bytes))
    flag_bytes = bytes(b for i, b in enumerate(blb) if i % 9 == 8)
    flags = list(struct.unpack('<' + 'B' * len(flag_bytes), flag_bytes))
    return np.array(ratings, dtype=np.float64), np.array(flags, dtype=np.uint8)


""" Example peak XML:
tests/test_parsers.py <?xml version="1.0"?>
<PycoPeakModel xmlns:xsd="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <ApexRT>5.14251207635566</ApexRT>
  <LeftRT>5.0705754833446273</LeftRT>
  <RightRT>5.245777979330482</RightRT>
  <IntensityRange>
    <double>0</double>
    <double>112808.6875</double>
  </IntensityRange>
  <Emphasize>false</Emphasize>
  <Width>0.084733989250721287</Width>
  <Method>Linear</Method>
  <CurvePointsStored>false</CurvePointsStored>
  <LeftBaseline>0</LeftBaseline>
  <RightBaseline>0</RightBaseline>
</PycoPeakModel>"""
def decode_peak_model(blb):
    zf = zipfile.ZipFile(io.BytesIO(blb))
    with zf.open(zf.namelist()[0]) as xml_file:
        xml_data = xml_file.read().decode('utf-8')
    etree = ElementTree.fromstring(xml_data)
    
    pk_model = base64.b64decode(etree.find("Data").text)
    
    zf = zipfile.ZipFile(io.BytesIO(pk_model))
    with zf.open(zf.namelist()[0]) as xml_file:
        xml_data = xml_file.read().decode('utf-8')
    etree = ElementTree.fromstring(xml_data)
    
    low, high = (float(t.text) for t in etree.find('IntensityRange').findall('double'))
    data = [float(etree.find('ApexRT').text), 
            float(etree.find('LeftRT').text),
            float(etree.find('RightRT').text),
            float(etree.find('Width').text), low, high]
    return pd.Series(data, index=["apexRT", "leftRT", "rightRT", "width", "intensityLow", "intensityHigh"])

def decode_peak_areas(blb):
    if pd.isnull(blb):
        return None, None
    assert len(blb) % 9 == 0, "Byte length must be a multiple of 9"
    
    area_bytes = bytes(b for i, b in enumerate(blb) if i % 9 != 8)
    areas = list(struct.unpack('<' + 'd' * (len(area_bytes) // 8), area_bytes))
    flag_bytes = bytes(b for i, b in enumerate(blb) if i % 9 == 8)
    flags = list(struct.unpack('<' + 'B' * len(flag_bytes), flag_bytes))
    return np.array(areas, dtype=np.float64), np.array(flags, dtype=np.uint8)

def decode_gap_fill_status(blb):
    if pd.isnull(blb):
        return None
    # drop every 5th byte (indices 4, 9, 14, ...)
    kept = bytes(b for j, b in enumerate(blb) if (j + 1) % 5 != 0)
    if len(kept) % 4:
        raise ValueError("Remaining byte count is not a multiple of 4")
    return np.array([int.from_bytes(kept[k:k+4], "little", signed=True) for k in range(0, len(kept), 4)], dtype=np.uint16)
    
def decode_gap_status(blb):
    if pd.isnull(blb):
        return None
    kept = bytes(b for j, b in enumerate(blb) if (j + 1) % 5 != 0)
    if len(kept) % 4:
        raise ValueError("Remaining byte count is not a multiple of 4")
    return np.array([int.from_bytes(kept[k:k+4], "little", signed=True) for k in range(0, len(kept), 4)], dtype=np.uint8)

def parse_xml_tag_text(tag):
    if tag is None:
        return None
    return tag.text

def parse_xml_double_list(tag):
    if tag is None:
        return []
    return [float(x.text) for x in tag.findall('double')]

def parse_xml_int_tag(tag):
    if tag is None:
        return None
    return int(tag.text)

def parse_xml_msorder(tag):
    if tag is not None and tag.text.startswith("MS"):
        return int(tag.text[2:])
    return None

def parse_xml_float_tag(tag):
    if tag is None:
        return None
    return float(tag.text)

def parse_scan_metadata(etree, header_tag_name="Header"):
    header_tag = etree.find(header_tag_name)
    
    spectrum_id = parse_xml_int_tag(header_tag.find("SpectrumID"))
    instrument = parse_xml_tag_text(header_tag.find("InstrumentName"))
    low_mz = parse_xml_float_tag(header_tag.find("LowPosition"))
    high_mz = parse_xml_float_tag(header_tag.find("HighPosition"))
    
    scan_event_tag = etree.find("ScanEvent")
    activation_types = parse_xml_tag_text(scan_event_tag.find("ActivationTypes"))
    activation_energies = parse_xml_double_list(scan_event_tag.find("ActivationEnergies"))
    collision_energies = parse_xml_double_list(scan_event_tag.find("SteppedCollisionEnergies"))
    
    ionization_source = parse_xml_tag_text(scan_event_tag.find("IonizationSource"))
    msorder = parse_xml_msorder(scan_event_tag.find("MSOrder"))
    polarity = parse_xml_tag_text(scan_event_tag.find("Polarity"))
    resolution = parse_xml_float_tag(scan_event_tag.find("ResolutionAtMass200"))
    
    return {"spectrum_id": spectrum_id, "instrument": instrument, "low_mz": low_mz, "high_mz": high_mz,
            "activation_types": activation_types, "activation_energies": activation_energies,
            "collision_energies": collision_energies, "ionization_source": ionization_source,
            "msorder": msorder, "polarity": polarity, "resolution": resolution}
    
def peaks_to_df(peak_tags):
    return pd.DataFrame(((float(pk.attrib['X']), float(pk.attrib['Y']), float(pk.attrib['Z']), float(pk.attrib['R']), float(pk.attrib['SN'])) for pk in peak_tags),
                        columns=["mz", "intensity", "Z", "resolution", "signalNoiseRatio"])
    
def decode_precursor_scans(etree):
    precursor_tag = etree.find("PrecursorInfo")
    if precursor_tag is None:
        return None
    
    metadata = parse_scan_metadata(precursor_tag, "SpectrumHeader")
    if metadata['spectrum_id'] == -1:
        return None
    
    monoisotopic_peak_centroids = peaks_to_df(precursor_tag.find('MonoisotopicPeakCentroids').findall('Peak'))
    isotope_cluster_peak_centroids = peaks_to_df(precursor_tag.find('IsotopeClusterPeakCentroids').findall('Peak'))
    metadata.update({"monoisotopic_peak_centroids": monoisotopic_peak_centroids, "isotope_cluster_peak_centroids": isotope_cluster_peak_centroids})
    
    
    return pd.Series(metadata, index=["spectrum_id", "instrument", "low_mz", "high_mz", "activation_types", "activation_energies", 
                                                  "collision_energies", "ionization_source", "msorder", "polarity", "resolution", 
                                                  "monoisotopic_peak_centroids", "isotope_cluster_peak_centroids"])
    
def decode_spectrum_from_xml(xml_data):
    etree = ElementTree.fromstring(xml_data)
    
    metadata = parse_scan_metadata(etree)
    precursor_scans = decode_precursor_scans(etree)
    peak_data = peaks_to_df(etree.find('PeakCentroids').findall('Peak'))
    return metadata, precursor_scans, peak_data

def decode_spectrum(blb):
    if pd.isnull(blb):
        return None
    zf = zipfile.ZipFile(io.BytesIO(blb))
    with zf.open(zf.namelist()[0]) as xml_file:
        xml_data = xml_file.read().decode('utf-8')
    return decode_spectrum_from_xml(xml_data)

def decode_spectrum_to_xml(blb):
    if pd.isnull(blb):
        return None
    zf = zipfile.ZipFile(io.BytesIO(blb))
    with zf.open(zf.namelist()[0]) as xml_file:
        xml_data = xml_file.read().decode('utf-8')
    return xml_data

def decode_retention_times(blb):
    if pd.isnull(blb):
        return None
    assert (len(blb)-4) % 8 == 0, "Struct must be 4 byte integer followed by 8 byte doubles"
    blen = int.from_bytes(blb[:4], "little")
    return np.array(struct.unpack('<' + 'd' * blen, blb[4:]), dtype=np.float64)