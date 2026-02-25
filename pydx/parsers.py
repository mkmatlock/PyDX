import base64
import struct
import zipfile
import io
import pandas as pd
import gzip
import numpy as np
from xml.etree import ElementTree

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

def decode_mol_structure(blb):
    return gzip.decompress(blb).decode('utf-8')

def decode_peak_ratings(blb):
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
    assert len(blb) % 9 == 0, "Byte length must be a multiple of 9"
    
    area_bytes = bytes(b for i, b in enumerate(blb) if i % 9 != 8)
    areas = list(struct.unpack('<' + 'd' * (len(area_bytes) // 8), area_bytes))
    flag_bytes = bytes(b for i, b in enumerate(blb) if i % 9 == 8)
    flags = list(struct.unpack('<' + 'B' * len(flag_bytes), flag_bytes))
    return np.array(areas, dtype=np.float64), np.array(flags, dtype=np.uint8)

def decode_gap_fill_status(blb):
    # drop every 5th byte (indices 4, 9, 14, ...)
    kept = bytes(b for j, b in enumerate(blb) if (j + 1) % 5 != 0)
    if len(kept) % 4:
        raise ValueError("Remaining byte count is not a multiple of 4")
    return np.array([int.from_bytes(kept[k:k+4], "little", signed=True) for k in range(0, len(kept), 4)], dtype=np.uint16)

def decode_spectrum(blb):
    zf = zipfile.ZipFile(io.BytesIO(blb))
    with zf.open(zf.namelist()[0]) as xml_file:
        xml_data = xml_file.read().decode('utf-8')
    etree = ElementTree.fromstring(xml_data)
    
    return pd.DataFrame(((float(pk.attrib['X']), float(pk.attrib['Y']), float(pk.attrib['Z']), float(pk.attrib['R']), float(pk.attrib['SN'])) for pk in etree.find('PeakCentroids').findall('Peak')),
                        columns=["mz", "intensity", "Z", "resolution", "signalNoiseRatio"])
    
def decode_gap_status(blb):
    kept = bytes(b for j, b in enumerate(blb) if (j + 1) % 5 != 0)
    if len(kept) % 4:
        raise ValueError("Remaining byte count is not a multiple of 4")
    return np.array([int.from_bytes(kept[k:k+4], "little", signed=True) for k in range(0, len(kept), 4)], dtype=np.uint8)

def decode_retention_times(blb):
    assert (len(blb)-4) % 8 == 0, "Struct must be 4 byte integer followed by 8 byte doubles"
    blen = int.from_bytes(blb[:4], "little")
    return np.array(struct.unpack('<' + 'd' * blen, blb[4:]), dtype=np.float64)