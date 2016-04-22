import logging
import os
import traceback

# Python 2/3 compatibility
import six
import numpy as np

import auroraplot as ap
from auroraplot.magdata import MagData
from auroraplot.magdata import load_iaga_2000

logger = logging.getLogger(__name__)

ftp_hostname = 'ftp.intermagnet.org'

def load_iaga_2002(file_name, archive_data, 
                      project, site, data_type, channels, start_time, 
                      end_time, **kwargs):
    assert data_type == 'MagData', 'Illegal data_type'
    iaga = load_iaga_2000(file_name)
    data = []
    for c in channels:
        iaga_col_name = site.upper() + c.upper()
        n = iaga['column_number'][iaga_col_name]
        data.append([float('nan') if x == '99999.00' else float(x) 
                     for x in iaga['data'][n]])
    
    data = np.array(data) * 1e-9
    r = MagData(project=project,
                site=site,
                channels=channels,
                start_time=start_time,
                end_time=end_time,
                sample_start_time=iaga['sample_time'], 
                sample_end_time=iaga['sample_time'] + \
                    archive_data['nominal_cadence'],
                integration_interval=None,
                nominal_cadence=archive_data['nominal_cadence'],
                data=data,
                units=archive_data['units'],
                sort=False)
    return r



sites = {
    'AAE': {
        'location': 'Addis Ababa, Ethiopia',
        'latitude': 9.035,
        'longitude': 38.766,
        'elevation': 2441.0,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'AAU (Ethiopia) / IPGP (France)',
        },
    'ABG': {
        'location': 'Alibag, India',
        'latitude': 18.638,
        'longitude': 72.872,
        'elevation': 7,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'Indian Institute of Geomagnetism',
        },
    'ABK': {
        'location': 'Abisko, Sweden',
        'latitude': 68.400,
        'longitude': 18.800,
        'reported': 'XYZF',
        'copyright': 'Sveriges geologiska undersokning',
        },
    'AIA': {
        'location': 'Vernadsky, Argentine Islands, Antarctica',
        'latitude': -65.300,
        'longitude': 295.700,
        'reported': 'XYZF',
        'copyright': 'Lviv Centre of Institute of Space Research',
        },
    'ALE': {
        'location': 'Alert, Canada',
        'latitude': 82.497,
        'longitude': 297.647,
        'elevation': 60.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },
    'AMS': {
        'location': 'Martin de Vivies, Amsterdam Island, Antarctica',
        'latitude': -37.796,
        'longitude': 77.574,
        'elevation': 50,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'EOST',
        },
    'ARS': {
        'location': 'Arti, Russia',
        'latitude': 59.400,
        'longitude': 58.600,
        'reported': 'HDZF',
        'copyright': 'Institute of Geophysics, Russia',
        },
    'ASC': {
        'location': 'Ascension Island',
        'latitude': -07.950,
        'longitude': 345.617,
        'elevation': 177,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'British Geological Survey',        
    },
    'BDV': {
        'location': 'Budkov, Czech Republic',
        'latitude': 49.100,
        'longitude': 14.000,
        'reported': 'XYZF',
        'copyright': 'Geofyzikalni ustav AV',
        },
    'BEL': {
        'location': 'Belsk, Poland',
        'latitude': 51.800,
        'longitude': 20.800,
        'reported': 'XYZF',
        'copyright': 'Polish Academy of Sciences',
        },
    'BLC': {
        'location': 'Baker Lake, Canada',
        'latitude': 64.318,
        'longitude': 263.988,
        'elevation': 30.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },
    'BMT': {
        'location': 'Beijing Ming Tombs, China',
        'latitude': 40.300,
        'longitude': 116.200,
        'elevation': 183,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'Chinese Academy of Sciences',
        },
    'BOU': {
        'location': 'Boulder, USA',
        'latitude': 40.137,
        'longitude': 254.764,
        'elevation': 1682,
        'reported': 'HDZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'United States Geological Survey (USGS)',
        },
    'BRD': {
        'location': 'Brandon, Canada',
        'latitude': 49.870,
        'longitude': 260.026,
        'elevation': 30.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },
    'BRW': {
        'location': 'Barrow, USA',
        'latitude': 71.322,
        'longitude': 203.378,
        'elevation': 12,
        'reported': 'HDZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'United States Geological Survey (USGS)',
        },
    'BSL': {
        'location': 'Stennis Space Center, USA',
        'latitude': 30.350,
        'longitude': 270.365,
        'elevation': 8,
        'reported': 'HDZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'United States Geological Survey (USGS)',
        },
    'CBB': {
        'location': 'Cambridge Bay, Canada',
        'latitude': 69.123,
        'longitude': 254.969,
        'elevation': 20.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },
    'CLF': {
        'location': 'Chambon la Foret, France',
        'latitude': 48.025,
        'longitude': 2.260,
        'elevation': 145.0,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'IPGP',
        },
    'CMO': {
        'location': 'College, USA',
        'latitude': 64.874,
        'longitude': 212.140,
        'elevation': 197,
        'reported': 'HDZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'United States Geological Survey (USGS)',
        },
    'CYG': {
        'location': 'Cheongyang, S. Korea',
        'latitude': 36.370,
        'longitude': 126.854,
        'elevation': 165,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZ',
        'copyright': 'Korea Meteorological Administration',
        },
    'CZT': {
        'location': 'Port Alfred, Crozet Islands, Antarctica',
        'latitude': -46.431,
        'longitude': 51.860,
        'elevation': 160,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'EOST',
        },
    'DED': {
        'location': 'Deadhorse, USA',
        'latitude': 70.356,
        'longitude': 211.207,
        'elevation': 10,
        'reported': 'HDZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'United States Geological Survey (USGS)',
        },
    'DLT': {
        'location': 'Da Lat, Vietnam',
        'latitude': 11.945,
        'longitude': 108.482,
        'elevation': 1583.0,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'VNCST (Vietnam) / IPGP (France)',
        },
    'DMC': {
        'location': 'DomeC, Concordia, Antarctica',
        'latitude': -75.250,
        'longitude': 124.167,
        'elevation': 3250,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'EOST',
        },
    'DOU': {
        'location': 'Dourbes, Belgium',
        'latitude': 50.100,
        'longitude': 4.600,
        'reported': 'HDZF',
        'copyright': 'Royal Meteorological Institute of Belgium',
        },
    'DRV': {
        'location': 'Dumont d\'Urville, Terre Adelie, Antarctica',
        'latitude': -66.655,
        'longitude': 140.007,
        'elevation': 30,
        'reported': 'XYZF',
        'sensor_orientation': 'XYZF',
        'copyright': 'EOST',
        },
    'EBR': {
        'location': 'Ebre, Spain',
        'latitude': 40.957,
        'longitude': 0.333,
        'elevation': 532,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'Observatori de l\'Ebre',
        },
    'EUA': {
        'location': 'Eureka, Canada',
        'latitude': 80.000,
        'longitude': 274.100,
        'elevation': 10.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },
    'FCC': {
        'location': 'Fort Churchill, Canada',
        'latitude': 58.759,
        'longitude': 265.912,
        'elevation': 15.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },
    'FRD': {
        'location': 'Fredericksburg, USA',
        'latitude': 38.205,
        'longitude': 282.627,
        'elevation': 69,
        'reported': 'HDZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'United States Geological Survey (USGS)',
        },
    'FRN': {
        'location': 'Fresno, USA',
        'latitude': 37.091,
        'longitude': 240.282,
        'elevation': 331,
        'reported': 'HDZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'United States Geological Survey (USGS)',
        },
    'FUR': {
        'location': 'Furstenfeldenbruck, Germany',
        'latitude': 48.200,
        'longitude': 11.300,
        'reported': 'XYZF',
        'copyright': 'Ludwig-Maximilians-Universitat Munchen',
        },
    'GAN': {
        'location': 'Gan Int. Airpt, Maldives',
        'latitude': -1.000,
        'longitude': 73.200,
        'reported': 'HDZF',
        'copyright': 'ETH Zurich',
        },
    'GCK': {
        'location': 'Grocka, Serbia',
        'latitude': 44.600,
        'longitude': 20.800,
        'reported': 'XYZF',
        'copyright': 'Geomagnetic College, Grocka',
        },
    'GUA': {
        'location': 'Guam',
        'latitude': 13.588,
        'longitude': 144.867,
        'elevation': 140,
        'reported': 'HDZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'United States Geological Survey (USGS)',
        },
    'GUI': {
        'location': 'Guimar-Tenerife',
        'latitude': 28.317,
        'longitude': 343.560,
        'elevation': 868,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'Instituto Geografico Nacional (Spain)',
        },
    'HER': {
        'location': 'Hermanus, South Africa',
        'latitude': -34.425,
        'longitude': 19.225,
        'elevation': 26,
        'reported': 'XYZG',
        'sensor_orientation': 'HDZF',
        'copyright': 'National Research Foundation',
        },
    'HLP': {
        'location': 'Hel Observatory, Poland',
        'latitude': 54.600,
        'longitude': 18.800,
        'reported': 'XYZF',
        'copyright': 'Polish Academy of Sciences',
        },
    'HON': {
        'location': 'Honolulu, USA',
        'latitude': 21.316,
        'longitude': 202.000,
        'elevation': 4,
        'reported': 'HDZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'United States Geological Survey (USGS)',
        },
    'HRB': {
        'location': 'Hurbanovo, Slovakia',
        'latitude': 47.867,
        'longitude': 18.183,
        'elevation': 112,
        'reported': 'XYZF',
        'sensor_orientation': 'XYZ',
        'copyright': 'GPI, Slovak Academy of Sciences (Slovakia)',
        },
    'HRN': {
        'location': 'Hornsund, Svalbard',
        'latitude': 77.000,
        'longitude': 15.500,
        'reported': 'XYZF',
        'copyright': 'Polish Academy of Sciences',
        },
    'HUA': {
        'location': 'Huancayo, Perua',
        'latitude': -12.100,
        'longitude': 284.600,
        'reported': 'XYZF',
        'copyright': 'Instituto Geofisico del Peru',
        },
    'IPM': {
        'location': 'Isla de Pascua, Chile',
        'latitude': -27.171,
        'longitude': 250.59,
        'elevation': 82.8,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'DMC (Chile) / IPGP (France)',
        },
    'IQA': {
        'location': 'Iqaluit, Canada',
        'latitude': 63.753,
        'longitude': 291.482,
        'elevation': 67.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },
    'IRT': {
        'location': 'Irkutsk (Patrony), Russia',
        'latitude': 52.200,
        'longitude': 104.500,
        'reported': 'HDZF',
        'copyright': 'Institute of Solar-Terrestrial Physics, Radio',
        },
    'JAI': {
        'location': 'Jaipur, India',
        'latitude': 26.917,
        'longitude': 75.800,
        'elevation': 0,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'Indian Institute of Geomagnetism',
        },
    'KOU': {
        'location': 'Kourou, French Guiana',
        'latitude': 5.210,
        'longitude': 307.269,
        'elevation': 10.0,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'IPGP (France)',
        },
    'LYC': {
        'location': 'Lycksele, Sweden',
        'latitude': 64.600,
        'longitude': 18.700,
        'reported': 'XYZF',
        'copyright': 'Sveriges geologiska undersokning',
        },
    'MEA': {
        'location': 'Meanook, Canada',
        'latitude': 54.616,
        'longitude': 246.653,
        'elevation': 700.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },
    'NCK': {
        'location': 'Nagcyenk, Hungary',
        'latitude': 47.600,
        'longitude': 16.700,
        'reported': 'XYZF',
        'copyright': 'Hungarian Academy of Sciences',
        },
    'NEW': {
        'location': 'Newport, USA',
        'latitude': 48.265,
        'longitude': 242.878,
        'elevation': 770,
        'reported': 'HDZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'United States Geological Survey (USGS)',
        },
    'NVS': {
        'location': 'Novosibirsk, Russia',
        'latitude': 54.800,
        'longitude': 83.200,
        'reported': 'XYZF',
        'copyright': 'Russian Academy of Sciences',
        },
    'ORC': {
        'location': 'Base Orcadas, Argentina',
        'latitude': -60.700,
        'longitude': 44.700,
        'reported': 'HDZF',
        'copyright': 'Argentine Met Service',
        },
    'OTT': {
        'location': 'Ottawa, Canada',
        'latitude': 45.403,
        'longitude': 284.448,
        'elevation': 75.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },
    'PAF': {
        'location': 'Port-aux-Francais, Kerguelen Islands',
        'latitude': -49.353,
        'longitude': 70.262,
        'elevation': 35,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'EOST',
        },
    'PIL': {
        'location': 'Pilar, Argentina',
        'latitude': -31.700,
        'longitude': 296.100,
        'reported': 'HDZF',
        'copyright': 'INTERMAGNET Edinburgh GIN',
        },
    'PPT': {
        'location': 'Pamatai',
        'latitude': -17.567,
        'longitude': 210.426,
        'elevation': 357.0,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'IPGP (France)',
        },
    'RES': {
        'location': 'Resolute Bay',
        'latitude': 74.690,
        'longitude': 265.105,
        'elevation': 30.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },
    'SFS': {
        'location': 'San Fernando, Spain',
        'latitude': 36.70,
        'longitude': 354.10,
        'elevation': 111.0,
        'reported': 'HDZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'Real Observatorio de la Armada (Spain)',
        },
    'SHU': {
        'location': 'Shumagin, USA',
        'latitude': 55.348,
        'longitude': 199.538,
        'elevation': 80,
        'reported': 'HDZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'United States Geological Survey (USGS)',
        },
    'SIT': {
        'location': 'Sitka, USA',
        'latitude': 57.058,
        'longitude': 224.674,
        'elevation': 24,
        'reported': 'HDZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'United States Geological Survey (USGS)',
        },
    'SJG': {
        'location': 'San Juan',
        'latitude': 18.113,
        'longitude': 293.849,
        'elevation': 424,
        'reported': 'HDZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'United States Geological Survey (USGS)',
        },
    'SNK': {
        'location': 'Sanikiluaq, Canada',
        'latitude': 56.500,
        'longitude': 280.800,
        'elevation': 20.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },
    'SOD': {
        'location': 'Sodankyla, Finland',
        'latitude': 67.400,
        'longitude': 26.600,
        'reported': 'XYZF',
        'copyright': 'Sodankylan geofysiikan observatorio',
        },
    'SPT': {
        'location': 'San Pablo-Toledo, Spain',
        'latitude': 39.550,
        'longitude': 355.650,
        'elevation': 922,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'Instituto Geografico Nacional (Spain)',
        },
    'STJ': {
        'location': 'St Johns, Canada',
        'latitude': 47.595,
        'longitude': 307.323,
        'elevation': 100.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },
    'TAM': {
        'location': 'Tamanrasset, Algeria',
        'latitude': 22.792,
        'longitude': 5.530,
        'elevation': 1373.0,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'CRAAG/IPGP (Algeria)',
        },
    'TEO': {
        'location': 'Teoloyucan, Mexico',
        'latitude': 19.800,
        'longitude': 260.800,
        'reported': 'HDZF',
        'copyright': 'Instituto de Geofisica U.N.A.M Ciudad Univers',
        },
    'TUC': {
        'location': 'Tucson, USA',
        'latitude': 32.174,
        'longitude': 249.267,
        'elevation': 946,
        'reported': 'HDZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'United States Geological Survey (USGS)',
        },
    'UPS': {
        'location': 'Uppsala, Sweden',
        'latitude': 59.900,
        'longitude': 17.400,
        'reported': 'XYZF',
        'copyright': 'Sveriges geologiska undersokning',
        },
    'VIC': {
        'location': 'Victoria, Canada',
        'latitude': 48.520,
        'longitude': 236.580,
        'elevation': 187.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },
    'VSS': {
        'location': 'Vassouras, Brazil',
        'latitude': -22.40,
        'longitude': 316.35,
        'elevation': 457,
        'reported': 'XYZF',
        'sensor_orientation': 'HDZF',
        'copyright': 'Observatorio National Rio de Janeiro - Brazil',
        },
    'YKC': {
        'location': 'Yellowknife, Canada',
        'latitude': 62.480,
        'longitude': 245.518,
        'elevation': 198.000,
        'reported': 'XYZF',
        'copyright': 'Geological Survey of Canada (GSC)',
        },


    }


default_site_values = {
    'elevation': np.nan,
    'start_time': None,
    'end_time': None,
    }

for sa in sites:
    s = sites[sa]
    for k in default_site_values:
        if k not in s:
            s[k] = default_site_values[k]
    
    if 'data_types' not in s:
        s['data_types'] = {}

    if 'MagData' not in s['data_types']:
        s['data_types']['MagData'] = {}
    
    if 'preliminary' not in s['data_types']['MagData']:
        s['data_types']['MagData']['preliminary'] = {
            'channels': list(s['reported']),
            'path': 'ftp://' + ftp_hostname \
                + '/preliminary/%Y/%m/IAGA2002/' \
                + sa.lower() + '%Y%m%dvmin.min',
            'duration': np.timedelta64(24, 'h'),
            'load_converter': load_iaga_2002,
            'nominal_cadence': np.timedelta64(60000000, 'us'),
            'units': 'T',
           }

    if 'definitive_minute' not in s['data_types']['MagData']:
        s['data_types']['MagData']['definitive_minute'] = {
            'channels': ['X', 'Y', 'Z', 'F'],
            'path': 'ftp://' + ftp_hostname \
                + '/minute/definitive/IAGA2002/%Y/%m/' \
                + sa.lower() + '%Y%m%ddmin.min.gz',
            'duration': np.timedelta64(24, 'h'),
            'load_converter': load_iaga_2002,
            'nominal_cadence': np.timedelta64(60, 's'),
            'units': 'T',
           }
        
ap.add_project('INTERMAGNET', sites)
