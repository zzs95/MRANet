ANATOMICAL_REGIONS = {
    "right lung": 0,
    "right upper lung zone": 1,
    "right mid lung zone": 2,
    "right lower lung zone": 3,
    "right hilar structures": 4,
    "right apical zone": 5,
    "right costophrenic angle": 6, # x
    "right hemidiaphragm": 7,
    "left lung": 8,
    "left upper lung zone": 9,
    "left mid lung zone": 10,
    "left lower lung zone": 11,
    "left hilar structures": 12,
    "left apical zone": 13,
    "left costophrenic angle": 14, # x
    "left hemidiaphragm": 15,
    "trachea": 16,
    "spine": 17,
    "right clavicle": 18,
    "left clavicle": 19,
    "aortic arch": 20,
    "mediastinum": 21,
    "upper mediastinum": 22,
    "svc": 23,
    "cardiac silhouette": 24,
    "cavoatrial junction": 25, # x
    "right atrium": 26,
    "carina": 27,
    "abdomen": 28,
}
# [53763128, 53783835, 53696374, 53693305, 53827932, 53717431, 53749724, 53704798, 53673185, 53717901, 53668363, 53704431, 53787388, 53734351, 53697306, 53706082, 53679465, 53679460, 53674238, 53749724, 53705230, 53673185, 53678489, 53666614, 53678489]
brown_IMAGE_TO_IGNORE = [572, 825, 709, 1003, 622, 779, 894, 697, 736, 49, 285, 394, 200, 563, 73, 589, 829, 902, 780, 775, 698, 828, 401, 827] # seed 6
COVID_REGIONS = {
    # 'Lines/tubes':[-1],
    'Lungs':[0,1,2,3,4,5,7, 8,9,10,11,12,13,15],
    'Pleura':[0,8],
    'Heart and mediastinum':[16, 20, 21, 22, 23, 24, 26, 27],
    'Bones':[17, 18, 19],
    # 'Others':[-1],
    }
import numpy as np
REGION_IDS = np.unique(np.concatenate([COVID_REGIONS[k] for k in COVID_REGIONS.keys()])).tolist()

REPORT_KEYS_raw = ['Report Text_FINDINGS', 'Report Text_HISTORY', 'Report Text_IMPRESSION' ]

REPORT_KEYS = [
    # 'Report Text_FIND_Lines/tubes', 
               'Report Text_FIND_Lungs', 'Report Text_FIND_Pleura', 'Report Text_FIND_Heart and mediastinum', 'Report Text_FIND_Bones',
            #    'Report Text_FIND_Others', 
            #    'Report Text_HISTORY', 
               'Report Text_IMPRESSION']
CLIN_KEYS = ['clin_feat_1', 'clin_feat_2', 'clin_feat_3', 'clin_feat_4', 'clin_feat_5', 'clin_feat_6', 'clin_feat_7', 'clin_feat_8', 'clin_feat_9', 'clin_feat_10', 'clin_feat_11', 'clin_feat_12', 'clin_feat_13', 'clin_feat_14', 'clin_feat_15', 'clin_feat_16']

