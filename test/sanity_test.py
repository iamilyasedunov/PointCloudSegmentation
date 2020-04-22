import pytest
import sys
import numpy as np
import pandas as pd
import yaml

sys.path.append('./obstacle-detection/')

from pipeline import common

num = '066'
path_to_label = './test/test-files/000'
path_to_scan = './test/test-files/000'

obstacle_lst =  {10: 'car',
                 11: 'bicycle',
                 13: 'bus',
                 15: 'motorcycle',
                 16: 'on-rails',
                 18: 'truck',
                 20: 'other-vehicle',
                 30: 'person',
                 31: 'bicyclist',
                 32: 'motorcyclist',
                 252: 'moving-car',
                 253: 'moving-bicyclist',
                 254: 'moving-person',
                 255: 'moving-motorcyclist',
                 256: 'moving-on-rails',
                 257: 'moving-bus',
                 258: 'moving-truck',
                 259: 'moving-other-vehicle'}

params = {'roi_x_min': 0, 'roi_x_max': 45,
              'roi_y_min': -14, 'roi_y_max': 14,
              'roi_z_min': -2, 'roi_z_max': 1,
              'eps': 0.65, 'min_samples': 6, 'leaf_size': 120, 
              'proc_labels': False}

              
def scan_label_preprocessing(path_to_label, path_to_scan):
    scan = np.fromfile(path_to_scan + num + '.bin', dtype=np.float32)
    scan = scan.reshape((-1, 4))

    label = np.fromfile(path_to_label + num + '.label', dtype=np.uint32)
    label = label.reshape((-1))
    label = [seg & 0xFFFF for seg in label]
    return scan, label

def get_pd_from_scan(scan):
    return pd.DataFrame(scan[:, :3], columns=['x', 'y', 'z'])

def get_pcloud(scan, label):
    cloud = pd.DataFrame(scan[:, :3], columns=['x', 'y', 'z'])
    cloud['seg_id'] = label
    return cloud

class TestClass:
    def test_pcloud_creating(self):
        scan, label = scan_label_preprocessing(path_to_label, path_to_scan)
        pcloud = get_pcloud(scan, label)
        assert len(pcloud) == len(scan) == len(label)
        
    def test_scan_label_preprocessing(self):
        scan, label = scan_label_preprocessing(path_to_label, path_to_scan)
        # test converting file from .bin and .label format to np.array
        assert len(scan) == len(label)

    def test_roi_filter(self):
        scan, _ = scan_label_preprocessing(path_to_label, path_to_scan)
        pcloud_bef = get_pd_from_scan(scan)
        pcloud_aft = common.roi_filter(pcloud_bef,  min_x=params['roi_x_min'], max_x=params['roi_x_max'],
                                                    min_y=params['roi_y_min'], max_y=params['roi_y_max'],
                                                    min_z=params['roi_z_min'], max_z=params['roi_z_max'],
                                                    verbose=False)
        # check that pointcloud shape decreased by roi filter
        assert pcloud_bef.shape > pcloud_aft.shape

    def test_roi_filter_rounded(self):
        scan, label = scan_label_preprocessing(path_to_label, path_to_scan)
        pcloud_bef = get_pcloud(scan, label)
        pcloud_aft = common.roi_filter_rounded(pcloud_bef,  min_x=params['roi_x_min'], max_x=params['roi_x_max'],
                                                    min_y=params['roi_y_min'], max_y=params['roi_y_max'],
                                                    min_z=params['roi_z_min'], max_z=params['roi_z_max'],
                                                    verbose=False)
        # check that pointcloud shape decreased by roi filter rounded
        assert pcloud_bef.shape > pcloud_aft.shape

        
