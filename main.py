from roifile import ImagejRoi
import suite2p
import os
import numpy as np
from math import dist


def setup():
    # suite2p input parameters
    ops = {
        # Main settings
        'nplanes': 1,
        'nchannels': 1,
        'functional_chan': 1,
        'tau': 1.0,
        'fs': 30.0,
        'do_bidiphase': False,
        'bidiphase': 0,
        'multiplane_parallel': False,
        'ignore_flyback': [],

        # Output Settings
        'preclassify': 0.0,
        'save_mat': False,
        'save_NWB': False,
        'combined': True,
        'reg_tif': False,
        'reg_tif_chan2': False,
        'aspect': 1.0,
        'delete_bin': False,
        'move_bin': False,

        # Registration
        'do_registration': True,
        'align_by_chan': 1,
        'nimg_init': 300,
        'batch_size': 500,
        'smooth_sigma': 1.15,
        'smooth_sigma_time': 0,
        'maxregshift': 0.1,
        'th_badframes': 1.0,
        'keep_movie_raw': False,
        'two_step_registration': False,

        # Nonrigid
        'nonrigid': True,
        'block_size': [128, 128],
        'snr_thresh': 1.2,
        'maxregshiftNR': 5,

        # 1P
        '1Preg': False,
        'spatial_hp_reg': 42,
        'pre_smooth': 0,
        'spatial_taper': 40,

        #Functional detection
        'roidetect': True,
        'denoise': False,
        'spatial_scale': 0,
        'threshold_scaling': 1.0,
        'max_overlap': 0.75,
        'max_iterations': 20,
        'high_pass': 100,
        'spatial_hp_detect': 25,

        # Anatomical detection
        'anatomical_only': 0,
        'diameter': 0,
        'cellprob_threshold': 0.0,
        'flow_threshold': 1.5,
        'pretrained_model': 'cyto',
        'spatial_hp_cp': 0,

        # Extraction/Neuropil
        'neuropil_extract': True,
        'allow_overlap': False,
        'inner_neuropil_radius': 2,
        'min_neuropil_pixels': 350,

        # Classify/Deconvolve
        'soma_crop': True,
        'spikedetect': True,
        'win_baseline': 60.0,
        'sig_baseline': 10.0,
        'neucoeff': 0.7,

        # Generic
        'look_one_level_down': False,
        'fast_disk': [],
        'mesoscan': False,
        'bruker': False,
        'bruker_bidirectional': False,
        'h5py': [],
        'h5py_key': 'data',
        'nwb_file': '',
        'nwb_driver': '',
        'nwb_series': '',
        'save_path0': '',
        'save_folder': [],
        'subfolders': [],
        'force_sktiff': False,
        'frames_include': -1,

        'subpixel': 10,
        'norm_frames': True,
        'force_refImg': False,
        'pad_fft': False,

        'sparse_mode': True,
        'connected': True,
        'nbinned': 5000,

        'lam_percentile': 50.0,
        'use_builtin_classifier': False,
        'classifier_path': '',
        'chan2_thres': 0.65,
        'baseline': 'maximin',
        'prctile_baseline': 8.0,

        }

    # location of suite2p folder containing motion corrected .bin file
    db = {
        'data_path' : [r"data\\"]
        }

    # load manually labeled rois
    manual_rois = ImagejRoi.fromfile('manual_labels.zip')

    return ops, db, manual_rois


def s2p_loop(ops, db, manual_rois):
    # running suite2p pipeline (should automatically detect existing .bin file)
    output_ops = suite2p.run_s2p(ops, 
                                db)
    # matching ROIs
    matches_bin, matches_roi = compare_rois(output_ops, manual_rois)
    return matches_bin, matches_roi


def compare_rois(output_ops, manual_rois, dist_thres=6):
    """Compare automatically generated ROIs from suite2p with manually labeled ROIs
    created in imagej

    Args:
        output_ops (dict): output parameters from suite2p
        manual_rois (dict): set of manually labeled ROIs from imagej
        dist_thres (int, optional): Threshold for maximal distance in pixels between ROI 
                                    centers that is considered to suffice for matching. 
                                    Defaults to 6.

    Returns:
        manual_rois_found (list): list of booleans indicating of manually labeled ROIs were present in s2p output
        manual_roi_matches (list): list of integers indicating which s2p ROI was matched to each manually labeled ROI
    """

    # load stats and iscell categorization
    stat = np.load(os.path.join(output_ops['save_path'], 'stat.npy'), allow_pickle=True)
    iscell = np.load(os.path.join(output_ops['save_path'], 'iscell.npy'))

    # compute centers
    manual_centers = []
    for manual_roi in manual_rois:
        center = [np.mean([manual_roi.left, manual_roi.right]), 
                  np.mean([manual_roi.top, manual_roi.bottom])]
        manual_centers.append(center)

    s2p_centers = []
    for s2p_roi in stat:
        center = [np.mean([np.min(s2p_roi['xpix']), np.max(s2p_roi['xpix'])]), 
                  np.mean([np.min(s2p_roi['ypix']), np.max(s2p_roi['ypix'])])]
        s2p_centers.append(center)

    # compare centers by computing distance
    manual_rois_found = []
    manual_rois_matches = []
    for manual_center in manual_centers:
        all_dists = np.array([dist(manual_center, x) for x in s2p_centers])
        corres_s2p_roi = np.arange(len(s2p_centers))[all_dists < dist_thres]
        if len(corres_s2p_roi) > 0:
            manual_rois_found.append(True)
            manual_rois_matches.append(corres_s2p_roi[np.argmin(corres_s2p_roi)])
        else:
            manual_rois_found.append(False)
            manual_rois_matches.append(np.nan)

    return manual_rois_found, manual_rois_matches


def main():
    ops, db, manual_rois = setup()
    matches_bin, matches_roi = s2p_loop(ops, db, manual_rois)


if __name__ == '__main__':
    main()