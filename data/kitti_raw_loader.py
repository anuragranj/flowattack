# Taken from https://github.com/ClementPinard/SfmLearner-Pytorch/
import numpy as np
from path import Path
from collections import Counter
from PIL import Image

def imresize(arr, sz):
    height, width = sz
    return np.array(Image.fromarray(arr).resize((width, height), resample=Image.BILINEAR))

class KittiRawLoader(object):
    def __init__(self,
                 dataset_dir,
                 static_frames_file=None,
                 img_height=128,
                 img_width=416,
                 min_speed=2,
                 get_gt=False):
        self.from_speed = static_frames_file is None
        if static_frames_file is not None:
            static_frames_file = Path(static_frames_file)
            self.collect_static_frames(static_frames_file)

        self.dataset_dir = Path(dataset_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.cam_ids = ['02', '03']
        self.date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        self.min_speed = min_speed
        self.get_gt = get_gt
        self.collect_train_folders()

    def collect_static_frames(self, static_frames_file):
        with open(static_frames_file, 'r') as f:
            frames = f.readlines()
        self.static_frames = {}
        for fr in frames:
            if fr == '\n':
                continue
            date, drive, frame_id = fr.split(' ')
            curr_fid = '%.10d' % (np.int(frame_id[:-1]))
            if drive not in self.static_frames.keys():
                self.static_frames[drive] = []
            self.static_frames[drive].append(curr_fid)

    def collect_train_folders(self):
        self.scenes = []
        for date in self.date_list:
            drive_set = (self.dataset_dir/date).dirs()
            for dr in drive_set:
                self.scenes.append(dr)

    def collect_scenes(self, drive):
        train_scenes = []
        for c in self.cam_ids:
            oxts = sorted((drive/'oxts'/'data').files('*.txt'))
            scene_data = {'cid': c, 'dir': drive, 'speed': [], 'frame_id': [], 'rel_path': drive.name + '_' + c}
            for n, f in enumerate(oxts):
                metadata = np.genfromtxt(f)
                speed = metadata[8:11]
                scene_data['speed'].append(speed)
                scene_data['frame_id'].append('{:010d}'.format(n))
            sample = self.load_image(scene_data, 0)
            if sample is None:
                return []
            scene_data['P_rect'] = self.get_P_rect(scene_data, sample[1], sample[2])
            scene_data['intrinsics'] = scene_data['P_rect'][:,:3]

            train_scenes.append(scene_data)
        return train_scenes

    def get_scene_imgs(self, scene_data):
        def construct_sample(scene_data, i, frame_id):
            sample = [self.load_image(scene_data, i)[0], frame_id]
            if self.get_gt:
                sample.append(self.generate_depth_map(scene_data, i))
            return sample

        if self.from_speed:
            cum_speed = np.zeros(3)
            for i, speed in enumerate(scene_data['speed']):
                cum_speed += speed
                speed_mag = np.linalg.norm(cum_speed)
                if speed_mag > self.min_speed:
                    frame_id = scene_data['frame_id'][i]
                    yield construct_sample(scene_data, i, frame_id)
                    cum_speed *= 0
        else:  # from static frame file
            drive = str(scene_data['dir'].name)
            for (i,frame_id) in enumerate(scene_data['frame_id']):
                if (drive not in self.static_frames.keys()) or (frame_id not in self.static_frames[drive]):
                    yield construct_sample(scene_data, i, frame_id)

    def get_P_rect(self, scene_data, zoom_x, zoom_y):
        #print(zoom_x, zoom_y)
        calib_file = scene_data['dir'].parent/'calib_cam_to_cam.txt'

        filedata = self.read_raw_calib_file(calib_file)
        P_rect = np.reshape(filedata['P_rect_' + scene_data['cid']], (3, 4))
        P_rect[0] *= zoom_x
        P_rect[1] *= zoom_y
        return P_rect

    def load_image(self, scene_data, tgt_idx):
        img_file = scene_data['dir']/'image_{}'.format(scene_data['cid'])/'data'/scene_data['frame_id'][tgt_idx]+'.png'
        if not img_file.isfile():
            return None
        img = np.array(Image.open(img_file))
        zoom_y = self.img_height/img.shape[0]
        zoom_x = self.img_width/img.shape[1]
        img = imresize(img, (self.img_height, self.img_width))
        return img, zoom_x, zoom_y

    def read_raw_calib_file(self, filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                        data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                        pass
        return data
