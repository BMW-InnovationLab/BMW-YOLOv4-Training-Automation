import glob
import os
import shutil
import cv2
import numpy as np
import utils


class MidgardConverter:
    '''Converts MIDGARD dataset for use with YOLOv4.'''

    def remove_contents_of_folder(self, folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def get_capture_size(self):
        return np.array(cv2.imread(f'{self.img_path}/image_00000.png').shape[:2][::-1])

    def get_flow_uv(self):
        flo_path = f'{self.img_path}/output/inference/run.epoch-0-flow-field/{self.i:06d}.flo'
        return utils.read_flow(flo_path)

    def prepare_sequence(self, sequence):
        images = glob.glob(f'{self.img_path}/*.png')
        annotations = glob.glob(f'{self.ann_path}/*.csv')
        images.sort()
        annotations.sort()

        for i, (img, ann) in enumerate(zip(images, annotations)):
            shutil.copy2(img, f'{self.img_dest_path}/{i:06d}.png')

            ann_path = f'{self.ann_dest_path}/{i:06d}.txt'
            with open(ann_path, 'w') as f:
                f.writelines(self.get_midgard_annotation(ann))

    def get_midgard_annotation(self, ann_path):
        lines = []

        with open(ann_path, 'r') as f:
            for line in f.readlines():
                values = [float(x) for x in line.split(',')]
                center = np.array([values[1] + values[3] / 2, values[2] + values[4] / 2]) / self.capture_size.astype(np.float)
                size = np.array([values[3], values[4]]) / (2.0 * self.capture_size.astype(np.float))
                lines.append(f'0 {center[0]} {center[1]} {size[0]} {size[1]}')

        return lines

    def process(self):
        self.midgard_path = os.environ['MIDGARD_PATH']
        self.dest_path = 'dataset'
        self.img_dest_path = f'{self.dest_path}/images'
        self.ann_dest_path = f'{self.dest_path}/labels/yolo'
        self.sequence = 'indoor-modern/sports-hall'

        self.img_path = f'{self.midgard_path}/{self.sequence}/images'
        self.ann_path = f'{self.midgard_path}/{self.sequence}/annotation'

        self.capture_size = self.get_capture_size()

        self.remove_contents_of_folder(self.img_dest_path)
        self.remove_contents_of_folder(self.ann_dest_path)

        self.prepare_sequence(self.sequence)


converter = MidgardConverter()
converter.process()
