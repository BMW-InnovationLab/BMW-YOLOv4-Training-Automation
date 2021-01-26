import glob
import os
import shutil
import cv2
import numpy as np
from enum import Enum
import utils


class MidgardConverter:
    '''Converts MIDGARD dataset for use with YOLOv4.'''

    class Mode(Enum):
        APPEARANCE_RGB = 0,
        FLOW_UV = 1,
        FLOW_UV_NORMALISED = 2,
        FLOW_RADIAL = 3

    def remove_contents_of_folder(self, folder: str) -> None:
        """Remove all content of a directory

        Args:
            folder (string): the directory to delete
        """
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def get_capture_shape(self) -> tuple:
        """Get the shape of the original image inputs.

        Returns:
            tuple: image shape
        """
        return np.array(cv2.imread(f'{self.img_path}/image_00000.png')).shape

    def get_flow_uv(self) -> np.ndarray:
        """Get the content of the .flo file for the current frame

        Returns:
            np.ndarray: (w, h, 2) array with flow vectors
        """
        flo_path = f'{self.img_path}/output/inference/run.epoch-0-flow-field/{self.i:06d}.flo'
        return utils.read_flow(flo_path)

    def process_image(self, src: str, dst: str) -> None:
        """Processes an image of the dataset and places it in the target directory

        Args:
            src (str): source image path
            dst (str): destination image path
        """
        if self.mode == MidgardConverter.Mode.APPEARANCE_RGB:
            shutil.copy2(src, dst)
        else:
            img = cv2.imread(src)
            img = cv2.resize(img, self.capture_shape)

            if self.mode == MidgardConverter.Mode.FLOW_UV:
                cv2.imwrite(dst, img)
            elif self.mode == MidgardConverter.Mode.FLOW_UV:
                img = np.zeros()
                cv2.imwrite(dst, img)

    def process_annot(self, src: str, dst: str) -> None:
        """Processes an annotation file of the dataset and places it in the target directory

        Args:
            src (str): source annotation file path
            dst (str): destination annotation file path
        """
        with open(dst, 'w') as f:
            f.writelines(self.get_midgard_annotation(src))

    def prepare_sequence(self, sequence: str) -> None:
        """Prepare a sequence of the MIDGARD dataset

        Args:
            sequence (str): which sequence to prepare, for example 'indoor-modern/sports-hall'
        """
        images = glob.glob(f'{self.img_path}/*.png')
        annotations = glob.glob(f'{self.ann_path}/*.csv')
        images.sort()
        annotations.sort()

        for i, (img_src, ann_src) in enumerate(zip(images, annotations)):
            self.process_image(img_src, f'{self.img_dest_path}/{i:06d}.png')
            self.process_annot(ann_src, f'{self.ann_dest_path}/{i:06d}.txt')

    def get_midgard_annotation(self, ann_path: str) -> list:
        """Returns a list of ground truth bounding boxes given an annotation file.

        Args:
            ann_path (str): the annotation .txt file to process

        Returns:
            list: a list of bounding boxes in format '0 {center_x} {center_y} {size_x} {size_y}' with coordinates in range [0, 1]
        """
        lines = []

        with open(ann_path, 'r') as f:
            for line in f.readlines():
                values = [float(x) for x in line.split(',')]
                center = np.array([values[1] + values[3] / 2, values[2] + values[4] / 2]) / self.resolution.astype(np.float)
                size = np.array([values[3], values[4]]) / (2.0 * self.resolution.astype(np.float))
                lines.append(f'0 {center[0]} {center[1]} {size[0]} {size[1]}')

        return lines

    def process(self) -> None:
        """Processes the MIDGARD dataset"""
        channel_options = {
            MidgardConverter.Mode.APPEARANCE_RGB: 3,
            MidgardConverter.Mode.FLOW_UV: 2,
            MidgardConverter.Mode.FLOW_UV_NORMALISED: 2,
            MidgardConverter.Mode.FLOW_RADIAL: 1,
        }

        self.midgard_path = os.environ['MIDGARD_PATH']
        self.dest_path = 'dataset'
        self.img_dest_path = f'{self.dest_path}/images'
        self.ann_dest_path = f'{self.dest_path}/labels/yolo'
        self.sequence = 'indoor-modern/sports-hall'

        self.img_path = f'{self.midgard_path}/{self.sequence}/images'
        self.flo_path = f'{self.img_path}/output/inference/run.epoch-0-flow-field'
        self.ann_path = f'{self.midgard_path}/{self.sequence}/annotation'

        self.mode = MidgardConverter.Mode.APPEARANCE_RGB
        self.channels = channel_options[self.mode]

        self.capture_shape = self.get_capture_shape()
        self.resolution = np.array(self.capture_shape)[:2][::-1]

        self.remove_contents_of_folder(self.img_dest_path)
        self.remove_contents_of_folder(self.ann_dest_path)

        self.prepare_sequence(self.sequence)


converter = MidgardConverter()
converter.process()
