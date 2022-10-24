import os.path as osp
import unittest

import cv2
import numpy as np

from videotofaces.detectors.coord_reg import CoordRegLandmarker


class TestLandmarker(unittest.TestCase):

    def test1(self):
        model = CoordRegLandmarker('cpu')
        testdir = osp.dirname(osp.realpath(__file__))
        imgs = [
            cv2.imread(osp.join(testdir, 'images', '00360_0.jpg')),
            cv2.imread(osp.join(testdir, 'images', '00715_0.jpg')),
        ]
        res = model(imgs)
        np.testing.assert_equal(res[0], np.array([[77, 94], [116, 89], [97, 118], [86, 148], [121, 142]]))
        np.testing.assert_equal(res[1], np.array([[58, 90], [98, 75], [67, 96], [65, 133], [100, 120]]))


if __name__ == '__main__':
    unittest.main()