import os.path as osp
import unittest

import cv2
import numpy as np

from videotofaces.detectors.rcnn import AnimeFRCNN


class TestRCNN(unittest.TestCase):

    def test_rcnn_mmdet_resnet50_animefaces(self):
        testdir = osp.dirname(osp.realpath(__file__))
        paths = [osp.join(testdir, 'images', 'anime_det_%u.jpg' % el) for el in [1, 2, 3, 4]]
        imgs = [cv2.imread(pt) for pt in paths]
        model = AnimeFRCNN()
        b, s, _ = model(imgs)
        self.assertEqual((len(b), len(s)), (4, 4))
        self.assertEqual((b[0].shape, s[0].shape), ((14, 4), (14,)))
        self.assertEqual((b[1].shape, s[1].shape), ((64, 4), (64,)))
        self.assertEqual((b[2].shape, s[2].shape), ((6, 4), (6,)))
        self.assertEqual((b[3].shape, s[3].shape), ((4, 4), (4,)))
        np.testing.assert_almost_equal(b[0][10], np.array([751.9342, 276.2107, 783.7333, 311.8178]), decimal=4)
        np.testing.assert_almost_equal(b[1][50], np.array([329.8422, 381.0872, 367.5275, 419.2162]), decimal=4)
        np.testing.assert_almost_equal(b[2][3], np.array([404.4612, 164.2291, 520.1513, 310.8856]), decimal=4)
        np.testing.assert_almost_equal(b[3][1], np.array([752.1040, 98.5442, 1095.4589, 422.9254]), decimal=4)
        np.testing.assert_almost_equal(s[0][5:10], np.array([0.9873, 0.9793, 0.9594, 0.9509, 0.8711]), decimal=4)
        np.testing.assert_almost_equal(s[1][-5:], np.array([0.6398, 0.5793, 0.5513, 0.4126, 0.2921]), decimal=4)
        np.testing.assert_almost_equal(s[2], np.array([0.9989, 0.9956, 0.7671, 0.7199, 0.6205, 0.0755]), decimal=4)
        np.testing.assert_almost_equal(s[3], np.array([0.9991, 0.9988, 0.9988, 0.9686]), decimal=4)
        

if __name__ == '__main__':
    unittest.main()