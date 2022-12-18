import os.path as osp
import unittest

import cv2
import numpy as np

from videotofaces import Detector, detmodels


class TestSSD(unittest.TestCase):

    def _run_inference(self, detmodels_val):
        model = Detector(detmodels_val)
        testdir = osp.dirname(osp.realpath(__file__))
        im1 = cv2.imread(osp.join(testdir, 'images', 'coco_val2017_000139.jpg'))
        im2 = cv2.imread(osp.join(testdir, 'images', 'coco_val2017_455157.jpg'))
        b, s, l = model([im1, im2])
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        return b, s, l

    def test_ssd_torchvision(self):
        b, s, l = self._run_inference(detmodels.SSD)
        self.assertEqual((b[0].shape, s[0].shape), ((200, 4), (200,)))
        self.assertEqual((b[1].shape, l[1].shape), ((200, 4), (200,)))
        np.testing.assert_almost_equal(b[0][100], np.array([63.50, 300.53, 417.14, 390.15]), decimal=2)
        np.testing.assert_almost_equal(b[0][10], np.array([466.41, 354.58, 640.00, 426.00]), decimal=2)
        np.testing.assert_almost_equal(b[1][20], np.array([388.02, 291.70, 578.63, 466.87]), decimal=2)
        np.testing.assert_almost_equal(b[1][-40], np.array([387.71, 253.64, 640.00, 358.64]), decimal=2)
        np.testing.assert_almost_equal(s[0][24:29], np.array([0.1136, 0.1127, 0.1123, 0.1087, 0.1075]), decimal=4)
        np.testing.assert_almost_equal(s[1][95:100], np.array([0.0534, 0.0530, 0.0530, 0.0529, 0.0527]), decimal=4)
        np.testing.assert_equal(l[0][115:130], np.array([64, 62, 1, 1, 84, 64, 64, 67, 86, 67, 84, 62, 67, 64, 67]))
        np.testing.assert_equal(l[1][45:60], np.array([15, 33, 15, 1, 67, 1, 1, 1, 1, 1, 31, 15, 67, 1, 15]))

    
if __name__ == '__main__':
    unittest.main()