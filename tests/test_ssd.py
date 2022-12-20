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

    def test_ssd_300_torchvision(self):
        b, s, l = self._run_inference(detmodels.SSD_300_TorchVision)
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

    def test_ssd_300_mmdetection(self):
        b, s, l = self._run_inference(detmodels.SSD_300_MMDetection)
        self.assertEqual((b[0].shape, s[0].shape), ((200, 4), (200,)))
        self.assertEqual((b[1].shape, l[1].shape), ((200, 4), (200,)))
        np.testing.assert_almost_equal(b[0][100], np.array([425.72, 264.66, 439.83, 304.39]), decimal=2)
        np.testing.assert_almost_equal(b[0][25], np.array([279.50, 219.78, 431.77, 256.41]), decimal=2)
        np.testing.assert_almost_equal(b[1][1], np.array([161.43, 164.21, 338.99, 477.41]), decimal=2)
        np.testing.assert_almost_equal(b[1][-40], np.array([124.60, 116.47, 175.72, 200.03]), decimal=2)
        np.testing.assert_almost_equal(s[0][26:31], np.array([0.1258, 0.1240, 0.1222, 0.1205, 0.1163]), decimal=4)
        np.testing.assert_almost_equal(s[1][80:85], np.array([0.0558, 0.0556, 0.0550, 0.0546, 0.0545]), decimal=4)
        np.testing.assert_equal(l[0][71:86], np.array([56, 56, 0, 58, 60, 56, 58, 56, 56, 73, 73, 73, 58, 72, 0]))
        np.testing.assert_equal(l[1][50:65], np.array([25, 25, 25, 73, 25, 60, 0, 56, 24, 56, 60, 0, 73, 56, 25]))

    def test_ssd_512_mmdetection(self):
        b, s, l = self._run_inference(detmodels.SSD_512_MMDetection)
        self.assertEqual((b[0].shape, s[0].shape), ((200, 4), (200,)))
        self.assertEqual((b[1].shape, l[1].shape), ((200, 4), (200,)))
        np.testing.assert_almost_equal(b[0][75], np.array([275.15, 211.36, 372.67, 231.54]), decimal=2)
        np.testing.assert_almost_equal(b[0][18], np.array([327.60, 220.27, 371.37, 309.70]), decimal=2)
        np.testing.assert_almost_equal(b[1][5], np.array([130.33, 198.97, 251.73, 498.45]), decimal=2)
        np.testing.assert_almost_equal(b[1][-66], np.array([536.68, 460.75, 616.41, 505.82]), decimal=2)
        np.testing.assert_almost_equal(s[0][12:17], np.array([0.1847, 0.1794, 0.1740, 0.1729, 0.1543]), decimal=4)
        np.testing.assert_almost_equal(s[1][44:49], np.array([0.0771, 0.0764, 0.0760, 0.0756, 0.0756]), decimal=4)
        np.testing.assert_equal(l[0][116:131], np.array([0, 73, 58, 60, 56, 56, 0, 56, 60, 56, 60, 56, 73, 60, 56]))
        np.testing.assert_equal(l[1][32:47], np.array([0, 13, 73, 0, 13, 60, 73, 13, 13, 13, 73, 13, 0, 13, 13]))

    
if __name__ == '__main__':
    unittest.main()