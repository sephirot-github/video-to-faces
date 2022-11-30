import os.path as osp
import unittest

import cv2
import numpy as np

from videotofaces import Detector, detmodels


class TestRCNN(unittest.TestCase):

    def _run_inference(self, detmodels_val):
        model = Detector(detmodels_val)
        testdir = osp.dirname(osp.realpath(__file__))
        im1 = cv2.imread(osp.join(testdir, 'images', 'coco_val2017_000139.jpg'))
        im2 = cv2.imread(osp.join(testdir, 'images', 'coco_val2017_455157.jpg'))
        b, s, l = model([im1, im2])
        self.assertEqual((len(b), len(s), len(l)), (2, 2, 2))
        return b, s, l

    def test_rcnn_torchvision_v1(self):
        b, s, l = self._run_inference(detmodels.FasterRCNN_TorchVision_ResNet50_v1)
        self.assertEqual(b[0].shape, (88, 4))
        self.assertEqual(s[0].shape, (88,))
        self.assertEqual(b[1].shape, (42, 4))
        self.assertEqual(l[1].shape, (42,))
        np.testing.assert_almost_equal(b[0][10], np.array([334.2528, 178.0546, 368.7292, 225.9883]), decimal=4)
        np.testing.assert_almost_equal(b[0][50], np.array([300.8515, 213.4518, 353.6335, 220.7821]), decimal=4)
        np.testing.assert_almost_equal(b[1][4], np.array([93.3499, 309.7274, 475.0378, 543.7596]), decimal=4)
        np.testing.assert_almost_equal(b[1][-10], np.array([208.9738, 258.5793, 226.1942, 311.5817]), decimal=4)
        np.testing.assert_almost_equal(s[0][20:25], np.array([0.6796, 0.6519, 0.6061, 0.5157, 0.5043]), decimal=4)
        np.testing.assert_almost_equal(s[1][5:10], np.array([0.7237, 0.6540, 0.5068, 0.3902, 0.3619]), decimal=4)
        np.testing.assert_equal(l[0][70:80], np.array([67, 67, 51, 67, 62, 64, 86, 62, 86, 62]))
        np.testing.assert_equal(l[1][10:25], np.array([67, 1, 1, 67, 15, 27, 77, 15, 32, 15, 15, 67, 67, 67, 15]))

    def test_rcnn_torchvision_v2(self):
        b, s, l = self._run_inference(detmodels.FasterRCNN_TorchVision_ResNet50_v2)
        self.assertEqual(b[0].shape, (75, 4))
        self.assertEqual(s[0].shape, (75,))
        self.assertEqual(b[1].shape, (46, 4))
        self.assertEqual(l[1].shape, (46,))
        np.testing.assert_almost_equal(b[0][20], np.array([352.6096, 205.6527, 361.4193, 219.8587]), decimal=4)
        np.testing.assert_almost_equal(b[0][60], np.array([143.4233, 276.5009, 181.1353, 286.9147]), decimal=4)
        np.testing.assert_almost_equal(b[1][1], np.array([183.4869, 367.3615, 465.6130, 558.6066]), decimal=4)
        np.testing.assert_almost_equal(b[1][-5], np.array([158.7605, 81.9451, 237.0260, 153.7705]), decimal=4)
        np.testing.assert_almost_equal(s[0][40:45], np.array([0.2162, 0.1905, 0.1892, 0.1671, 0.1590]), decimal=4)
        np.testing.assert_almost_equal(s[1][10:15], np.array([0.6772, 0.6085, 0.4874, 0.3936, 0.3767]), decimal=4)
        np.testing.assert_equal(l[0][25:40], np.array([86, 86, 62, 44, 62, 86, 67, 67, 67, 64, 67, 64, 67, 86, 86]))
        np.testing.assert_equal(l[1][-15:], np.array([15, 73, 18, 67, 84, 31, 15, 15, 15, 15, 2, 15, 1, 15, 84]))

    def test_rcnn_torchvision_mobile_hires(self):
        b, s, l = self._run_inference(detmodels.FasterRCNN_TorchVision_MobileNetV3L_HiRes)
        self.assertEqual(b[0].shape, (80, 4))
        self.assertEqual(s[0].shape, (80,))
        self.assertEqual(b[1].shape, (57, 4))
        self.assertEqual(l[1].shape, (57,))
        np.testing.assert_almost_equal(b[0][24], np.array([489.3871, 170.8812, 518.7584, 282.5359]), decimal=4)
        np.testing.assert_almost_equal(b[0][70], np.array([578.3845, 250.5025, 604.5890, 266.7317]), decimal=4)
        np.testing.assert_almost_equal(b[1][5], np.array([298.9149, 275.8839, 592.3077, 512.6057]), decimal=4)
        np.testing.assert_almost_equal(b[1][-8], np.array([157.6056, 173.1693, 230.4536, 378.7680]), decimal=4)
        np.testing.assert_almost_equal(s[0][5:10], np.array([0.8966, 0.8761, 0.8308, 0.7757, 0.7747]), decimal=4)
        np.testing.assert_almost_equal(s[1][25:30], np.array([0.1595, 0.1541, 0.1459, 0.1360, 0.1359]), decimal=4)
        np.testing.assert_equal(l[0][60:75], np.array([64, 86, 79, 44, 84, 64, 84, 82, 64, 62, 53, 67, 82, 64, 64]))
        np.testing.assert_equal(l[1][:15], np.array([28, 1, 15, 1, 15, 67, 1, 15, 67, 62, 1, 15, 28, 27, 15]))

    def test_rcnn_torchvision_mobile_lores(self):
        b, s, l = self._run_inference(detmodels.FasterRCNN_TorchVision_MobileNetV3L_LoRes)
        self.assertEqual(b[0].shape, (32, 4))
        self.assertEqual(s[0].shape, (32,))
        self.assertEqual(b[1].shape, (15, 4))
        self.assertEqual(l[1].shape, (15,))
        np.testing.assert_almost_equal(b[0][5], np.array([544.0790, 299.3280, 594.0629, 397.1502]), decimal=4)
        np.testing.assert_almost_equal(b[1][0], np.array([213.9087, 131.7052, 448.1617, 298.2360]), decimal=4)
        np.testing.assert_almost_equal(s[0][17:22], np.array([0.1037, 0.1011, 0.0793, 0.0790, 0.0786]), decimal=4)
        np.testing.assert_equal(l[1], np.array([28, 1, 15, 1, 27, 15, 27, 67, 31, 33, 15, 31, 62, 31, 1]))


if __name__ == '__main__':
    unittest.main()