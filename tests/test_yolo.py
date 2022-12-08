import os.path as osp
import unittest

import cv2
import numpy as np

from videotofaces import Detector, detmodels


class TestYOLOv3(unittest.TestCase):

    def test_anime(self):
        model = Detector(detmodels.YOLOv3_Anime)
        testdir = osp.dirname(osp.realpath(__file__))
        imgs = [cv2.imread(osp.join(testdir, 'images', 'anime_det_%u.jpg' % i)) for i in [1, 2, 3, 4]]
        b, s, _ = model(imgs)
        res = [np.hstack([b[i], s[i][:, None]]) for i in range(len(b))]
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (12, 5))
        self.assertEqual(res[1].shape, (69, 5))
        self.assertEqual(res[2].shape, (8, 5))
        self.assertEqual(res[3].shape, (4, 5))
        np.testing.assert_almost_equal(res[0][4], np.array([614.6, 298.6086, 656.0338, 337.2882, 0.5093]), decimal=4)
        np.testing.assert_almost_equal(res[1][3], np.array([461.1025, 230.8962, 500.1655, 269.5829, 0.9740]), decimal=4)
        np.testing.assert_almost_equal(res[2][2], np.array([397.1730, 155.0277, 520.8693, 325.2291, 0.9882]), decimal=4)
        np.testing.assert_almost_equal(res[3][1], np.array([439.94296, 22.34629, 538.27466, 119.58426, 0.99201]), decimal=4)

    def test_irl(self):
        model = Detector(detmodels.YOLOv3_Wider)
        testdir = osp.dirname(osp.realpath(__file__))
        imgs = [cv2.imread(osp.join(testdir, 'images', 'irl_det_%u.jpg' % i)) for i in [1, 2, 3, 4]]
        b, s, _ = model(imgs)
        res = [np.hstack([b[i], s[i][:, None]]) for i in range(len(b))]
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (20, 5))
        self.assertEqual(res[1].shape, (10, 5))
        self.assertEqual(res[2].shape, (100, 5))
        self.assertEqual(res[3].shape, (93, 5))
        np.testing.assert_almost_equal(res[0][10], np.array([286.4944, 335.9040, 354.3441, 426.0989, 0.9969]), decimal=4)
        np.testing.assert_almost_equal(res[3][25], np.array([460.0020, 143.5856, 493.6367, 193.8361, 0.8309]), decimal=4)

    def test_coco(self):
        model = Detector(detmodels.YOLOv3_COCO)
        testdir = osp.dirname(osp.realpath(__file__))
        im1 = cv2.imread(osp.join(testdir, 'images', 'coco_val2017_000139.jpg'))
        im2 = cv2.imread(osp.join(testdir, 'images', 'coco_val2017_455157.jpg'))
        b, s, l = model([im1, im2])
        self.assertEqual([len(b), len(s), len(l)], [2, 2, 2])
        self.assertEqual(b[0].shape, (100, 4))
        self.assertEqual(s[0].shape, (100,))
        self.assertEqual(b[1].shape, (34, 4))
        self.assertEqual(s[1].shape, (34,))
        np.testing.assert_almost_equal(b[0][15], np.array([449.3380, 121.4010, 461.3889, 141.6084]), decimal=4)
        np.testing.assert_almost_equal(b[1][2], np.array([162.6238, 157.5956, 351.1005, 415.0244]), decimal=4)
        np.testing.assert_almost_equal(s[0][40:45], np.array([0.0249, 0.0228, 0.0222, 0.0216, 0.0187]), decimal=4)
        np.testing.assert_equal(l[1][14:29], np.array([13, 73, 26, 13, 26, 0, 13, 13, 13, 13, 13, 13, 13, 73, 63]))
        return


if __name__ == '__main__':
    unittest.main()