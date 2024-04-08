import os.path as osp
import unittest

import cv2
import numpy as np

from videotofaces.detectors.yolo import RealYOLO


class TestYOLOv3(unittest.TestCase):

    def test_wider(self):
        testdir = osp.dirname(osp.realpath(__file__))
        paths = [osp.join(testdir, 'images', 'irl_det_%u.jpg' % el) for el in [1, 2, 3, 4]]
        imgs = [cv2.imread(pt) for pt in paths]
        model = RealYOLO()
        b, s, _ = model(imgs)
        self.assertEqual((len(b), len(s)), (4, 4))
        res = [np.hstack([b[i], s[i][:, None]]) for i in range(len(b))]
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (20, 5))
        self.assertEqual(res[1].shape, (10, 5))
        self.assertEqual(res[2].shape, (100, 5))
        self.assertEqual(res[3].shape, (93, 5))
        np.testing.assert_almost_equal(res[0][10], np.array([286.4944, 335.9040, 354.3441, 426.0989, 0.9969]), decimal=4)
        np.testing.assert_almost_equal(res[3][25], np.array([460.0020, 143.5856, 493.6367, 193.8361, 0.8309]), decimal=4)


if __name__ == '__main__':
    unittest.main()