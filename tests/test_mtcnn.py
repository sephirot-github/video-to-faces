import os.path as osp
import unittest

import cv2
import numpy as np

from videotofaces.detectors.mtcnn import RealMTCNN


class TestMTCNN(unittest.TestCase):

    def test_main(self):
        testdir = osp.dirname(osp.realpath(__file__))
        paths = [osp.join(testdir, 'images', 'irl_det_%u.jpg' % el) for el in [1, 2, 3, 4]]
        imgs = [cv2.imread(pt) for pt in paths]
        model = RealMTCNN()
        res = model(imgs)
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (15, 5))
        self.assertEqual(res[1].shape, (5, 5))
        self.assertEqual(res[2].shape, (51, 5))
        self.assertEqual(res[3].shape, (28, 5))
        np.testing.assert_almost_equal(res[0][7], np.array([682.8788, 122.9998, 739.7405, 192.9459, 0.9997]), decimal=4)
        np.testing.assert_almost_equal(res[1][-1], np.array([927.6433, 221.3357, 974.1216, 276.0959, 0.9989]), decimal=4)
        np.testing.assert_almost_equal(res[2][44], np.array([162.0115, 53.9863, 173.8801, 67.2544, 0.8978]), decimal=4)
        np.testing.assert_almost_equal(res[3][22], np.array([150.9578, 234.9925, 199.8160, 301.9932, 0.9934]), decimal=4)
        return


if __name__ == '__main__':
    unittest.main()