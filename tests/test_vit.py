import os.path as osp
import unittest

import cv2
import numpy as np
import torch

from videotofaces.encoders.vit import VitEncoderAnime, VitEncoder

class TestVIT(unittest.TestCase):

    def test_vit_anime(self):
        testdir = osp.dirname(osp.realpath(__file__))
        imgs = [cv2.imread(osp.join(testdir, 'images', 'aniface%u.jpg' % i)) for i in [1, 2]]
        model = VitEncoderAnime('cpu', False)
        res = model(imgs)
        self.assertEqual(res.shape, (2, 768))
        np.testing.assert_almost_equal(res[0][100:105], np.array([-0.4530, -2.1694, 0.0624, -0.7991, -0.3798]), decimal=4)
        np.testing.assert_almost_equal(res[1][640:645], np.array([0.3255, -0.6816, -0.1108,  0.2946,  1.7022]), decimal=4)

    def test_vit_face(self):
        testdir = osp.dirname(osp.realpath(__file__))
        imgs = [cv2.imread(osp.join(testdir, 'images', '00%u_0.jpg' % i)) for i in [360, 715]]
        model = VitEncoder('cpu', False)
        res = model(imgs)
        self.assertEqual(res.shape, (2, 512))
        np.testing.assert_almost_equal(res[0][485:490], np.array([1.1649, -0.4840, 1.0156, -0.7108, -0.0953]), decimal=4)
        np.testing.assert_almost_equal(res[1][250:255], np.array([0.9379, -0.4167, -0.8178, 0.4852,  0.5944]), decimal=4)


if __name__ == '__main__':
    unittest.main()