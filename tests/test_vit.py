import os.path as osp
import unittest

import cv2
import numpy as np

from videotofaces.encoders.vit import AnimeVIT


class TestVIT(unittest.TestCase):

    def test_vit_anime(self):
        testdir = osp.dirname(osp.realpath(__file__))
        paths = [osp.join(testdir, 'images', 'anime_enc_%u.jpg' % el) for el in [1, 2]]
        imgs = [cv2.imread(pt) for pt in paths]
        model = AnimeVIT()
        emb = model(imgs)
        self.assertEqual(emb.shape, (2, 768))
        np.testing.assert_almost_equal(emb[0][100:105], np.array([-0.4530, -2.1694, 0.0624, -0.7991, -0.3798]), decimal=4)
        np.testing.assert_almost_equal(emb[1][640:645], np.array([0.3255, -0.6816, -0.1108,  0.2946,  1.7022]), decimal=4)

  
if __name__ == '__main__':
    unittest.main()