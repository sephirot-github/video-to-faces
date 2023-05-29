import os.path as osp
import unittest

import cv2
import numpy as np
import PIL.Image
import torch

from videotofaces.encoders.vit import VitEncoderAnime, VitEncoder, VitClip

class TestVIT(unittest.TestCase):

    def test_vit_anime(self):
        testdir = osp.dirname(osp.realpath(__file__))
        imgs = [cv2.imread(osp.join(testdir, 'images', 'aniface%u.jpg' % i)) for i in [1, 2]]
        model = VitEncoderAnime('cpu', False)
        res = model(imgs)
        self.assertEqual(res.shape, (2, 768))
        np.testing.assert_almost_equal(res[0][100:105], np.array([-0.4530, -2.1694, 0.0624, -0.7991, -0.3798]), decimal=4)
        np.testing.assert_almost_equal(res[1][640:645], np.array([0.3255, -0.6816, -0.1108,  0.2946,  1.7022]), decimal=4)
        # and large one
        # model = VitEncoderAnime('cpu', True)

    def test_vit_face(self):
        testdir = osp.dirname(osp.realpath(__file__))
        imgs = [cv2.imread(osp.join(testdir, 'images', '00%u_0.jpg' % i)) for i in [360, 715]]
        model = VitEncoder('cpu', False)
        res = model(imgs)
        self.assertEqual(res.shape, (2, 512))
        np.testing.assert_almost_equal(res[0][485:490], np.array([1.1649, -0.4840, 1.0156, -0.7108, -0.0953]), decimal=4)
        np.testing.assert_almost_equal(res[1][250:255], np.array([0.9379, -0.4167, -0.8178, 0.4852,  0.5944]), decimal=4)

    def test_vit_clip_similarity(self):
        # images are taken from this question on similarity search: https://stackoverflow.com/a/71567609
        testdir = osp.dirname(osp.realpath(__file__))
        names = ['bear1', 'bear2', 'cat1', 'cat1copy', 'cat2', 'city1', 'city2']
        paths = [osp.join(testdir, 'images', 'sml_%s.jpg' % nm) for nm in names]
        imgs = [PIL.Image.open(pt) for pt in paths]
        model = VitClip('cpu', 'B-32')
        emb = model(imgs)
        self.assertEqual(emb.shape, (7, 512))
        np.testing.assert_almost_equal(emb[0][:8],      np.array([ 0.2963, -0.3522,  0.1775,  0.4773, -0.1679, -0.1457, -0.0195,  0.4109]), decimal=4)
        np.testing.assert_almost_equal(emb[1][100:108], np.array([-0.1674, -0.1948,  0.1851,  0.0471, -0.2797,  0.3056, -1.1395, -0.0650]), decimal=4)
        np.testing.assert_almost_equal(emb[4][260:268], np.array([-0.0787,  0.4456, -0.3024,  0.2425,  0.1895, -0.4241, -0.0121, -0.1657]), decimal=4)
        np.testing.assert_almost_equal(emb[-1][-8:],    np.array([-0.0244, -0.1880, -0.3196,  0.1533,  0.0351,  0.0785, -0.1745,  0.2113]), decimal=4)


if __name__ == '__main__':
    unittest.main()