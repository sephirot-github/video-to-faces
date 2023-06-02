import os.path as osp
import sys
import unittest

import cv2
import numpy as np
import PIL.Image
import torch

from videotofaces.encoders.vit import VitEncoderAnime, VitEncoder, VitClip, VitTorchVision

EXTENDED = False


class TestVIT(unittest.TestCase):

    def test_vit_anime(self):
        testdir = osp.dirname(osp.realpath(__file__))
        imgs = [cv2.imread(osp.join(testdir, 'images', 'aniface%u.jpg' % i)) for i in [1, 2]]
        model = VitEncoderAnime('cpu', 'B-16-Danbooru-Faces', classify=True)
        prob, emb = model(imgs)
        self.assertEqual(emb.shape, (2, 768))
        np.testing.assert_almost_equal(emb[0][100:105], np.array([-0.4530, -2.1694, 0.0624, -0.7991, -0.3798]), decimal=4)
        np.testing.assert_almost_equal(emb[1][640:645], np.array([0.3255, -0.6816, -0.1108,  0.2946,  1.7022]), decimal=4)
        self.assertEqual(prob.shape, (2, 3263))
        pred = model.get_predictions(prob)
        self.assertEqual([len(pred), len(pred[0]), len(pred[1])], [2, 5, 5])
        txt = [[(nm, '%.6f' % pb) for nm, pb in group[:2]] for group in pred]
        self.assertEqual(txt[0], [('shirai_kuroko', '99.998939'), ('fukuzawa_yumi', '0.000314')])
        self.assertEqual(txt[1], [('misaka_mikoto', '99.951982'), ('misaka_imouto', '0.040299')])
        if EXTENDED:
            model = VitEncoderAnime('cpu', 'L-16-Danbooru-Faces', classify=True)
            prob, emb = model(imgs)
            self.assertEqual(emb.shape, (2, 1024))
            np.testing.assert_almost_equal(emb[0][900:905], np.array([-1.5694, 0.1522, -1.8948, -1.2867, -1.8749]), decimal=4)
            np.testing.assert_almost_equal(emb[1][175:180], np.array([0.3184, -1.2670, -0.0992, -0.0231,  0.5195]), decimal=4)
            self.assertEqual(prob.shape, (2, 3263))
            pred = model.get_predictions(prob)
            self.assertEqual([len(pred), len(pred[0]), len(pred[1])], [2, 5, 5])
            txt = [[(nm, '%.5f' % pb) for nm, pb in group[:2]] for group in pred]
            self.assertEqual(txt[0], [('shirai_kuroko', '99.97436'), ('misaka_imouto', '0.00944')])
            self.assertEqual(txt[1], [('misaka_mikoto', '99.99293'), ('misaka_imouto', '0.00635')])
            model = VitEncoderAnime('cpu', 'B-16-Danbooru-Full', classify=True)
            prob, _ = model(imgs)
            pred = model.get_predictions(prob)
            txt = [[(nm, '%.5f' % pb) for nm, pb in group[:2]] for group in pred]
            self.assertEqual(txt[0], [('shirai_kuroko', '96.67075'), ('sakura_kyouko', '1.81579')])
            self.assertEqual(txt[1], [('misaka_mikoto', '99.94698'), ('misaka_imouto', '0.04537')])

    def test_vit_torchvision(self):
        testdir = osp.dirname(osp.realpath(__file__))
        names = ['bear1', 'cat1', 'city1']
        paths = [osp.join(testdir, 'images', 'sml_%s.jpg' % nm) for nm in names]
        imgs = [PIL.Image.open(pt) for pt in paths]
        model = VitTorchVision('cpu', 'B-16')
        prob, _ = model(imgs)
        pred = model.get_predictions(prob)
        txt = [[(nm, '%.3f' % pb) for nm, pb in group[:3]] for group in pred]
        self.assertEqual(prob.shape, (3, 1000))
        self.assertEqual([len(pred), len(pred[0]), len(pred[1])], [3, 5, 5])
        self.assertEqual(txt[0], [('American black bear', '73.204'), ('brown bear', '20.141'), ('unicycle', '1.799')])
        self.assertEqual(txt[1], [('Egyptian cat', '50.508'), ('tabby', '31.426'), ('Persian cat', '5.010')])
        self.assertEqual(txt[2], [('steel arch bridge', '38.447'), ('planetarium', '10.220'), ('balloon', '7.224')])

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
        if EXTENDED:
            model = VitClip('cpu', 'B-16')
            emb = model(imgs)
            self.assertEqual(emb.shape, (7, 512))
            np.testing.assert_almost_equal(emb[0][:8],      np.array([-0.6649, -0.9117, -0.1872,  0.7836,  0.1861, -0.2109,  0.0424,  0.2385]), decimal=4)
            np.testing.assert_almost_equal(emb[1][100:108], np.array([-0.3041, -0.1231,  0.1070, -0.8348,  0.1730, -0.0067,  0.2839, -0.0824]), decimal=4)
            np.testing.assert_almost_equal(emb[4][260:268], np.array([ 0.1359, -0.7762,  0.0599, -0.3834,  0.0420,  0.0850, -0.8429,  0.3719]), decimal=4)
            np.testing.assert_almost_equal(emb[-1][-8:],    np.array([ 0.3142,  0.7684, -0.0044,  0.3631, -0.1112,  0.3121,  0.3534,  0.4730]), decimal=4)
            model = VitClip('cpu', 'L-14')
            emb = model(imgs)
            self.assertEqual(emb.shape, (7, 768))
            np.testing.assert_almost_equal(emb[0][:8],      np.array([ 0.0813, -0.5508,  0.0933, -0.7855,  0.3889,  1.0453,  0.1818,  0.1567]), decimal=4)
            np.testing.assert_almost_equal(emb[1][100:108], np.array([ 0.4566,  0.3669, -0.4060, -0.0128, -0.0802, -0.2200, -1.2715, -0.0013]), decimal=4)
            np.testing.assert_almost_equal(emb[4][260:268], np.array([ 0.3235, -0.4756, -0.7215, -0.5527, -0.3293, -0.2288, -0.1420, -0.6778]), decimal=4)
            np.testing.assert_almost_equal(emb[-1][-8:],    np.array([-0.2794,  0.3624,  0.1038,  0.0030,  0.3306, -0.0087,  0.4609,  0.5214]), decimal=4)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        EXTENDED = sys.argv[1] == 'EXTENDED'
        print('EXTENDED')
        sys.argv.pop()
    unittest.main()