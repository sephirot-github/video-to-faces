import os.path as osp
import unittest

import cv2
import numpy as np

from videotofaces import RetinaFaceDetector


class TestRetinaFace(unittest.TestCase):

    def test1(self):
        model = RetinaFaceDetector('cpu')
        testdir = osp.dirname(osp.realpath(__file__))
        im1 = cv2.imread(osp.join(testdir, 'images', 'img_726.jpg'))
        im2 = cv2.imread(osp.join(testdir, 'images', 'img_588.jpg'))
        im3 = cv2.imread(osp.join(testdir, 'images', '2_Demonstration_Demonstration_Or_Protest_2_58.jpg'))
        im4 = cv2.imread(osp.join(testdir, 'images', '17_Ceremony_Ceremony_17_171.jpg'))
        im5 = cv2.imread(osp.join(testdir, 'images', '12_Group_Group_12_Group_Group_12_10.jpg'))
        im6 = cv2.imread(osp.join(testdir, 'images', '12_Group_Group_12_Group_Group_12_29.jpg'))
        r1 = model([im1])[0]
        r2 = model([im2])[0]
        r3 = model([im3])[0]
        r4 = model([im4])[0]
        rb = model([im5, im6])

        self.assertEqual(r1.shape, (7, 5))
        self.assertAlmostEqual(r1[0][0], 86)
        self.assertAlmostEqual(r1[4][3], 79)
        self.assertAlmostEqual(r1[0][4], 0.99970, places=5)
        self.assertAlmostEqual(r1[4][4], 0.93230, places=5)
        
        self.assertEqual(r2.shape, (8, 5))
        self.assertAlmostEqual(r2[0][1], 31)
        self.assertAlmostEqual(r2[7][2], 307)
        self.assertAlmostEqual(r2[1][4], 0.99907, places=5)
        self.assertAlmostEqual(r2[6][4], 0.034624, places=6)
        
        self.assertEqual(r3.shape, (327, 5))
        np.testing.assert_almost_equal(r3[92], np.array([10., 180., 37., 216., 0.12444]), decimal=5)
        self.assertEqual(r3[92][1], 180)
 
        self.assertEqual(r4.shape, (72, 5))
        np.testing.assert_almost_equal(r4[12], np.array([8., 130., 44., 176., 0.99538]), decimal=5)
        np.testing.assert_almost_equal(r4[-5], np.array([353., 317., 404., 384., 0.02173]), decimal=5)

        self.assertEqual(len(rb), 2)
        self.assertEqual(rb[0].shape, (24, 5))
        self.assertEqual(rb[1].shape, (20, 5))
        np.testing.assert_almost_equal(rb[0][10], np.array([541., 105., 591., 169., 0.99779]), decimal=5)
        np.testing.assert_almost_equal(rb[0][22], np.array([585., 129., 612., 164., 0.02089]), decimal=5)
        np.testing.assert_almost_equal(rb[1][4], np.array([931., 213., 981., 282., 0.98094]), decimal=5)
        np.testing.assert_almost_equal(rb[1][8], np.array([298., 194., 355., 270., 0.04269]), decimal=5)


if __name__ == '__main__':
    unittest.main()