import os.path as osp
import unittest

import cv2
import numpy as np

from videotofaces import RetinaFaceDetector


class TestRetinaFace(unittest.TestCase):

    def test_mobilenet(self):
        model = RetinaFaceDetector('cpu', 'mobilenet')
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
        self.assertAlmostEqual(r1[0][0], 86.70745, places=5)
        self.assertAlmostEqual(r1[4][3], 79.6413, places=4)
        self.assertAlmostEqual(r1[0][4], 0.99970, places=5)
        self.assertAlmostEqual(r1[4][4], 0.93230, places=5)
        
        self.assertEqual(r2.shape, (8, 5))
        self.assertAlmostEqual(r2[0][1], 31.763332, places=6)
        self.assertAlmostEqual(r2[7][2], 307.38235, places=5)
        self.assertAlmostEqual(r2[1][4], 0.99907, places=5)
        self.assertAlmostEqual(r2[6][4], 0.034624, places=6)
        
        self.assertEqual(r3.shape, (350, 5))
        np.testing.assert_almost_equal(r3[1], np.array([451.24835, 312.8681, 506.46442, 387.36926, 0.99936]), decimal=5)
        np.testing.assert_almost_equal(r3[92], np.array([10.99331, 180.34042, 37.50953, 216.78476, 0.12444]), decimal=5)
   
        self.assertEqual(r4.shape, (73, 5))
        np.testing.assert_almost_equal(r4[12], np.array([8.02442, 130.91562, 44.31978, 176.37561, 0.99538]), decimal=5)
        np.testing.assert_almost_equal(r4[-5], np.array([751.04456, 5.32989, 795.59546, 48.92341, 0.02167]), decimal=5)

        self.assertEqual(len(rb), 2)
        self.assertEqual(rb[0].shape, (24, 5))
        self.assertEqual(rb[1].shape, (20, 5))
        np.testing.assert_almost_equal(rb[0][10], np.array([541.10864, 105.35722, 591.18164, 169.09984, 0.99779]), decimal=4)
        np.testing.assert_almost_equal(rb[0][22], np.array([585.0051, 129.43068, 612.2147, 164.56346, 0.02089]), decimal=4)
        np.testing.assert_almost_equal(rb[1][4], np.array([931.3569, 213.46419, 981.17365, 282.21033, 0.98094]), decimal=4)
        np.testing.assert_almost_equal(rb[1][8], np.array([298.3359, 194.7117, 355.0907, 270.89612, 0.04269]), decimal=4)

    def test_resnet(self):
        model = RetinaFaceDetector('cpu', 'resnet')
        testdir = osp.dirname(osp.realpath(__file__))
        im1 = cv2.imread(osp.join(testdir, 'images', 'img_726.jpg'))
        #im2 = cv2.imread(osp.join(testdir, 'images', 'img_588.jpg'))
        #im3 = cv2.imread(osp.join(testdir, 'images', '2_Demonstration_Demonstration_Or_Protest_2_58.jpg'))
        #im4 = cv2.imread(osp.join(testdir, 'images', '17_Ceremony_Ceremony_17_171.jpg'))
        im5 = cv2.imread(osp.join(testdir, 'images', '12_Group_Group_12_Group_Group_12_10.jpg'))
        im6 = cv2.imread(osp.join(testdir, 'images', '12_Group_Group_12_Group_Group_12_29.jpg'))
        r1 = model([im1])[0]
        #r2 = model([im2])[0]
        #r3 = model([im3])[0]
        #r4 = model([im4])[0]
        rb = model([im5, im6])

        self.assertEqual(r1.shape, (6, 5))
        np.testing.assert_almost_equal(r1[0][:4], np.array([83.71, 49.78, 168.86, 169.30]), decimal=2)
        np.testing.assert_almost_equal(r1[4][:4], np.array([225.37, 47.23, 245.87, 79.22]), decimal=2)
        self.assertAlmostEqual(r1[0][4], 0.9994067)
        self.assertAlmostEqual(r1[4][4], 0.9867643)
                
        #self.assertEqual(r2.shape, (9, 5))
        #np.testing.assert_almost_equal(r2[1][:4], np.array([134.65, 30.26, 190.53, 107.31]), decimal=2)
        #np.testing.assert_almost_equal(r2[-1][:4], np.array([24.32, 62.72, 34.13, 76.91]), decimal=2)
        #self.assertAlmostEqual(r2[1][4], 0.9996773)
        #self.assertAlmostEqual(r2[-1][4], 0.0263380)
        
        #self.assertEqual(r3.shape, (130, 5))
        #np.testing.assert_almost_equal(r3[10], np.array([518.38293, 231.06493, 562.0498, 291.08487, 0.9989656]), decimal=4)
        #np.testing.assert_almost_equal(r3[90], np.array([744.04175, 52.911484, 755.9617, 68.882805, 0.0690808]), decimal=4)
   
        #self.assertEqual(r4.shape, (43, 5))
        #np.testing.assert_almost_equal(r4[20], np.array([663.446, 75.807, 707.074, 132.183, 0.99165523]), decimal=3)
        
        self.assertEqual(len(rb), 2)
        self.assertEqual(rb[0].shape, (14, 5))
        self.assertEqual(rb[1].shape, (9, 5))
        np.testing.assert_almost_equal(rb[0][5], np.array([290.9456, 339.1132, 352.7952, 420.90085, 0.99942017]), decimal=4)
        np.testing.assert_almost_equal(rb[1][1], np.array([81.9536, 199.9863, 172.0615, 331.8756, 0.99976426]), decimal=4)
        

if __name__ == '__main__':
    unittest.main()