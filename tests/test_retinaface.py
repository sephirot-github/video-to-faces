import os.path as osp
import unittest

import cv2
import numpy as np

from videotofaces import RetinaFaceDetector


class TestRetinaFace(unittest.TestCase):

    # from WIDER_val:
    # irl_det_1 = "12_Group_Group_12_Group_Group_12_10.jpg"
    # irl_det_2 = "12_Group_Group_12_Group_Group_12_29.jpg"
    # irl_det_3 = "2_Demonstration_Demonstration_Or_Protest_2_58.jpg"
    # irl_det_4 = "17_Ceremony_Ceremony_17_171.jpg"

    def test_mobilenet(self):
        model = RetinaFaceDetector('biubug6_mobilenet')
        testdir = osp.dirname(osp.realpath(__file__))
        imgs = [cv2.imread(osp.join(testdir, 'images', 'irl_det_%u.jpg' % i)) for i in [1, 2, 3, 4]]
        rb = model(imgs)
        self.assertEqual(len(rb), 4)
        self.assertEqual(rb[0].shape, (24, 5))
        self.assertEqual(rb[1].shape, (20, 5))
        self.assertEqual(rb[2].shape, (358, 5))
        self.assertEqual(rb[3].shape, (76, 5))
        np.testing.assert_almost_equal(rb[0][10], np.array([541.10864, 105.35722, 591.18164, 169.09984, 0.99779]), decimal=4)
        np.testing.assert_almost_equal(rb[0][22], np.array([585.0051, 129.43068, 612.2147, 164.56346, 0.02089]), decimal=4)
        np.testing.assert_almost_equal(rb[1][4], np.array([931.3569, 213.46419, 981.17365, 282.21033, 0.98094]), decimal=4)
        np.testing.assert_almost_equal(rb[1][8], np.array([298.3359, 194.7117, 355.0907, 270.89612, 0.04269]), decimal=4)
        np.testing.assert_almost_equal(rb[2][1], np.array([451.18222, 312.84677, 506.44626, 387.15103, 0.99937]), decimal=5)
        np.testing.assert_almost_equal(rb[2][92], np.array([454.6434, 258.5825, 471.7198, 280.2969, 0.1227]), decimal=4)
        np.testing.assert_almost_equal(rb[3][12], np.array([196.038, 235.07074, 256.24564, 313.40533, 0.99574]), decimal=5)
        np.testing.assert_almost_equal(rb[3][-5], np.array([621.6558, 4.7005, 637.6158, 27.7780, 0.0208]), decimal=4)

    def test_resnet50_A(self):
        model = RetinaFaceDetector('biubug6_resnet50')
        testdir = osp.dirname(osp.realpath(__file__))
        imgs = [cv2.imread(osp.join(testdir, 'images', 'irl_det_%u.jpg' % i)) for i in [1, 2]]
        rb = model(imgs)
        self.assertEqual(len(rb), 2)
        self.assertEqual(rb[0].shape, (14, 5))
        self.assertEqual(rb[1].shape, (9, 5))
        np.testing.assert_almost_equal(rb[0][5], np.array([290.9456, 339.1132, 352.7952, 420.90085, 0.99942017]), decimal=4)
        np.testing.assert_almost_equal(rb[1][1], np.array([81.9536, 199.9863, 172.0615, 331.8756, 0.99976426]), decimal=4)

    def test_resnet50_B(self):
        model = RetinaFaceDetector('bbt_resnet50_mixed')
        testdir = osp.dirname(osp.realpath(__file__))
        imgs = [cv2.imread(osp.join(testdir, 'images', 'irl_det_%u.jpg' % i)) for i in [3, 4]]
        rb = model(imgs)
        self.assertEqual(rb[0].shape, (43, 5))
        self.assertEqual(rb[1].shape, (27, 5)) 
        np.testing.assert_almost_equal(rb[0][2], np.array([162.3630, 301.0547, 217.6370, 372.3162, 0.9209]), decimal=4)
        np.testing.assert_almost_equal(rb[1][6], np.array([398.3187, 309.1938, 479.2593, 406.1411, 0.8273]), decimal=4)

    def test_resnet152(self):
        model = RetinaFaceDetector('bbt_resnet152_mixed')
        testdir = osp.dirname(osp.realpath(__file__))
        imgs = [cv2.imread(osp.join(testdir, 'images', 'irl_det_%u.jpg' % i)) for i in [3, 4]]
        rb = model(imgs)
        self.assertEqual(rb[0].shape, (43, 5))
        self.assertEqual(rb[1].shape, (26, 5))
        np.testing.assert_almost_equal(rb[0][1], np.array([46.0378, 243.8717, 103.3133, 305.6753, 0.9241]), decimal=4)
        np.testing.assert_almost_equal(rb[1][4], np.array([24.19395, 221.45657, 83.07513, 294.79721, 0.85426]), decimal=5)


if __name__ == '__main__':
    unittest.main()