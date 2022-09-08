import cv2
import torch
import torch.nn as nn

from utils.download import prep_weights_file

# https://arxiv.org/abs/1704.04861
# https://github.com/deepinsight/insightface/tree/master/alignment/coordinate_reg

class ConvUnit(nn.Module):
    def __init__(self, cin, cout, k, s, p, grp=1, relu=True):
        super(ConvUnit, self).__init__()
        cin, cout, grp = int(cin), int(cout), int(grp)
        self.conv = nn.Conv2d(cin, cout, k, s, p, groups=grp, bias=False)
        self.bn = nn.BatchNorm2d(cout, eps=0.001)
        self.relu = nn.PReLU(cout)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, width_multiplier=1):
        super(MobileNet, self).__init__()
        a = width_multiplier
        self.main = nn.Sequential(
            ConvUnit(3, 32*a, 3, 2, 1),
            ConvUnit(32*a, 32*a, 3, 1, 1, grp=32*a), ConvUnit(32*a, 64*a, 1, 1, 0),
            ConvUnit(64*a, 64*a, 3, 2, 1, grp=64*a), ConvUnit(64*a, 128*a, 1, 1, 0),
            ConvUnit(128*a, 128*a, 3, 1, 1, grp=128*a), ConvUnit(128*a, 128*a, 1, 1, 0),
            ConvUnit(128*a, 128*a, 3, 2, 1, grp=128*a), ConvUnit(128*a, 256*a, 1, 1, 0),
            ConvUnit(256*a, 256*a, 3, 1, 1, grp=256*a), ConvUnit(256*a, 256*a, 1, 1, 0),
            ConvUnit(256*a, 256*a, 3, 2, 1, grp=256*a), ConvUnit(256*a, 512*a, 1, 1, 0),
            *[el for tup in [(ConvUnit(512*a, 512*a, 3, 1, 1, grp=512*a), ConvUnit(512*a, 512*a, 1, 1, 0)) for _ in range(5)] for el in tup],
            ConvUnit(512*a, 512*a, 3, 2, 1, grp=512*a), ConvUnit(512*a, 1024*a, 1, 1, 0),
            ConvUnit(1024*a, 1024*a, 3, 1, 1, grp=1024*a), ConvUnit(1024*a, 1024*a, 1, 1, 0)
        )
        self.outro = nn.Sequential(
            ConvUnit(1024*a, 128*a, 3, 2, 1),
            nn.Flatten(),
            nn.Linear(int(128*a)*3*3, 212)
        )
    def forward(self, x):
        x = self.main(x)
        x = self.outro(x)
        return x
        
# https://github.com/deepinsight/insightface/tree/master/model_zoo#21-2d-face-alignment
# https://github.com/nttstar/insightface-resources/blob/master/alignment/images/2d106markup.jpg

class MobileNetLandmarker():
    def __init__(self, device):
        print('Initializing MobileNet model for landmark detection')
        self.model = MobileNet(0.5).to(device)
        wf = prep_weights_file('https://drive.google.com/uc?id=1H1-KkskDrQvQ_J_bFLxZeie6YvTfB7KE', 'mbnet_2d106det_insightface.pt', gdrive=True)
        wd = torch.load(wf, map_location=torch.device(device))
        self.model.load_state_dict(wd)
        self.model.eval()
    
    def __call__(self, images):
        input = cv2.dnn.blobFromImages(images, 1 / 128, (192, 192), (127.5, 127.5, 127.5), swapRB=True)
        with torch.no_grad():
            lm = self.model(torch.from_numpy(input)).numpy()
        lm = lm.reshape(-1, 106, 2)
        lm = lm[:, [38, 88, 86, 52, 61], :]
        lm = (lm + 1) * (192 // 2)
        lm = lm.round().astype(int)
        return lm

# ==================== EXTRA / NOTES ====================

def convert_weights_2d106det_insightface():
    '''This is the code I used to convert Insightface's ONNX model's weights to pytorch
       import it from here and call directly if needed
       it should create the same 'mbnet_2d106det_insightface.pt' file that's used above
       need "pip install onnx" to work
       for browsing onnx model, I used https://netron.app mentioned here: https://github.com/onnx/onnx/issues/1425'''
    import shutil, os
    import os.path as osp
    home = os.getcwd()
    dst = prep_weights_file('https://drive.google.com/uc?id=1M5685m-bKnMCt0u2myJoEK5gUY3TDt_1', '2d106det.onnx', gdrive=True)
    os.chdir(osp.dirname(dst))
    print('working at: ' + os.getcwd())

    import onnx, onnx.numpy_helper
    onnx_model = onnx.load('2d106det.onnx')
    src = dict([(raw.name, onnx.numpy_helper.to_array(raw)) for raw in onnx_model.graph.initializer])
    
    match = []
    layers = ['conv_1', 'conv_2_dw', 'conv_2', 'conv_3_dw', 'conv_3', 'conv_4_dw', 'conv_4', 'conv_5_dw',
                        'conv_5', 'conv_6_dw', 'conv_6', 'conv_7_dw', 'conv_7', 'conv_8_dw', 'conv_8', 'conv_9_dw',
                        'conv_9', 'conv_10_dw', 'conv_10', 'conv_11_dw', 'conv_11', 'conv_12_dw', 'conv_12','conv_13_dw',
                        'conv_13', 'conv_14_dw', 'conv_14', 'conv_15']
    for l in layers:
        match.append(l + '_conv2d_weight')
        match.append(l + '_batchnorm_gamma')
        match.append(l + '_batchnorm_beta')
        match.append(l + '_batchnorm_moving_mean')
        match.append(l + '_batchnorm_moving_var')
        match.append(l + '_empty') # placeholder
        match.append(l + '_relu_gamma')
    match.append('fc1_weight')
    match.append('fc1_bias')

    model = MobileNet(0.5)
    dst = model.state_dict()
    for i, w in enumerate(dst):
        if w.endswith('.num_batches_tracked'):
            continue
        val = src[str(match[i])]
        if dst[w].dim() == 1:
            val = val.squeeze()
        dst[w] = torch.Tensor(val)
    model.load_state_dict(dst)
    model.eval()
    torch.save(model.state_dict(), 'mbnet_2d106det_insightface.pt')
    print('saved mbnet_2d106det_insightface.pt')
    os.remove('2d106det.onnx')
    print('removed 2d106det.onnx')
    os.chdir(home)
    print('returned to: ' + home)