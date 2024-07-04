import unittest
from mmdet.apis import init_detector, inference_detector
class TestMMDetection(unittest.TestCase):

    def test_predict_model(self):
        config_file = '../configs/mosquito_model/mask-rcnn_r50-caffe_fpn_ms-poly-1x_mosquito.py'
        checkpoint_file = 'mark_rcnn_mosquito.pth'
        model = init_detector(config_file, checkpoint_file, device='cuda:0')  # device='cpu' or device='cuda:0'
        inference_detector(model, '../data/mosquito/MyInference/aedes.jpg')
        print('Done')

if __name__ == '__main__':
    unittest.main()