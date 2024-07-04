import unittest
import torch

class TestStringMethods(unittest.TestCase):

    def test_found_object(self):
        model_fpath = '../test/mark_rcnn_mosquito.pth'
        model = torch.jit.load(model_fpath)

        image_fpath = '../test/anopheles.jpg'
        result = model(image_fpath)

        self.assertIsNotNone(result)

        list_results = list(result)
        self.assertIsNone(list_results[0])

        first_result = list_results[0]
        print(f"Names in the model: {first_result.names()}")

        list_boxes = list(first_result.boxes)
        box = list_boxes[0]

        index_object = int(box.cls[0])

        self.assertEqual(index_object, 0)

        detected_classname = first_result.names[index_object]
        self.assertEqual(detected_classname, '00-aedes-dau')

        self.assertGreater(box.conf[0], 0.5)

        first_result.save(filename='output.jpg')

if __name__ == '__main__':
    unittest.main()