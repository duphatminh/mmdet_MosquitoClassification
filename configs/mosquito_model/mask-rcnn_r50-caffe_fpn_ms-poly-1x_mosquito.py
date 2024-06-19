# The new config inherits a base config to highlight the necessary modification
# Lua chon model nao?
_base_ = '../mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=6), mask_head=dict(num_classes=6)))

# Modify dataset related settings
# duong dan den thu muc data
data_root = 'data/mosquito/'
# class de dat ten cho doi tuong can detect, palette de dinh nghia mau cua bounding box
metainfo = {
    'classes': ('00-aedes-dau', '00-aedes-than', '01-anopheles-dau', '01-anopheles-than', '02-culex-dau', '02-culex-than', ),
    'palette': [
        (220, 20, 60),
        (30, 144, 255),
        (34, 139, 34),
        (255, 215, 0),
        (255, 69, 0),
        (138, 43, 226),
    ]
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
		    # thu muc du lieu goc
        data_root=data_root,
        metainfo=metainfo,
        # chi dinh duong dan den file annotation
        ann_file='annotations/instances_Train.json',
        # chi dinh duong dan den file chua anh training
        data_prefix=dict(img='images/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_Test.json',
        data_prefix=dict(img='images/')))
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'annotations/instances_Test.json')
test_evaluator = val_evaluator