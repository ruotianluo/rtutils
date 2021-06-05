import os
import mmcv

def get_coco_filename(cocoid):
    cocoid = int(cocoid)
    if os.path.isfile('/share/data/vdata/coco/images/train2017/%012d.jpg' %(cocoid)):
        return '/share/data/vdata/coco/images/train2017/%012d.jpg' %(cocoid)
    else:
        return '/share/data/vdata/coco/images/val2017/%012d.jpg' %(cocoid)

coco_image_infos = None

def get_coco_image_infos():
    coco_image_infos = mmcv.load('/share/data/vdata/coco/annotations/captions_train2017.json')['images'] + mmcv.load('/share/data/vdata/coco/annotations/captions_val2017.json')['images']
    coco_image_infos = {_['id']: _ for _ in coco_image_infos}
    return coco_image_infos


def get_coco_url(cocoid):
    global coco_image_infos
    if coco_image_infos is None:
        coco_image_infos = get_coco_image_infos()
    cocoid = int(cocoid)
    return coco_image_infos[cocoid]['coco_url']