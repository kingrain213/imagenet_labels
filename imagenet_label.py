import sys
import os
import cv2
import shutil
from scipy import io
import json
import yaml

#解决imagenet标签映射问题，Imagenet val包含50000张图像，真值在ILSVRC2012_validation_ground_truth.txt中
#按照索引和图片名找对应关系，但是索引与pytorch分类模型预测并不一致，需要通过meta.mat进行映射

#先根据meta和groundtruth文件将原始一个大文件夹中图片拆分成WIND类型的1000个小文件夹
def re_assign_imagenet_step1():
    """
    move valimg to correspongding folders.
    val_id(start from 1) -> ILSVRC_ID(start from 1) -> WIND
    organize like:
    /val
       /n01440764
           images
       /n01443537
           images
        .....
    """
    # load synset, val ground truth and val images list
    meta_file=r'D:\data\ILSVRC2012_devkit_t12\ILSVRC2012_devkit_t12\data\meta.mat'
    label_txt = r'D:\data\ILSVRC2012_devkit_t12\ILSVRC2012_devkit_t12\data\ILSVRC2012_validation_ground_truth.txt'
    dst_val_dir=r'D:\data\img_val_v2'
    val_dir = r'D:\data\ILSVRC2012_img_val'

    synset = io.loadmat(meta_file)    
    ground_truth = open(label_txt)

    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]
    
    root, _, filenames = next(os.walk(val_dir))
    for filename in filenames:
        # val image name -> ILSVRC ID -> WIND
        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id-1]
        WIND = synset['synsets'][ILSVRC_ID-1][0][1][0]
        print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))

        # move val images
        output_dir = os.path.join(dst_val_dir, WIND)
        if os.path.isdir(output_dir):
            pass
        else:
            os.mkdir(output_dir)
        shutil.copy(os.path.join(root, filename), os.path.join(output_dir, filename))
		
# 第二部，把拷贝好的文件夹重新命名到0~999
def re_assign_imagenet_step2():

    src_imgdir = r'D:\data\img_val_v2'
    dst_imgdir = r'D:\data\img_val_v3'
    jsonfile = r'D:\data\imagenet_class_index.json'

    with open(jsonfile) as fd:
        data = json.load(fd)
    
    new_maps = {}
    for i in range(1000):
        # fp = open(r'D:\data\imagenet_class.yaml', 'a')
        # str2write = str(i) + ': ' + str(data[str(i)][::-1])
        # fp.write(str2write+'\n')
        # fp.close()

        new_maps[data[str(i)][0]] = str(i)

    firfiles = os.listdir(src_imgdir)
    for eachfile in firfiles:
        newfir = new_maps[eachfile]
        dstdir = os.path.join(dst_imgdir, newfir)
        os.makedirs(dstdir, exist_ok=True)

        imgfiles = os.listdir(os.path.join(src_imgdir, eachfile))
        imgfiles = [os.path.join(src_imgdir, eachfile, v) for v in imgfiles]
        for eachimg in imgfiles:
            shutil.copy(eachimg, dstdir)

        
    config_path = r'D:\data\imagenet_class.yaml'
    with open(config_path) as fd:
        config = yaml.load(fd, Loader=yaml.CLoader)

    print('re_assign_imagenet_v2_sec finished')

if __name__ == '__main__':
    re_assign_imagenet_step1()
	  re_assign_imagenet_step2()
