# imagenet_labels
imagenet label index imagenet class imagenet分类标签映射

ImageNet class label
解决Imagenet标签类别映射问题
#解决imagenet标签映射问题，Imagenet val包含50000张图像，真值在ILSVRC2012_validation_ground_truth.txt中，标签0~999
直接映射的话与pytorch模型标签不一致，需要根据meta文件进行转换

#按照索引和图片名找对应关系，但是索引与pytorch分类模型预测并不一致，需要通过meta.mat进行映射
