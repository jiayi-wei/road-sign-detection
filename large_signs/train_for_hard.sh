rm data/cache/*.pkl
python2 ./tools/train_net.py --gpu 1 --solver models/pascal_voc/VGG16/faster_rcnn_end2end/solver_wjy_finetune.prototxt --weights output/faster_rcnn_end2end/hs/vgg16_faster_rcnn_iter_80000_third.caffemodel --imdb hs --iters 40000 --cfg experiments/cfgs/faster_rcnn_end2end.yml
