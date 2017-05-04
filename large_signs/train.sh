rm data/cache/*.pkl
python2 ./tools/train_net.py --gpu 0 --solver models/pascal_voc/VGG16/faster_rcnn_end2end/solver.prototxt --weights data/imagenet_models/VGG16.v2.caffemodel --imdb hs --iters 80000 --cfg experiments/cfgs/faster_rcnn_end2end.yml
