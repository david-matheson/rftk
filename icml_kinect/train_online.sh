#!/bin/bash
python train_online.py -i source_image_640x480/ -p poses/large_train_poses.txt -n 16000 -m 1 -t 25 -s 1.0 -r 10 -e 10 -d 500 >> train_online.log