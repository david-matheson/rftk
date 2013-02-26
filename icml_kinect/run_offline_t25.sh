#!/bin/bash
python train_offline.py -i source_image_640x480/ -p poses/large_train_poses.txt  -t 25 -n 25
python train_offline.py -i source_image_640x480/ -p poses/large_train_poses.txt  -t 25 -n 50
python train_offline.py -i source_image_640x480/ -p poses/large_train_poses.txt  -t 25 -n 100
python train_offline.py -i source_image_640x480/ -p poses/large_train_poses.txt  -t 25 -n 200
python train_offline.py -i source_image_640x480/ -p poses/large_train_poses.txt  -t 25 -n 500
python train_offline.py -i source_image_640x480/ -p poses/large_train_poses.txt  -t 25 -n 1000

python train_online.py -i source_image_640x480/ -p poses/large_train_poses.txt -t 25 -s 1.2 -r 1000 -e 100 -n 25 -m 10
python train_online.py -i source_image_640x480/ -p poses/large_train_poses.txt -t 25 -s 1.2 -r 1000 -e 100 -n 100 -m 10
python train_online.py -i source_image_640x480/ -p poses/large_train_poses.txt -t 25 -s 1.2 -r 1000 -e 100 -n 200 -m 10
python train_online.py -i source_image_640x480/ -p poses/large_train_poses.txt -t 25 -s 1.2 -r 1000 -e 100 -n 500 -m 10
python train_online.py -i source_image_640x480/ -p poses/large_train_poses.txt -t 25 -s 1.2 -r 1000 -e 100 -n 1000 -m 10