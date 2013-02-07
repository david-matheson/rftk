import numpy as np
import matplotlib.pyplot as pl

import predict
import bodyparts

def classification_accuracy(depthsIn, labelsIn, classificationTreesIn):
    incorrectClassificationCount = 0
    nonBackgroundCount = 0
    background = 19

    (numberOfImgs,_,_) = depthsIn.shape

    for imgId in range(numberOfImgs):
        print "Img %d of %d" % (imgId, numberOfImgs)
        groundTruthLabels = labelsIn[imgId,:,:]
        (m,n) = groundTruthLabels.shape
        (pred_labels, probs) = predict.classifyPixels(depthsIn[imgId,:,:], classificationTreesIn)

        incorrectClassificationCount = incorrectClassificationCount + np.sum((groundTruthLabels != pred_labels) & (groundTruthLabels != bodyparts.background))
        nonBackgroundCount = nonBackgroundCount + np.sum( groundTruthLabels != bodyparts.background )

    return float(nonBackgroundCount - incorrectClassificationCount) / float(nonBackgroundCount)


# # Construct label image from trees
# def label_predictions(depth, classificationTrees, probabilityThreshold):
#     (M,N) = depth.shape
#     reconstructedImg = np.ones((M, N, 3))
#     print "classifyPixels start"
#     (labels, probs) = predict.classifyPixels(depth, classificationTrees)
#     print "classifyPixels end"

#     for m in range(M):
#         for n in range(N):
#             label = labels[m,n]
#             if probs[m,n] > probabilityThreshold:
#                 reconstructedImg[m,n] = bodyparts.get_color(label) / 255.0

#     return reconstructedImg

