import numpy as np
import matplotlib.pyplot as pl
import rftk.buffers as buffers
import rftk.predict as predict

head = 0
torso0L = 1
torso0R = 2
torso1L = 3
torso1R = 4
torso2L = 5
torso2R = 6

leg0L = 7
leg0R = 8
leg1L = 9
leg1R = 10
leg2L = 11
leg2R = 12

arm0L = 13
arm0R = 14
arm1L = 15
arm1R = 16
arm2L = 17
arm2R = 18

background = 19

number_of_body_parts = 20

colors = {}
colors[head   ] = np.array([56.0, 170, 0])
colors[torso0L] = np.array([156.0, 60, 134])
colors[torso0R] = np.array([204.0, 54, 118])
colors[torso1L] = np.array([158.0, 100, 114])
colors[torso1R] = np.array([205.0, 85, 116])
colors[torso2L] = np.array([155.0, 129, 101])
colors[torso2R] = np.array([192.0, 143, 110])

colors[leg0L  ] = np.array([160.0, 153, 114])
colors[leg0R ] = np.array([193.0, 138, 73])
colors[leg1L  ] = np.array([31.0, 109, 82])
colors[leg1R  ] = np.array([228.0, 193, 0])
colors[leg2L  ] = np.array([55.0, 171, 112])
colors[leg2R  ] = np.array([232.0, 229, 0])

colors[arm0L  ] = np.array([118.0, 63, 153])
colors[arm0R  ] = np.array([224.0, 64, 98])
colors[arm1L  ] = np.array([85.0, 49, 139])
colors[arm1R  ] = np.array([223.0, 62, 45])
colors[arm2L  ] = np.array([21.0, 71, 155])
colors[arm2R  ] = np.array([224.0, 90, 0])


colors[background ] = np.array([64.0, 64, 64])

def get_color(color_id):
    return colors[color_id]

max_depth = 6.0

def load_training_data(numpy_filename):
    f = open(numpy_filename, 'rb')
    depths = np.load(f)
    labels = np.load(f)
    pixel_indices = np.load(f)
    pixel_labels = np.load(f)
    depths_buffer = buffers.as_tensor_buffer(depths)
    del depths
    del labels
    pixel_indices_buffer = buffers.as_matrix_buffer(pixel_indices)
    del pixel_indices
    pixel_labels_buffer = buffers.as_vector_buffer(pixel_labels)
    del pixel_labels
    return depths_buffer, pixel_indices_buffer, pixel_labels_buffer


# Reconstruct depth image
def depth_to_image(depth):
    (M, N) = depth.shape
    rgb = 255.0 * np.ones((M,N,3))
    for m in range(M):
        for n in range(N):
            rgb[m,n,0] = depth[m,n]/max_depth
            rgb[m,n,1] = depth[m,n]/max_depth
            rgb[m,n,2] = depth[m,n]/max_depth
    return rgb

# Reconstruct image from labels
def labels_to_image(labels):
    (M,N) = labels.shape
    img = np.ones((M, N, 3))

    for body_part in range(number_of_body_parts):
        img[labels == body_part] =  colors[body_part] / 255.0

    return img

# Reconstruct image from labels
def labels_to_image_ushort(labels):
    (M,N) = labels.shape
    img = np.ones((M, N, 3), dtype=np.uint8)

    for body_part in range(number_of_body_parts):
        img[(labels == body_part)] =  np.array(colors[body_part], dtype=np.uint8)

    return img

# Reconstruct image from labels with min probability
def labels_to_image_ushort_min_prob(labels, ground_labels, probabilities, min_probablity):
    (M,N) = labels.shape
    img = np.ones((M, N, 3), dtype=np.uint8)

    for body_part in range(number_of_body_parts):
        img[(labels == body_part) & (probabilities > min_probablity)] = np.array(colors[body_part], dtype=np.uint8)

    return img

def to_indices(image_id, where_ids):
    rows = where_ids[0]
    colms = where_ids[1]
    assert( len(rows) == len(colms) )
    indices = np.zeros((len(rows), 3),  dtype=np.int32)
    indices[:,0] = image_id
    indices[:,1] = rows.T
    indices[:,2] = colms.T
    return indices

def classify_body_pixels(depth, ground_labels, forest):
    buffer_collection = buffers.BufferCollection()
    assert(depth.ndim == 2)
    m,n = depth.shape
    pixel_indices = to_indices(0, np.where(ground_labels != background))
    (number_of_non_background_pixels,_) = pixel_indices.shape
    buffer_collection.AddInt32MatrixBuffer(buffers.PIXEL_INDICES, buffers.as_matrix_buffer(pixel_indices))
    buffer_collection.AddFloat32Tensor3Buffer(buffers.DEPTH_IMAGES, buffers.as_tensor_buffer(depth))

    forest_predictor = predict.ForestPredictor(forest)
    yprobs_buffer = buffers.Float32MatrixBuffer()
    forest_predictor.PredictYs(buffer_collection, number_of_non_background_pixels, yprobs_buffer)
    yprobs = buffers.as_numpy_array(yprobs_buffer)
    (_, ydim) = yprobs.shape
    img_yprobs = np.zeros((m,n), dtype=np.float32)
    img_yprobs[ground_labels != background].shape
    yprobs.max(axis=1).shape
    img_yprobs[ground_labels != background] = yprobs.max(axis=1)
    img_yhat = np.zeros((m,n), dtype=np.int32)
    img_yhat[ground_labels != background] = np.argmax(yprobs, axis=1)

    return img_yhat, img_yprobs

classificationTreesGlobal = None
depthsGlobal = None
labelsGlobal = None

def classification_accuracy_image(imgId):
    global classificationTreesGlobal
    global depthsGlobal
    global labelsGlobal

    (numberOfImgs,_,_) = depthsGlobal.shape
    groundTruthLabels = labelsGlobal[imgId,:,:]
    (m,n) = groundTruthLabels.shape
    pred_labels, pred_probs = classify_body_pixels(depthsGlobal[imgId,:,:], labelsGlobal[imgId,:,:], classificationTreesGlobal)

    incorrectClassificationCount = np.sum((groundTruthLabels != pred_labels) & (groundTruthLabels != background))
    nonBackgroundCount = np.sum(groundTruthLabels != background)
    return (incorrectClassificationCount, nonBackgroundCount)

def classification_accuracy(depthsIn, labelsIn, classificationTreesIn, number_of_jobs=4):
    from joblib import Parallel, delayed
    global classificationTreesGlobal
    global depthsGlobal
    global labelsGlobal

    classificationTreesGlobal = classificationTreesIn
    depthsGlobal = depthsIn
    labelsGlobal = labelsIn

    (numberOfImgs,_,_) = depthsIn.shape
    incorrectClassificationCount = 0
    nonBackgroundCount = 0

    counts = Parallel(n_jobs=number_of_jobs)(delayed(classification_accuracy_image)(imgId)
      for imgId in range(numberOfImgs))

    incorrectClassificationCount = sum([x[0] for x in counts])
    nonBackgroundCount = sum([x[1] for x in counts])

    return float(nonBackgroundCount - incorrectClassificationCount) / float(nonBackgroundCount)


def plot_classification_img(figures_path, figure_id, depth, ground_labels, forest):
    labels, probs = classify_body_pixels(depth, ground_labels, forest)

    img = depth_to_image(depth)
    pl.imshow(img)
    pl.draw()
    pl.savefig(figures_path + "%d-depth.png" % (figure_id))
    pl.show()

    img = labels_to_image(ground_labels)
    pl.imshow(img)
    pl.draw()
    pl.savefig(figures_path + "%d-groundLabels.png" % (figure_id))
    pl.show()

    img = labels_to_image_ushort_min_prob(labels, ground_labels, probs, 0.0)
    pl.imshow(img)
    pl.draw()
    pl.savefig(figures_path + "%d-predictedLabels.png" % (figure_id))
    pl.show()

    img = labels_to_image_ushort_min_prob(labels, ground_labels, probs, 0.5)
    pl.imshow(img)
    pl.draw()
    pl.savefig(figures_path + "%d-predictedLabelsConfident.png" % (figure_id))
    pl.show()


def plot_classification_imgs(figures_path, depths, ground_labels, forest):
    (numberOfImgs,_,_) = depths.shape
    for imgId in range(numberOfImgs):
        print "Img %d of %d" % (imgId, numberOfImgs)
        plot_classification_img(figures_path, imgId, depths[imgId,:,:], ground_labels[imgId,:,:], forest)
