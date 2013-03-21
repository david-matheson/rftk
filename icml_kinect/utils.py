import numpy as np
import matplotlib.pyplot as pl
import array
import pickle
import itertools

import rftk.asserts
import rftk.bootstrap
import rftk.buffers as buffers
import rftk.forest_data  as forest_data
import rftk.features
import rftk.feature_extractors as feature_extractors
import rftk.best_split as best_splits
import rftk.predict as predict
import rftk.train as train

import rftk.utils.predict as predict_utils
import rftk.utils.forest as forest_utils

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

bodyPartMirror = {
        head: head,
        torso0L: torso0R,
        torso0R: torso0L,
        torso1L: torso1R,
        torso1R: torso1L,
        torso2L: torso2R,
        torso2R: torso2L,
        leg0L: leg0R,
        leg0R: leg0L,
        leg1L: leg1R,
        leg1R: leg1L,
        leg2L: leg2R,
        leg2R: leg2L,
        arm0L: arm0R,
        arm0R: arm0L,
        arm1L: arm1R,
        arm1R: arm1L,
        arm2L: arm2R,
        arm2R: arm2L,
        background: background
    }

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


jointMirrorMap = {
    'Head':'Head',
    'Neck':'Neck',
    'Hips':'Hips',
    'UpArm_L':'UpArm_R',
    'UpArm_R':'UpArm_L',
    'LoArm_L':'LoArm_R',
    'LoArm_R':'LoArm_L',
    'Mit_L':'Mit_R',
    'Mit_R':'Mit_L',
    'UpLeg_L':'UpLeg_R',
    'UpLeg_R':'UpLeg_L',
    'LoLeg_L':'LoLeg_R',
    'LoLeg_R':'LoLeg_L',
    'Ankle_L':'Ankle_R',
    'Ankle_R':'Ankle_L',
    'Foot_L':'Foot_R',
    'Foot_R':'Foot_L',
    'Toe_L':'Toe_R',
    'Toe_R':'Toe_L'
}


# Load depth data from an exr file
def load_depth( filename ):
    import OpenEXR
    import Imath
    fileH = OpenEXR.InputFile(filename)

    # Compute the size
    dw = fileH.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R,G,B, Z) = [array.array('f', fileH.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B", "Z") ]

    depth = np.fromstring(fileH.channel("Z", FLOAT), dtype = np.float32)
    depth.shape = (size[1], size[0]) # Numpy arrays are (row, col)
    return depth / 10

# Reconstruct depth image
def reconstruct_depth_image(depth, labels):
    (M, N) = depth.shape
    rgb = 255.0 * np.ones((M,N,3))
    for m in range(M):
        for n in range(N):
            rgb[m,n,0] = (depth[m,n]-1.0)/5.0
            rgb[m,n,1] = (depth[m,n]-1.0)/5.0
            rgb[m,n,2] = (depth[m,n]-1.0)/5.0
    return rgb

# Find the closest label color
# This is needed because blender does interpolation on texture color
def find_closest(color):
    minKey = background
    minDist = 100000
    for key, value in colors.iteritems():
        diff = color*255.0 - value
        if (np.vdot(diff,diff)) < minDist:
            minDist = np.vdot(diff,diff)
            minKey = key

    return minKey

# Determine class labels for all pixels
def image_labels(img):
    (M,N,_) = img.shape
    labels = np.zeros((M,N), dtype=np.int32)

    for m in range(M):
        for n in range(N):
            colorId = find_closest(img[m][n])
            labels[m][n] = colorId
    return labels


def to_indices(image_id, where_ids):
    rows = where_ids[0]
    colms = where_ids[1]
    assert( len(rows) == len(colms) )
    indices = np.zeros((len(rows), 3),  dtype=np.int32)
    indices[:,0] = image_id
    indices[:,1] = rows.T
    indices[:,2] = colms.T
    return indices

def sample_pixels(depth, labels, number_datapoints):
    indices_array_complete = np.zeros((number_datapoints, 3), dtype=np.int32)
    data_points_per_label = number_datapoints / number_of_body_parts
    actual_number_datapoints = 0
    for label in range(number_of_body_parts):
        indices_array = to_indices(0, np.where(labels == label))
        np.random.shuffle(indices_array)
        m,n = indices_array.shape
        number_of_valid_datapoints = min(m,data_points_per_label)
        indices_array = indices_array[0:number_of_valid_datapoints, :]
        indices_array_complete[actual_number_datapoints:actual_number_datapoints + number_of_valid_datapoints, :] = indices_array
        actual_number_datapoints += number_of_valid_datapoints
    indices_array_complete = indices_array_complete[0:actual_number_datapoints, :]
    pixel_labels = labels[indices_array_complete[:, 1], indices_array_complete[:, 2]]

    return indices_array_complete, pixel_labels


# def build_closest_image(img, depth):
#     (M,N) = depth.shape
#     cloestImg = np.zeros(img.shape)

#     for m in range(M):
#         for n in range(N):
#             colorId = find_closest(img[m][n])
#             colorValueAsArray = get_color(colorId)
#             cloestImg[m][n] = colorValueAsArray / 255.0
#     return cloestImg


# Reconstruct image from labels
def reconstruct_label_image(depth, labels):
    (M,N) = depth.shape
    img = np.ones((M, N, 3))

    for body_part in range(number_of_body_parts):
        img[labels == body_part] =  colors[body_part] / 255.0

    return img


# Reconstruct image from labels
def reconstruct_label_image_ushort(depth, labels, ground_labels):
    (M,N) = depth.shape
    img = np.ones((M, N, 3), dtype=np.uint8)

    for body_part in range(number_of_body_parts):
        img[(labels == body_part)] =  np.array(colors[body_part], dtype=np.uint8)

    return img

# Reconstruct image from labels
def reconstruct_label_image_min_probability_ushort(depth, labels, ground_labels, probabilities, min_probablity):
    (M,N) = depth.shape
    img = np.ones((M, N, 3), dtype=np.uint8)

    for body_part in range(number_of_body_parts):
        img[(labels == body_part) & (probabilities > min_probablity)] = np.array(colors[body_part], dtype=np.uint8)

    return img


def classify_pixels(depth, forest):
    buffer_collection = buffers.BufferCollection()
    assert(depth.ndim == 2)
    m,n = depth.shape
    pixel_indices = np.array( list(itertools.product( np.zeros(1), range(m), range(n) )), dtype=np.int32 )
    buffer_collection.AddInt32MatrixBuffer(buffers.PIXEL_INDICES, buffers.as_matrix_buffer(pixel_indices))
    depth_buffer = buffers.as_tensor_buffer(depth)
    buffer_collection.AddFloat32Tensor3Buffer(buffers.DEPTH_IMAGES, depth_buffer)

    yprobs_buffer = buffers.Float32MatrixBuffer()
    forest_predictor = predict.ForestPredictor(forest)
    forest_predictor.PredictYs(buffer_collection, m*n, yprobs_buffer)
    yprobs = buffers.as_numpy_array(yprobs_buffer)
    (_, ydim) = yprobs.shape
    img_yprobs = yprobs.reshape((m,n,ydim))
    img_yhat = np.argmax(img_yprobs, axis=2)
    return img_yhat, img_yprobs.max(axis=2)

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
    print "Img %d of %d" % (imgId, numberOfImgs)
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

    counts = Parallel(n_jobs=10)(delayed(classification_accuracy_image)(imgId)
      for imgId in range(numberOfImgs))

    incorrectClassificationCount = sum([x[0] for x in counts])
    nonBackgroundCount = sum([x[1] for x in counts])

    return float(nonBackgroundCount - incorrectClassificationCount) / float(nonBackgroundCount)

def plot_classification_imgs(figures_path, depths, ground_labels, forest):
    (numberOfImgs,_,_) = depths.shape
    for imgId in range(numberOfImgs):
        print "Img %d of %d" % (imgId, numberOfImgs)
        labels, probs = classify_body_pixels(depths[imgId,:,:], ground_labels[imgId,:,:], forest)

        img = reconstruct_depth_image(depths[imgId,:,:], ground_labels[imgId,:,:])
        pl.imshow(img)
        pl.draw()
        pl.savefig(figures_path + "%d-depth.png" % (imgId))
        #pl.show()

        img = reconstruct_label_image(depths[imgId,:,:], ground_labels[imgId,:,:])
        pl.imshow(img)
        pl.draw()
        pl.savefig(figures_path + "%d-groundLabels.png" % (imgId))
        #pl.show()

        img = reconstruct_label_image_min_probability_ushort(depths[imgId,:,:], labels, ground_labels[imgId,:,:], probs, 0.0)
        pl.imshow(img)
        pl.draw()
        pl.savefig(figures_path + "%d-predictedLabels.png" % (imgId))
        #pl.show()

        img = reconstruct_label_image_min_probability_ushort(depths[imgId,:,:], labels, ground_labels[imgId,:,:], probs, 0.5)
        pl.imshow(img)
        pl.draw()
        pl.savefig(figures_path + "%d-predictedLabelsConfident.png" % (imgId))
        #pl.show()