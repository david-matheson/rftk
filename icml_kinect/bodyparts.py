import numpy as np

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

# Find the closest label color
# This is needed because blender does interpolation on texture color
def find_closest(color):
    minKey = -1
    minDist = 100000
    for key, value in colors.iteritems():
        diff = color*255.0 - value
        if (np.vdot(diff,diff)) < minDist:
            minDist = np.vdot(diff,diff)
            minKey = key

    return minKey

# Determine class labels for all pixels
def image_labels(img, depth):
    (M,N) = depth.shape
    labels = np.zeros((M,N), dtype=np.uint8)

    for m in range(M):
        for n in range(N):
            colorId = find_closest(img[m][n])
            labels[m][n] = colorId
    return labels

def build_closest_image(img, depth):
    (M,N) = depth.shape
    cloestImg = np.zeros(img.shape)

    for m in range(M):
        for n in range(N):
            colorId = find_closest(img[m][n])
            colorValueAsArray = get_color(colorId)
            cloestImg[m][n] = colorValueAsArray / 255.0
    return cloestImg


# Reconstruct image from pipeline labels
def reconstruct_label_image(depth, labels):
    (M,N) = depth.shape
    img = np.ones((M, N, 3))

    for body_part in range(number_of_body_parts):
        img[labels == body_part] =  colors[body_part] / 255.0

    return img


# Reconstruct image from pipeline labels
def reconstruct_label_image_ushort(depth, labels):
    (M,N) = depth.shape
    img = np.ones((M, N, 3), dtype=np.uint8)

    for body_part in range(number_of_body_parts):
        img[labels == body_part] =  np.array(colors[body_part], dtype=np.uint8)

    return img

# Reconstruct image from pipeline labels
def reconstruct_label_image_min_probability_ushort(depth, labels, probabilities, min_probablity):
    (M,N) = depth.shape
    img = np.ones((M, N, 3), dtype=np.uint8)

    for body_part in range(number_of_body_parts):
        img[(labels == body_part) & (probabilities > min_probablity)] = np.array(colors[body_part], dtype=np.uint8)

    return img
