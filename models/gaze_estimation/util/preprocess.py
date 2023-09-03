import numpy as np
import cv2
import util.gaze
from scipy.spatial.transform import Rotation as R

def preprocess_unityeyes_image(img, json_data):
    ow = 160
    oh = 96
    # Prepare to segment eye image
    ih, iw = img.shape[:2]
    ih_2, iw_2 = ih/2.0, iw/2.0

    heatmap_w = int(ow/2)
    heatmap_h = int(oh/2)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def process_coords(coords_list):
        coords = [eval(l) for l in coords_list]
        return np.array([(x, ih-y, z) for (x, y, z) in coords])
    
    interior_landmarks = process_coords(json_data['interior_margin_2d'])
    caruncle_landmarks = process_coords(json_data['caruncle_2d'])
    iris_landmarks = process_coords(json_data['iris_2d'])

    left_corner = np.mean(caruncle_landmarks[:, :2], axis=0)
    right_corner = interior_landmarks[8, :2]
    eye_width = 1.5 * abs(left_corner[0] - right_corner[0])
    eye_middle = np.mean([np.amin(interior_landmarks[:, :2], axis=0),
                          np.amax(interior_landmarks[:, :2], axis=0)], axis=0)

    # Normalize to eye width.
    scale = ow/eye_width

    translate = np.asmatrix(np.eye(3))
    translate[0, 2] = -eye_middle[0] * scale
    translate[1, 2] = -eye_middle[1] * scale

    rand_x = np.random.uniform(low=-10, high=10)
    rand_y = np.random.uniform(low=-10, high=10)
    recenter = np.asmatrix(np.eye(3))
    recenter[0, 2] = ow/2 + rand_x
    recenter[1, 2] = oh/2 + rand_y

    scale_mat = np.asmatrix(np.eye(3))
    scale_mat[0, 0] = scale
    scale_mat[1, 1] = scale

    angle = 0 #np.random.normal(0, 1) * 20 * np.pi/180
    rotation = R.from_rotvec([0, 0, angle]).as_matrix()

    transform = recenter * rotation * translate * scale_mat
    transform_inv = np.linalg.inv(transform)
    
    # Apply transforms
    eye = cv2.warpAffine(img, transform[:2], (ow, oh))

    rand_blur = np.random.uniform(low=0, high=20)
    eye = cv2.GaussianBlur(eye, (5, 5), rand_blur)

    # Normalize eye image
    eye = cv2.equalizeHist(eye)
    eye = eye.astype(np.float32)
    eye = eye / 255.0

    # Gaze
    # Convert look vector to gaze direction in polar angles
    look_vec = np.array(eval(json_data['eye_details']['look_vec']))[:3].reshape((1, 3))
    #look_vec = np.matmul(look_vec, rotation.T)

    gaze = util.gaze.vector_to_pitchyaw(-look_vec).flatten()
    gaze = gaze.astype(np.float32)

    iris_center = np.mean(iris_landmarks[:, :2], axis=0)

    landmarks = np.concatenate([interior_landmarks[:, :2],  # 8
                                iris_landmarks[::2, :2],  # 8
                                iris_center.reshape((1, 2)),
                                [[iw_2, ih_2]],  # Eyeball center
                                ])  # 18 in total

    landmarks = np.asmatrix(np.pad(landmarks, ((0, 0), (0, 1)), 'constant', constant_values=1))
    landmarks = np.asarray(landmarks * transform[:2].T) * np.array([heatmap_w/ow, heatmap_h/oh])
    landmarks = landmarks.astype(np.float32)

    # Swap columns so that landmarks are in (y, x), not (x, y)
    # This is because the network outputs landmarks as (y, x) values.
    temp = np.zeros((34, 2), dtype=np.float32)
    temp[:, 0] = landmarks[:, 1]
    temp[:, 1] = landmarks[:, 0]
    landmarks = temp

    heatmaps = get_heatmaps(w=heatmap_w, h=heatmap_h, landmarks=landmarks)

    assert heatmaps.shape == (34, heatmap_h, heatmap_w)

    return {
        'img': eye,
        'transform': np.asarray(transform),
        'transform_inv': np.asarray(transform_inv),
        'eye_middle': np.asarray(eye_middle),
        'heatmaps': np.asarray(heatmaps),
        'landmarks': np.asarray(landmarks),
        'gaze': np.asarray(gaze)
    }


def gaussian_2d(w, h, cx, cy, sigma=1.0):
    """Generate heatmap with single 2D gaussian."""
    xs, ys = np.meshgrid(
        np.linspace(0, w - 1, w, dtype=np.float32),
        np.linspace(0, h - 1, h, dtype=np.float32)
    )

    assert xs.shape == (h, w)
    alpha = -0.5 / (sigma ** 2)
    heatmap = np.exp(alpha * ((xs - cx) ** 2 + (ys - cy) ** 2))
    return heatmap


def get_heatmaps(w, h, landmarks):
    heatmaps = []
    for (y, x) in landmarks:
        heatmaps.append(gaussian_2d(w, h, cx=x, cy=y, sigma=2.0))
    return np.array(heatmaps)
