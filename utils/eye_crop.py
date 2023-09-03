import face_alignment
import torch
def get_eye_coords(fa, image):
    image = image.squeeze(0)
    image = image * 128 + 128
    image = image.to(torch.uint8)
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy()

    try:
        preds = fa.get_landmarks(image)[0]
    except:
        return [None] * 8

    x, y = 5, 9
    left_eye_left = preds[36]
    left_eye_right = preds[39]
    eye_y_average = (left_eye_left[1] + left_eye_right[1]) // 2
    left_eye = [int(left_eye_left[0]) - x, int(eye_y_average - y), int(left_eye_right[0]) + x, int(eye_y_average + y)]
    right_eye_left = preds[42]
    right_eye_right = preds[45]
    eye_y_average = (right_eye_left[1] + right_eye_right[1]) // 2
    right_eye = [int(right_eye_left[0]) - x, int(eye_y_average - y), int(right_eye_right[0]) + x, int(eye_y_average + y)]
    return [*left_eye, *right_eye]
