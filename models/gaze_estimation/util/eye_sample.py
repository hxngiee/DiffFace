
class EyeSample:
    def __init__(self, orig_img, img, is_left, transform_inv, estimated_radius):
        self._orig_img = orig_img.copy()
        self._img = img.copy()
        self._is_left = is_left
        self._transform_inv = transform_inv
        self._estimated_radius = estimated_radius
    @property
    def orig_img(self):
        return self._orig_img

    @property
    def img(self):
        return self._img

    @property
    def is_left(self):
        return self._is_left

    @property
    def transform_inv(self):
        return self._transform_inv

    @property
    def estimated_radius(self):
        return self._estimated_radius