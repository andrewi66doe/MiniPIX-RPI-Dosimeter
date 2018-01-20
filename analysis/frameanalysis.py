from numpy import array, zeros
from math import sqrt

from analysis.boundingrect import qhull2d, min_bounding_rect

MINIPIX_HEIGHT = 256
MINIPIX_WIDTH = 256


class Calibration:
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None
        self.t = None

    def apply_calibration(self, frame):

        e_frame = zeros(65536).reshape(256, 256)

        for x in range(256):
            for y in range(256):
                energy = 0

                A = self.a[x][y]
                T = self.t[x][y]
                B = self.b[x][y] - A * T - frame[x][y]
                C = T * frame[x][y] - self.b[x][y] * T - self.c[x][y]

                if A != 0 and (B * B - 4.0 * A * C) >= 0:
                    energy = ((B * -1) + sqrt(B * B - 4.0 * A * C)) / 2.0 / A
                    if energy < 0:
                        energy = 0

                e_frame[x][y] = energy
        return e_frame

    def load_file(self, fobj):
        return [line.split for line in fobj.readline()]

    def load_calib_a(self, filename):
        with open(filename, 'r') as a:
            file_a = self.load_file(a)
            self.a = array(file_a)

    def load_calib_b(self, filename):
        with open(filename, 'r') as b:
            file_b = self.load_file(b)
            self.b = array(file_b)

    def load_calib_c(self, filename):
        with open(filename, 'r') as c:
            file_c = self.load_file(c)
            self.c = array(file_c)

    def load_calib_t(self, filename):
        with open(filename, 'r') as t:
            file_t = self.load_file(t)
            self.t = array(file_t)


class Frame:
    def __init__(self, framedata, calibration):
        self.cluster_count = 0
        self.clusters = []
        self.acq_time = None
        self.framedata = framedata
        self.data_array = None
        self.calibration = calibration

    def do_clustering(self):
        arr = array(self.framedata).reshape(256, 256)
        self.data_array = self.calibration.apply_calibration(arr)

        for x in range(256):
            for y in range(256):
                if self.data_array[x][y] > 0:
                    cluster = self._floodfill(x, y)
                    self.clusters.append(cluster)

    def _floodfill(self, x, y):
        to_fill = set()
        to_fill.add((x, y))

        cluster_pixels = []

        while not len(to_fill) == 0:
            x, y = to_fill.pop()

            pixel = self.data_array[y][x]
            self.data_array[x][y] = -1

            cluster_pixels.append((x, y, self.data_array[x][y]))

            for x, y in pixel.surrounding_pixels():
                if self.data_array[x][y].value > 0 and not self.data_array[x][y] == -1:
                    to_fill.add((x, y))

        return Cluster(cluster_pixels)

    def _surrounding_pixels(self, x, y):
        pixels = []

        if self._inbounds(x, y - 1):
            pixels.append((x, y - 1))
        if self._inbounds(x, y + 1):
            pixels.append((x, y + 1))
        if self._inbounds(x - 1, y):
            pixels.append((x - 1, y))
        if self._inbounds(x + 1, y):
            pixels.append((x + 1, y))
        if self._inbounds(x - 1, y + 1):
            pixels.append((x - 1, y + 1))
        if self._inbounds(x + 1, y + 1):
            pixels.append((x + 1, y + 1))
        if self._inbounds(x - 1, y - 1):
            pixels.append((x - 1, y - 1))
        if self._inbounds(x + 1, y - 1):
            pixels.append((x + 1, y - 1))

        return pixels

    @staticmethod
    def _inbounds(x, y):
        if x > MINIPIX_WIDTH - 1 or x < 0:
            return False
        if y > MINIPIX_HEIGHT - 1 or y < 0:
            return False

        return True


class Cluster:
    def __init__(self, indices):
        self.indices = indices

        self.bounding_rect = BoundingBox(indices)
        self.pixel_count = len(indices)
        self.lw_ratio = self.bounding_rect.height / self.bounding_rect.width
        self.density = self.pixel_count / self.bounding_rect.area

        self.energy = sum([index[2] for index in indices])
        self.track_length = self._get_track_length()
        self.LET = self.energy / self.track_length

    def _get_track_length(self):
        return None

    def _get_inner_pixels(self):
        return len(
            list(
                filter(lambda x: self._is_inner_pixel(x), self.indices)))

    def _is_inner_pixel(self, pixel):
        points = [(x[0], x[1]) for x in self.indices]

        x, y, _ = pixel

        count = 0

        if (x, y - 1) in points:
            count += 1
        if (x, y + 1) in points:
            count += 1
        if (x - 1, y) in points:
            count += 1
        if (x + 1, y) in points:
            count += 1
        if (x - 1, y + 1) in points:
            count += 1
        if (x + 1, y + 1) in points:
            count += 1
        if (x - 1, y - 1) in points:
            count += 1
        if (x + 1, y - 1) in points:
            count += 1

        return count > 4


class BoundingBox:
    def __init__(self, pixels):
        self.rotation = None
        self.area = None
        self.width = None
        self.height = None
        self.center = None
        self.corners = None
        self.pixels = pixels
        self._calculate()

    def _calculate(self):
        hull = qhull2d(array(self.pixels))
        hull = hull[::-1]
        self.rotation, self.area, self.width, self.height, self.center, self.corners = min_bounding_rect(hull)
