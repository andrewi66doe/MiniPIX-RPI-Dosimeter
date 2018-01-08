from threading import Thread, Event
from queue import Queue
from random import randint
from time import sleep

DESIRED_DETECTOR_AREA_3_PERCENT = 1966  # 3% of the detector area in pixels
DESIRED_DETECTOR_AREA_4_PERCENT = 2621
DESIRED_DETECTOR_AREA_5_PERCENT = 3276


class MiniPIX(Thread):
    def __init__(self, variable=False, shutter_time=1, **kwargs):
        Thread.__init__(self, **kwargs)
        self.variable = variable
        self.shutter_time = shutter_time
        self.max_shutter_time = 4
        self.min_shutter_time = .01
        self.max_ramp_rate = 0
        self.data = Queue()
        self.stop_acquisitions = Event()
        self.shutdown = Event()

    @staticmethod
    def _take_aquisition(shutter_time):
        """
        :param shutter_time: Length of time to expose MiniPIX for
        :return:
        """
        sleep(shutter_time)
        # Test data for now since I don't actually have a MiniPix
        # acquisition = [[randint(0, 1) for _ in range(256)] for _ in range(256)]

        # Generate frames with 3 percent covered
        acquisition = []
        for _ in range(7):
            acquisition.append([1 for _ in range(256)])
        for _ in range(249):
            acquisition.append([0 for _ in range(256)])
        return acquisition

    @staticmethod
    def _total_hit_pixels(frame):
        """
        :param frame: Frame of acquired MiniPIX data
        :return:
        """
        total_hit_pixels = sum([x.count(1) for x in frame])
        return total_hit_pixels

    def _variable_frame_rate(self):
        shutter_time = self.shutter_time
        acq = self._take_aquisition(shutter_time)
        self.data.put(acq)
        count = self._total_hit_pixels(acq)

        while not self.stop_acquisitions.is_set():
            hit_rate = count/shutter_time
            shutter_time = DESIRED_DETECTOR_AREA_3_PERCENT/hit_rate

            if shutter_time < self.min_shutter_time:
                shutter_time = self.min_shutter_time
            if shutter_time > self.max_shutter_time:
                shutter_time = self.max_shutter_time

            print("ShutterTime: {} Count: {}".format(shutter_time, count))
            acq = self._take_aquisition(shutter_time)
            self.data.put(acq)
            count = self._total_hit_pixels(acq)

    def _constant_frame_rate(self):
        while not self.stop_acquisitions.is_set():
            acq = self._take_aquisition(self.shutter_time)
            self.data.put(acq)

    def _begin_acquisitions(self):
        if self.variable:
            self._variable_frame_rate()
        else:
            self._constant_frame_rate()

    def stop_acquisitions(self):
        self.stop_acquisitions.set()

    def start_acquisitions(self):
        self.stop_acquisitions.clear()

    def get_last_acquisition(self, block=True):
        return self.data.get(block=block)

    def run(self):
        while not self.shutdown.is_set():
            self._begin_acquisitions()


if __name__ == "__main__":
    minipix = MiniPIX(variable=True)
    minipix.start()
    for x in range(3):
        print("Retrieving acquisition")
        minipix.get_last_acquisition()

    print("Stopping acquisitions")
    minipix.stop_acquisitions()
    sleep(1)
    print("Restarting acquisitions")
    minipix.start_acquisitions()
    sleep(5)
    minipix.stop_acquisitions()
    minipix.shutdown.set()

