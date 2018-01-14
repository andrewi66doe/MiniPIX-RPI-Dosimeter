
from threading import Thread, Event
from Queue import Queue
from time import sleep


DESIRED_DETECTOR_AREA_3_PERCENT = 1966  # 3% of the detector area in pixels
DESIRED_DETECTOR_AREA_4_PERCENT = 2621
DESIRED_DETECTOR_AREA_5_PERCENT = 3276


class MiniPIXAcquisition(Thread):
    def __init__(self, 
                 minipix,
                 pixet,
                 variable_frate=False,
                 shutter_time=1,
                 detector_area=DESIRED_DETECTOR_AREA_3_PERCENT,
                 **kwargs):
        """
        :param minipix: MiniPIX object
        :param variable_frate: Capture with a variable frame rate if set to true
        :param shutter_time: Initial shutter time
        :param detector_area: Detector area parameter used by variable frame rate algorithm
        """
        Thread.__init__(self, **kwargs)
        self.minipix = minipix
        self.pixet = pixet
        self.variable = variable_frate
        self.shutter_time = shutter_time
        self.detector_area = detector_area
        self.max_shutter_time = 2
        self.min_shutter_time = .03  # 30 frames per second
        self.max_ramp_rate = 0
        self.data = Queue()
        self.stop_acquisitions = Event()
        self.shutdown_flag = Event()

    def _take_aquisition(self):
        """
        :param shutter_time: Length of time to expose MiniPIX for
        :return:
        """

        self.minipix.doSimpleAcquisition(1, self.shutter_time, self.pixet.PX_FTYPE_AUTODETECT, "ouput.pmf")
        frame = self.minipix.lastAcqFrameRefInc()

        return frame.data()

    @staticmethod
    def _total_hit_pixels(frame):
        """
        :param frame: Frame of acquired MiniPIX data
        :return:
        """
        total_hit_pixels = [x > 0 for x in frame].count(True)
        return total_hit_pixels

    def _variable_frame_rate(self):
        acq = self._take_aquisition()
        self.data.put(acq)
        count = self._total_hit_pixels(acq)

        while not self.stop_acquisitions.is_set():
            hit_rate = count / self.shutter_time
            if hit_rate != 0:
                self.shutter_time = self.detector_area / hit_rate
            else:
                self.shutter_time = self.max_shutter_time

            if self.shutter_time < self.min_shutter_time:
                self.shutter_time = self.min_shutter_time
            if self.shutter_time > self.max_shutter_time:
                self.shutter_time = self.max_shutter_time

            acq = self._take_aquisition()
            self.data.put(acq)
            count = self._total_hit_pixels(acq)

    def _constant_frame_rate(self):
        while not self.stop_acquisitions.is_set():
            acq = self._take_aquisition()
            self.data.put(acq)

    def _begin_acquisitions(self):
        if self.variable:
            self._variable_frame_rate()
        else:
            self._constant_frame_rate()

    def pause_acquisitions(self):
        self.stop_acquisitions.set()

    def start_acquisitions(self):
        self.stop_acquisitions.clear()

    def shutdown(self):
        self.stop_acquisitions.set()
        self.shutdown_flag.set()

    def get_last_acquisition(self, block=True):
        return self.data.get(block=block)

    def run(self):
        while not self.shutdown_flag.is_set():
            self._begin_acquisitions()


if __name__ == "__main__":
    pixet = initialize_minipix()
    minipix = pixet.devices()[0]

    acquisitions = MiniPIXAcquisition(minipix, variable_frate=True)
    acquisitions.start()

    while acquisitions.is_alive():
        acquisitions.data.get()

    acquisitions.pause_acquisitions()
    sleep(1)
    acquisitions.start_acquisitions()
    sleep(5)
    acquisitions.pause_acquisitions()
    acquisitions.shutdown()
