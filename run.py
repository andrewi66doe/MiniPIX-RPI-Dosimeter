import smbus
import pypixet

from numpy import array, nonzero
from picamera import PiCamera
from time import sleep, strftime

from settings import i2CBUS
from orientation.bmp280 import get_environmentals
from acquisition.minipixacquisition import MiniPIXAcquisition
from analysis.frameanalysis import Frame, Calibration


class RPIDosimeter:
    # Initialize i2c devices, minipix, logging facilities etc.
    def __init__(self):
        self.calibration = Calibration()
        self.calibration.load_calib_a("a.txt")
        self.calibration.load_calib_b("b.txt")
        self.calibration.load_calib_c("c.txt")
        self.calibration.load_calib_t("t.txt")
        # Initialize miniPIX driver subsystem
        pypixet.start()
        pixet = pypixet.pixet
        device = pixet.devices()[0]

        if device.fullName() != "MiniPIX H06-W0239":
            print("No minipix found exiting...")
            exit(0)
        device.loadConfigFromFile("H06-W0239.xml")

        print("Found device: {}".format(device.fullName()))

        self.minipix = MiniPIXAcquisition(device, pixet, variable_frate=False, shutter_time=5)
        self.minipix.daemon = True
        self.bus = smbus.SMBus(i2CBUS)
        #self.camera = PiCamera()

        self.running = False

    def capture_image(self):
        filename = strftime("%Y-%m-%d-%H:%M:%S")
        self.camera.capture("/home/pi/images/" + filename + ".jpg")
        sleep(2)

    def main(self):
        # self.capture_image()
        self.minipix.start()
        self.running = True

        while self.minipix.is_alive():
            # If there's an acquisition available for analysis
            if not self.minipix.data.empty():
                acq, count = self.minipix.get_last_acquisition(block=True)
                arr = array(acq, dtype='float32')
                energy = self.calibration.apply_calibration(arr)

                frame = Frame(energy)
                if count > 0:
                    frame.do_clustering()
                dose = (sum(energy[nonzero(energy)])/96081.3)/self.minipix.shutter_time
                print("Pixel Count: {} Clusters: {} Total Energy: {} DoseRate: {}".format(count, frame.cluster_count, sum(energy), dose*60))

                for i, cluster in enumerate(frame.clusters):
                    print("\tCluster: {} Density: {} energy: {}".format(i, cluster.density, cluster.energy))
    
            #temp, pressure = get_environmentals(self.bus)
            #print("Temperature: {0:.2f} C Pressure: {1:.2f} mbar".format(temp, pressure))

    def shutdown(self):
        print("Stopping acquisitions...")
        self.minipix.shutdown()
        print("Exiting main thread...")
        # Wait for minipix to shutdown properly
        sleep(2)

if __name__ == "__main__":
    app = RPIDosimeter()
    try:
        app.main()
    except KeyboardInterrupt:
        app.shutdown()
