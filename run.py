import smbus
import pypixet

from numpy import array, nonzero
from numpy import sum as npsum
from picamera import PiCamera
from time import sleep, strftime

from settings import i2CBUS
from orientation.bmp280 import get_environmentals
from acquisition.minipixacquisition import MiniPIXAcquisition, take_acquisition
from analysis.frameanalysis import Frame, Calibration
from downlink.processcmd import HASPCommandHandler, SerialConnectionTest


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
        self.pixet = pypixet.pixet
        self.device = self.pixet.devices()[0]

        if self.device.fullName() != "MiniPIX H06-W0239":
            print("No minipix found exiting...")
            exit(0)
        self.device.loadConfigFromFile("H06-W0239.xml")

        print("Found device: {}".format(self.device.fullName()))

        # Allows for retrieval of MiniPIX frames at regular intervals
        self.minipix = MiniPIXAcquisition(self.device, self.pixet, variable_frate=False, shutter_time=1)
        self.minipix.daemon = True

        # Allows for regular handling of uplink commmands from HASP
        self.serial_connection = SerialConnectionTest()
        self.cmd_handler = HASPCommandHandler(self.serial_connection)
        self.cmd_handler.start()

        self.bus = smbus.SMBus(i2CBUS)

        self.running = False

    def __del__(self):
        self.serial_connection.close()


    def capture_image(self):
        filename = strftime("%Y-%m-%d-%H:%M:%S")
        self.camera.capture("/home/pi/images/" + filename + ".jpg")
        sleep(2)

    def main(self):
        # self.capture_image()
        self.minipix.start()
        self.running = True

        while True:
            # If there's an acquisition available for analysis
            acq, count = self.minipix.get_last_acquisition(block=True)
            arr = array(acq) 
            energy = self.calibration.apply_calibration(arr)
                
            frame = Frame(array(energy))
            if count > 0:
                frame.do_clustering()
            total_energy = npsum(energy[nonzero(energy)]) 
            dose = (total_energy/96081.3)/self.minipix.shutter_time
            print("Pixel Count: {} Clusters: {} Total Energy: {:.5f} DoseRate: {}".format(count, frame.cluster_count, total_energy, dose*60))

            for i, cluster in enumerate(frame.clusters):
                print("\tCluster: {} Density: {:.2f} energy: {:.5f}".format(i, cluster.density, cluster.energy))
    
            #temp, pressure = get_environmentals(self.bus)
            #print("Temperature: {0:.2f} C Pressure: {1:.2f} mbar".format(temp, pressure))

    def shutdown(self):
        print("Stopping acquisitions...")
        self.minipix.shutdown()
        self.minipix.join()
        print("Exiting main thread...")
        print("Stopping HASP command handler thread...")
        self.cmd_handler.shutdown_flag.set()
        self.cmd_handler.join()
        # Wait for minipix to shutdown properly
        sleep(2)

if __name__ == "__main__":
    app = RPIDosimeter()
    try:
        app.main()
    except KeyboardInterrupt:
        app.shutdown()
