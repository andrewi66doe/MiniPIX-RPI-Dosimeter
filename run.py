import smbus
import pypixet

from picamera import PiCamera
from time import sleep, strftime
from Queue import Empty

from settings import i2CBUS
from orientation.bmp280 import get_environmentals
from acquisition.minipixacquisition import MiniPIXAcquisition

# Initialize i2c devices, minipix, logging facilities etc.
def init():
    pass


class RPIDosimeter:
    # Initialize i2c devices, minipix, logging facilities etc.
    def __init__(self):
        pypixet.start()
        pixet = pypixet.pixet
        device = pixet.devices()[0]

        if device.fullName() != "MiniPIX H06-W0239":
            print("No minipix found exiting...")
            exit(0)
        device.loadConfigFromFile("H06-W0239.xml")

        print("Found device: {}".format(device.fullName()))

        self.minipix = MiniPIXAcquisition(device,pixet, variable_frate=True)
        self.bus = smbus.SMBus(i2CBUS)
        self.camera = PiCamera()

    def capture_image(self):
        filename = strftime("%Y-%m-%d-%H:%M:%S")
        self.camera.capture("/home/pi/images/" + filename + ".jpg")
        sleep(2)

    def main(self):
        #self.capture_image()
        self.minipix.start()

        while self.minipix.is_alive():
            try:
                acq = self.minipix.get_last_acquisition(block=False)
                print("Received acquisition in main thread...")
            except Empty:
                pass
            temp, pressure = get_environmentals(self.bus)
            print("Temperature: {0:.2f} C Pressure: {1:.2f} mbar".format(temp, pressure))

    def shutdown(self):
        print("Stopping acquisitions...")
        self.minipix.shutdown()
        print("Exiting main thread...")


if __name__ == "__main__":
    app = RPIDosimeter()
    try:
        app.main()
    except KeyboardInterrupt:
        app.shutdown()
