import smbus

from picamera import PiCamera
from time import sleep, strftime

from settings import i2CBUS
from orientation.bmp280 import get_environmentals


# Initialize i2c devices, minipix, logging facilities etc.
def init():
    pass


class RPIDosimeter:
    # Initialize i2c devices, minipix, logging facilities etc.
    def __init__(self):
        self.bus = smbus.SMBus(i2CBUS)
        self.camera = PiCamera()

    def capture_image(self):
        filename = strftime("%Y-%m-%d-%H:%M:%S")
        self.camera.capture("/home/pi/images/" + filename + ".jpg")
        sleep(2)

    def main(self):
        temp, pressure = get_environmentals(self.bus)
        print("Temperature: {0:.2f} C Pressure: {1:.2f} mbar".format(temp, pressure))
        self.capture_image()

if __name__ == "__main__":
    app = RPIDosimeter()
    app.main()
