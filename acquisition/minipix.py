import pypixet


def initialize_minipix():
    pypixet.start()
    pixet = pypixet.pixet
    return pixet
