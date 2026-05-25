from enum import IntEnum


class DetectClass(IntEnum):
    CAR = 2
    MOTORCYCLE = 3
    BUS = 5
    TRUCK = 7
    TRAFFIC_LIGHT = 9

DETECTION_CLASSES = {
    DetectClass.CAR,
    DetectClass.MOTORCYCLE,
    DetectClass.BUS,
    DetectClass.TRUCK,
    DetectClass.TRAFFIC_LIGHT
}

VEHICLE_CLASSES = {
    DetectClass.CAR,
    DetectClass.MOTORCYCLE,
    DetectClass.BUS,
    DetectClass.TRUCK,
}

TRAFFIC_LIGHT_CLASSES = {
    DetectClass.TRAFFIC_LIGHT
}

VEHICLE_HEIGHTS = {
        DetectClass.CAR : 1.5,
        DetectClass.MOTORCYCLE : 1.1,
        DetectClass.BUS : 3.2,
        DetectClass.TRUCK : 9
    }