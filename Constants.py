

class LPR:
    # Minimum vehicle bounding-box area (pixels²) before attempting plate recognition.
    # ~150×100px — below this the plate is too small to read reliably.
    MIN_VEHICLE_AREA = 15000


class DetectClass:
    Car = 2
    Motorcycle = 3
    Bus = 5
    Truck = 7
    Traffic_Light = 9
    Detection_Classes = [Car, Motorcycle, Bus, Truck, Traffic_Light]
    Vehicle_Classes = [Car, Motorcycle, Bus, Truck]
    Traffic_Light_Class = [Traffic_Light]
