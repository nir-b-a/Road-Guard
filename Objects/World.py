from dataclasses import dataclass, field
from Objects.Vehicle import Vehicle
from Objects.TrafficLight import TrafficLight


@dataclass
class FrameData:
    vehicle_ids: list[int] = field(default_factory=list)
    traffic_light_ids: list[int] = field(default_factory=list)




class World:
    def __init__(self, frame_count):
        # we dont need these
        #self.vehicle_ids: list[int] = []
        #self.traffic_light_ids: list[int] = []

        self.frame_count = frame_count

        self.vehicles: dict[int, Vehicle] = {}
        self.traffic_lights: dict[int, TrafficLight] = {}

        self.objects_in_frame: dict[int, FrameData] = {}


    """def addVehicle(self, vehicle: Vehicle):
        self.vehicles[vehicle.id] = vehicle"""
    
    def addVehicle(self, id, vehicle_type, start_frame):
        veh = Vehicle(id, vehicle_type, start_frame, start_frame)
        self.vehicles[veh.id] = veh
        return veh

    def addTrafficLight(self, id, start_frame):
        trl = TrafficLight(id, start_frame, start_frame)
        self.traffic_lights[trl.id] = trl
        return trl

    def registerFrame(self, frame_id: int, vehicle_ids: list[int], traffic_light_ids: list[int]):
        self.objects_in_frame[frame_id] = FrameData(vehicle_ids, traffic_light_ids)

    # Updates the bounding box of the objects and their end_frame. (we may need to add a code that checks if yolo "forgot" some object and by that set his end_frame)
    #def updateObjectsData(self, ):


    def getVehiclesInFrame(self, frame_id: int) -> list[Vehicle]:
        frame = self.objects_in_frame.get(frame_id)
        if not frame:
            return []
        return [self.vehicles[vid] for vid in frame.vehicle_ids if vid in self.vehicles]
    
    def getTrafficLightsInFrame(self, frame_id: int) -> list[TrafficLight]:
        frame = self.objects_in_frame.get(frame_id)
        if not frame:
            return []
        return [self.traffic_lights[vid] for vid in frame.traffic_light_ids if vid in self.traffic_lights]

    def getVehicle(self, vehicle_id: int) -> Vehicle | None:
        return self.vehicles.get(vehicle_id)
    
    def getTrafficLight(self, traffic_light_id: int) -> TrafficLight | None:
        return self.traffic_lights.get(traffic_light_id)
