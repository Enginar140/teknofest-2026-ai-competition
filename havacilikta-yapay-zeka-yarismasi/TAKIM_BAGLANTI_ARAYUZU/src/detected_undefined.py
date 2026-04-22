"""Görev 3: Tanımsız (referans) nesne tespitleri için sunucu payload."""


class DetectedUndefinedObject:
    def __init__(
        self,
        object_id: str,
        top_left_x: float,
        top_left_y: float,
        bottom_right_x: float,
        bottom_right_y: float,
    ):
        self.object_id = object_id
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y

    def create_payload(self, evaluation_server: str) -> dict:
        return {
            "object_id": str(self.object_id),
            "top_left_x": str(self.top_left_x),
            "top_left_y": str(self.top_left_y),
            "bottom_right_x": str(self.bottom_right_x),
            "bottom_right_y": str(self.bottom_right_y),
        }
