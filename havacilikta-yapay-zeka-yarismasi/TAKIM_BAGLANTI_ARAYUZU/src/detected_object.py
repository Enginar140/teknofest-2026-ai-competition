class DetectedObject:
    def __init__(
        self,
        cls: int,
        landing_status: int,
        motion_status: int,
        top_left_x: float,
        top_left_y: float,
        bottom_right_x: float,
        bottom_right_y: float,
    ):
        self.cls = int(cls)
        self.landing_status = str(landing_status)
        self.motion_status = str(motion_status)
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y

    def create_payload(self, evaulation_server):
        # API sınıf URL'si: Teknofest sınıf id 0–3 → çoğu sunucu 1–4 endpoint kullanır
        cls_idx = int(self.cls) + 1
        return {
            "cls": self.generate_api_url("classes/", str(cls_idx), evaulation_server),
            "landing_status": str(self.landing_status),
            "motion_status": str(self.motion_status),
            "top_left_x": str(self.top_left_x),
            "top_left_y": str(self.top_left_y),
            "bottom_right_x": str(self.bottom_right_x),
            "bottom_right_y": str(self.bottom_right_y),
        }

    @staticmethod
    def generate_api_url(cls_endpoint, cls_id, evaulation_server):
        """Sınıf için API URL (taban adres sonunda tek /)."""
        base = evaulation_server.rstrip("/") + "/"
        return base + cls_endpoint + cls_id + "/"
