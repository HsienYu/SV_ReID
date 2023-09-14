import argparse
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

COLORS = sv.ColorPalette.default()

ZONE_IN_POLYGONS = [
    np.array([
        [145, 178], [141, 550], [393, 546], [389, 166]
    ])
]

ZONE_OUT_POLYGONS = [
    np.array([
        [845, 154], [841, 546], [1125, 546], [1125, 150]
    ])
]


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)

        detections_all.class_id = np.vectorize(
            lambda x: self.tracker_id_to_zone_id.get(x, -1)
        )(detections_all.tracker_id)
        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    frame_resolution_wh: Tuple[int, int],
    triggering_position: sv.Position = sv.Position.CENTER,
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=frame_resolution_wh,
            triggering_position=triggering_position,
        )
        for polygon in polygons
    ]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO ByteTrack")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("./model/yolov8n.pt")
    model.to('cuda:0')

    tracker = sv.ByteTrack()

    resolution_wh = (frame_width, frame_height)

    zones_in = initiate_polygon_zones(
        ZONE_IN_POLYGONS, resolution_wh, sv.Position.CENTER
    )
    zones_out = initiate_polygon_zones(
        ZONE_OUT_POLYGONS, resolution_wh, sv.Position.CENTER
    )
    box_annotator = sv.BoxAnnotator(color=COLORS)

    trace_annotator = sv.TraceAnnotator(
        color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
    )
    detections_manager = DetectionsManager()

    def annotate_frame(
        frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        for i, (zone_in, zone_out) in enumerate(zip(zones_in, zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = trace_annotator.annotate(
            annotated_frame, detections)
        annotated_frame = box_annotator.annotate(
            annotated_frame, detections, labels
        )

        for zone_out_id, zone_out in enumerate(zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in detections_manager.counts:
                counts = detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(
                        detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(
                        x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )

        return annotated_frame


    while cap.isOpened():
        ret, frame = cap.read()


        results = model(
            frame, verbose=True, agnostic_nms=True, classes=0,
            conf=0.3, iou=0.7
        )[0]
        detections = sv.Detections.from_ultralytics(results)
        detections.class_id = np.zeros(len(detections))
        detections = tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []

        for i, (zone_in, zone_out) in enumerate(zip(zones_in, zones_out)):
            detections_in_zone = detections[zone_in.trigger(
                detections=detections)]
            detections_in_zones.append(detections_in_zone)
            detections_out_zone = detections[zone_out.trigger(
                detections=detections)]
            detections_out_zones.append(detections_out_zone)

        detections = detections_manager.update(
            detections, detections_in_zones, detections_out_zones
        )

        cv2.imshow("SV_ReID", annotate_frame(frame, detections))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
