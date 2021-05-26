import cv2
from blobs.blob2 import Blob, get_centroid, box_contains_point, get_area
from collections import OrderedDict
from YOLO.yolo_detector import get_bounding_boxes
import multiprocessing as mp



cap = cv2.VideoCapture('./videos/video.mp4')

blobs = OrderedDict()
blob_id = 1
frame_counter = 0
DETECTION_FRAME_RATE = 5
MAX_CONSECUTIVE_TRACKING_FAILURES = 10

_, frame = cap.read()
initial_bboxes = get_bounding_boxes(frame)
for box in initial_bboxes:
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, tuple(box))
    _blob = Blob(box, tracker)
    blobs[blob_id] = _blob
    print(blobs)

f_height, f_width, _ = frame.shape
cl_y = round(1/7 * f_height)
counting_line = [(0, cl_y), (f_width, cl_y)]
vehicle_count = 0

while cap.get(cv2.CAP_PROP_POS_FRAMES) // mp.cpu_count() + 1 < cap.get(cv2.CAP_PROP_FRAME_COUNT):
    k = cv2.waitKey(1)
    print(cap.get(cv2.CAP_PROP_POS_FRAMES))
    _, frame = cap.read()

    for _id, blob in list(blobs.items()):
        success, box = blob.tracker.update(frame)
        if success:
            blob.num_consecutive_tracking_failures = 0
            blob.update(box)

            # draw and label bounding boxes
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'v_' + str(_id), (x, y - 2), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            blob.num_consecutive_tracking_failures += 1

        if blob.num_consecutive_tracking_failures >= MAX_CONSECUTIVE_TRACKING_FAILURES:
            del blobs[_id]

        if blob.centroid[1] <= cl_y and not blob.counted:
            blob.counted = True
            vehicle_count += 1


    if frame_counter >= DETECTION_FRAME_RATE:
        # rerun detection
        boxes = get_bounding_boxes(frame)

        # add new blobs
        for box in boxes:
            box_centroid = get_centroid(box)
            box_area = get_area(box)
            match_found = False
            for _id, blob in blobs.items():
                if (blob.area >= box_area and box_contains_point(blob.bounding_box, box_centroid)) \
                        or (box_area >= blob.area and box_contains_point(box, blob.centroid)):
                    match_found = True
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, tuple(box))
                    blob.update(box, tracker)
                    break

            if not match_found:
                blob_id += 1
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, tuple(box))
                _blob = Blob(box, tracker)
                blobs[blob_id] = _blob

        for id_a, blob_a in list(blobs.items()):
            for id_b, blob_b in list(blobs.items()):
                if blob_a == blob_b:
                    break

                if blob_a.area >= blob_b.area and box_contains_point(blob_a.bounding_box, blob_b.centroid):
                    del blobs[id_b]
                elif blob_b.area >= blob_a.area and box_contains_point(blob_b.bounding_box, blob_a.centroid):
                    del blobs[id_a]

        frame_counter = 0
    cv2.line(frame, counting_line[0], counting_line[1], (255, 0, 0), 3)
    cv2.putText(frame, 'Count: ' + str(vehicle_count), (int(f_width)-300, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

    resized_frame = cv2.resize(frame, (858, 480))
    cv2.imshow('Nilansh Tracking', resized_frame)

    frame_counter += 1
    if k & 0xFF == ord('q'):
        print('Video exited.')
        break

# end capture, close window
cap.release()
cv2.destroyAllWindows()
