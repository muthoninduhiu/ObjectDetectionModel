# Define an empty list to store the object detection results
from yolo import results

detections = []

# Iterate over the results object and extract the relevant information
for i in range(len(results.xyxy[0])):
    bbox = results.xyxy[0][i]
    label = results.names[int(bbox[5])]
    score = bbox[4]
    x1, y1, x2, y2 = map(int, bbox[:4])

    # Append the object detection results to the detections list
    detections.append({
        'label': label,
        'score': score,
        'bbox': (x1, y1, x2, y2)
    })


