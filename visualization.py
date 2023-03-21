import cv2
from prediction_results import detections
from yolo import results, img

# Iterate over the results object and display the image with bounding boxes around each detected object
for i in range(len(results.xyxy[0])):
    bbox = results.xyxy[0][i]
    label = results.names[int(bbox[5])]
    score = bbox[4]

    # Only draw labels with a score of 0.5 and above
    if score >= 0.4:
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 21, 255), 3)
        cv2.putText(img, f"{label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display the labels separately for each object
        for detection in detections:
            if detection['bbox'] == list(bbox):
                labels = detection['label'].split(',')
                y_offset = 20
                for label in labels:
                    cv2.putText(img, label.strip(), (x1, y1 - y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 10, 255), 2)
                    y_offset += 20

# Print the object detection results
for detection in detections:
    print(f"Label: {detection['label']}, Score: {detection['score']:.2f}, Boxes: {detection['bbox']}")

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
