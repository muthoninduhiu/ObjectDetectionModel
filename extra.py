# # Get the predicted bounding boxes, labels, and scores
# boxes = results.xyxy[0].cpu().numpy().tolist()
# labels = results.names
# scores = results.xyxy[0][:, 4].cpu().numpy().tolist()
# print("Boxes:{}\nLabels:{}\nScores:{}".format(boxes, labels, scores))
# # Plot the image and the predicted boxes
# # Loop through the boxes and draw them on the image
# for box, label, score in zip(boxes, labels, scores):
#     if score > 0.1 and label in classes:
#         # Only display boxes with confidence score > 0.1 and class in the defined classes
#         x1, y1, x2, y2 = map(int, box[:4])
#         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         cv2.putText(img, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
#
# Show the image