from ultralytics import SAM

# Load a model
model = SAM("sam2_b.pt")

# Display model information (optional)
model.info()


# Segment with point prompt
results = model("videos/video_10.mp4", points=[150, 150], labels=[1], )

print(results)