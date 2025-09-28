import kagglehub

# Download latest version
path = kagglehub.dataset_download("ritikagiridhar/2000-hand-gestures")

print("Path to dataset files:", path)