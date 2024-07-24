import matplotlib.pyplot as plt

# Example data (replace with your actual results)
models = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l', 'YOLOv8x']
parameters = [5, 10, 20, 40, 80]  # Replace with actual parameter counts
latency = [1.0, 1.5, 2.0, 2.5, 3.0]  # Replace with actual latency values
mAP = [35, 40, 45, 50, 55]  # Replace with actual mAP values

# Plot mAP vs Parameters
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(parameters, mAP, 'o-', label='YOLOv8')
for i, model in enumerate(models):
    plt.text(parameters[i], mAP[i], model, fontsize=12, ha='right')
plt.xlabel('Parameters (M)')
plt.ylabel('COCO mAP')
plt.title('COCO mAP vs Parameters')
plt.legend()

# Plot mAP vs Latency
plt.subplot(1, 2, 2)
plt.plot(latency, mAP, 'o-', label='YOLOv8')
for i, model in enumerate(models):
    plt.text(latency[i], mAP[i], model, fontsize=12, ha='right')
plt.xlabel('Latency (ms/img)')
plt.ylabel('COCO mAP')
plt.title('COCO mAP vs Latency')
plt.legend()

plt.tight_layout()
plt.show()
