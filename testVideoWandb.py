import wandb
import numpy as np

wandb.init(project="your_project_name")

# Create a dummy video
video = np.random.randint(0, 255, (10, 3, 128, 128), dtype=np.uint8)

# Log video
wandb.log({"video": wandb.Video(video, fps=4, format="mp4")})

wandb.finish()
