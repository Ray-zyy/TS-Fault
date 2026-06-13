# Trained checkpoints

GitHub's file-size limits make it impractical to version the model weights here, so the **trained checkpoints are hosted on Hugging Face** instead:

https://huggingface.co/Ray6666/TS-Fault

Download them into this folder and the evaluation scripts will pick them up automatically:

```python
from huggingface_hub import snapshot_download
snapshot_download("Ray6666/TS-Fault", local_dir="checkpoints")
```

```bash
# or with the CLI
huggingface-cli download Ray6666/TS-Fault --local-dir checkpoints
```
