# Human Vision for Bimanual Manipulation

🌐 **Project Website:** [https://ian-chuang.github.io/human-vision/](https://ian-chuang.github.io/human-vision/)

### Dataset Visualization

Visualize the dataset + eye-tracking:
[👉 View Dataset](https://huggingface.co/spaces/iantc104/av_aloha_visualize_dataset?dataset=iantc104%2Fav_aloha_sim_thread_needle&episode=0)

## Code Structure

* **Flow Transformer policy:**
  `lerobot/lerobot/common/policies/flare/`

* **Gaze prediction and cropping:**
  `lerobot/lerobot/common/policies/gaze_vision_encoder.py`

## ⚙️ Installation

```bash
pip install -e ./gym_av_aloha
pip install -e ./lerobot
```

## Training and Evaluation

```bash
python lerobot/train_flare.py
```
