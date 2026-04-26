# 🦟 Photonic Mosquito Defense System

> An AI-powered mosquito species detection and targeting system 
> built for real-world vector control applications.

## Motivation

In 2018, I spent two months bedridden with malaria in Nigeria. 
That experience gave me a direct understanding of what vector-borne 
disease costs a person — and motivated me to explore whether precision 
engineering could offer a fundamentally different approach to mosquito 
control: one that is targeted, chemical-free, and intelligent.

*Anopheles gambiae* — the primary malaria vector in sub-Saharan Africa 
— is one of the seven species this system is trained to identify.

## What This System Does

- Detects and classifies 6 mosquito species + non-mosquito objects in real time
- Tracks individual mosquitoes across video frames with persistent IDs
- Predicts trajectory and overlays a simulated targeting reticle
- Logs all detection events with timestamps and confidence scores

## Species Detected

| Species | Medical Relevance |
|---|---|
| Anopheles gambiae | Primary malaria vector (sub-Saharan Africa) |
| Anopheles funestus | Major malaria vector |
| Anopheles other | Secondary malaria vectors |
| Aedes | Dengue, Zika, Yellow Fever |
| Culex | Lymphatic filariasis |
| Mansonia | Filariasis, some encephalitis |
| Non-mosquito | Negative class |

## System Architecture
Video Input → YOLOv8 Detector → Species Classifier
↓
Multi-Object Tracker (ByteTrack)
↓
Trajectory Predictor (Kalman Filter)
↓
Simulated Targeting Reticle + Event Logger

## Current Status

- [x] Dataset acquired and validated (2,719 training instances, 7 classes)
- [x] YOLOv8n model trained on mosquito species dataset
- [ ] Real-time detection pipeline
- [ ] Multi-object tracking with persistent IDs
- [ ] Trajectory prediction and targeting overlay
- [ ] Wing-beat frequency classifier (audio)
- [ ] ROS 2 simulation bridge

## Tech Stack

- Python 3.x
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy / SciPy
- FilterPy (Kalman filtering)

## Dataset

Species-labeled mosquito dataset from Roboflow Universe  
License: CC BY 4.0  
Classes: 7 | Train: 2,719 | Val: 473 | Test: 806

## Roadmap

**Phase 1 (Now):** Software prototype — detection, tracking, simulated targeting  
**Phase 2 (3–6 months):** ROS 2 simulation, embedded system design  
**Phase 3 (Future):** Raspberry Pi / Jetson Nano deployment, hardware integration  

## Author
Madueke Martins
