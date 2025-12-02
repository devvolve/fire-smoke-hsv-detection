# 🔥 HSV-Based Fire & Smoke Detection (Mini Project 2)

Mini Project 2 for **CS 5190.01 – Computer Vision**  
Author: **Daniyal Dianati**  
Instructor: **Dr. Sai Chandra Kosaraju**  
Semester: **Fall 2025**

This project implements a **classical computer vision prototype** for detecting fire and smoke in images using
**HSV color thresholding**, binary masks, and contour-based bounding boxes.

It serves as the lightweight **“mock mode”** for a larger real-time Fire & Smoke Detection pipeline that will later
use **YOLOv8** for deep-learning-based detection.

---

## 📂 Project Structure

```text
fire-smoke-hsv-detection/
│
├── src/
│   └── hsv_detect.py          # HSV-based fire/smoke detection script
│
├── data/                      # Sample test images (fire, smoke, normal)
│   ├── fire1.jpg
│   ├── smoke1.jpg
│   └── normal1.jpg
│
├── results/                   # Masks, annotated outputs, JSON files
│
├── report/
│   └── Mini_Project_2_HSV_Report.docx
│
├── presentation/
│   └── Mini_Project_2_HSV_Presentation.pptx
│
├── requirements.txt
└── README.md

dd