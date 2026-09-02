# Medical Image Analysis: Brain Tumor Segmentation & Classification

A MATLAB project for analyzing brain Magnetic Resonance Imaging (MRI) scans:
* **3D Segmentation (`Segmentation.m`):** Isolates the tumor volume from NIfTI scans (BraTS dataset) by combining FLAIR and T2w sequences through thresholding and 3D morphological operations, evaluated using the Dice Similarity Coefficient.
* **Multi-Class Classification (`Classification.m`):** Distinguishes between four classes (*Glioma*, *Meningioma*, *Pituitary Tumor*, and *No Tumor*) by extracting Histogram of Oriented Gradients (HOG) features and training three machine learning classifiers (Random Forest, Support Vector Machine, and Neural Network).

---

## Requirements

* **MATLAB** (R2020b or later recommended)
* Required MATLAB Toolboxes:
  * *Image Processing Toolbox*
  * *Statistics and Machine Learning Toolbox*
  * *Deep Learning Toolbox*

---

## Datasets & Setup

Download the datasets from the official sources:
* **BraTS (Segmentation):** Volumetric NIfTI scans available at [Medical Segmentation Decathlon](http://medicaldecathlon.com).
* **Brain Tumor MRI (Classification):** 2D labeled images and ground-truth JSON files available on [Roboflow Universe](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset).

Place the extracted files in the root folder according to this structure:

```text
Medical-Image-Analysis/
├── BRATS/
│   ├── imagesTr/*.nii.gz         # Multi-modal NIfTI MRI scans
│   └── labelsTr/*.nii.gz         # Ground-truth segmentation masks
└── dataset_classification/
    ├── train/                    # JPG training images + _groundtruth.json
    └── test/                     # JPG test images + _groundtruth.json
