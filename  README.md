# 🏥 Pneumonia Detection from Chest X-Rays

A deep learning project that detects pneumonia from chest X-ray images using Convolutional Neural Networks (CNN) and Transfer Learning with ResNet50. This system provides real-time analysis through a web interface.

## 📋 Project Overview

This project demonstrates a complete medical AI pipeline:
- Data exploration and analysis of chest X-ray images
- CNN model development from scratch
- Transfer learning with pre-trained models (VGG16, ResNet50)
- Model interpretation and performance evaluation
- Flask web application deployment for real-time predictions

## 🚀 Features

- **Deep Learning Models**: CNN, VGG16, ResNet50
- **Medical-Grade Performance**: 83.3% recall (minimizes missed pneumonia cases)
- **Real-Time Analysis**: Web interface for instant X-ray analysis
- **Model Interpretability**: Understanding model decision-making process
- **Professional Interface**: Medical disclaimer and recommendations

## 📊 Results

### Model Performance Comparison
| Model | Accuracy | Recall | Precision | F1-Score |
|-------|----------|--------|-----------|----------|
| CNN Baseline | 73.2% | 58.5% | 97.9% | 73.2% |
| VGG16 | ~85% | ~75% | ~95% | ~84% |
| **ResNet50** | **~85%** | **83.3%** | **~90%** | **~86%** |

### Key Achievements
- **83.3% recall** - Critical for medical safety (few missed cases)
- **Real-time predictions** - < 5 seconds per X-ray
- **Web deployment** - Accessible interface for healthcare professionals

## 🛠️ Tech Stack

- **Programming Language**: Python 3.8+
- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Web Framework**: Flask
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

## 📁 Project Structure
