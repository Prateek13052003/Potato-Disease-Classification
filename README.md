
# 🥔 Potato Disease Classification using Deep Learning

A Convolutional Neural Network (CNN) based image classification project that detects potato leaf diseases:

* 🌿 Early Blight
* 🍂 Late Blight
* ✅ Healthy

Built using **TensorFlow & Keras**.

---

## 📌 Dataset

Dataset Source:
👉 [https://www.kaggle.com/arjuntejaswi/plant-village](https://www.kaggle.com/arjuntejaswi/plant-village)

The dataset contains potato leaf images categorized into 3 classes:

* Potato___Early_blight
* Potato___Late_blight
* Potato___healthy

---

## 🚀 Project Workflow

### 1️⃣ Data Preparation

* Dataset split using `split-folders` tool:

```bash
pip install split-folders
splitfolders --ratio 0.8 0.1 0.1 -- ./training/PlantVillage/
```

Split Ratio:

* 80% Training
* 10% Validation
* 10% Testing

---

### 2️⃣ Data Augmentation

Used `ImageDataGenerator` for:

* Rescaling (1./255)
* Rotation
* Horizontal Flip

```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True
)
```

---

### 3️⃣ Model Architecture

Custom CNN Architecture:

* 6 Convolutional Layers
* MaxPooling after each Conv layer
* Flatten layer
* Dense layer (64 units)
* Output layer (Softmax – 3 classes)

Total Parameters: **183,747**

---

### 4️⃣ Model Compilation

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
```

---

### 5️⃣ Training Results

* Epochs: 20
* Final Training Accuracy: **97.15%**
* Final Validation Accuracy: **95.83%**
* Test Accuracy: **97.44%**
* Test Loss: **0.076**

---

### 📊 Training Graphs

* Accuracy Curve
* Loss Curve

(Generated using Matplotlib)

---

## 🔍 Prediction (Inference)

Sample prediction logic:

```python
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence
```

---

## 💾 Saving the Model

Model saved in `.h5` format:

```python
model.save("../potatoes.h5")
```

This allows easy deployment to:

* FastAPI
* GCP
* Docker
* Cloud platforms

---

## 🛠️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* PIL
* ImageDataGenerator

---

## 📈 Performance

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 97.15% |
| Validation Accuracy | 95.83% |
| Test Accuracy       | 97.44% |

---

## 🌟 Future Improvements

* Add Transfer Learning (ResNet / EfficientNet)
* Deploy using FastAPI
* Build React frontend
* Add Grad-CAM visualization
* Convert to TensorFlow Lite for mobile deployment

---

## 👨‍💻 Author

Prateek Choudhary

---
