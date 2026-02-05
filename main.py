import os
import subprocess
import sys


# ==========================================
# 0. AUTO-INSTALL MISSING LIBRARIES
# ==========================================
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


required_packages = ["numpy", "matplotlib", "opencv-python", "tensorflow", "scikit-learn", "seaborn"]
print("Checking and installing required libraries... (This may take a minute)")
for package in required_packages:
    try:
        __import__(package.split("-")[0] if package != "opencv-python" and package != "scikit-learn" else (
            "cv2" if package == "opencv-python" else "sklearn"))
    except ImportError:
        print(f"Installing {package}...")
        install_package(package)

# Now import everything safely
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


# ==========================================
# 1. ADVANCED DATA GENERATION (Physics Based)
# ==========================================
class AdvancedMissileGenerator:
    def __init__(self, img_size=64):
        self.img_size = img_size

    def generate_curve(self, img):
        # Generate a Bezier-like curve (Simulating wind effect on missile)
        p0 = np.random.randint(5, self.img_size - 5, 2)
        p2 = np.random.randint(5, self.img_size - 5, 2)
        p1 = np.random.randint(0, self.img_size, 2)

        # Calculate curve points
        t = np.linspace(0, 1, num=50)
        curve_x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        curve_y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]

        points = np.vstack((curve_x, curve_y)).astype(np.int32).T
        cv2.polylines(img, [points], isClosed=False, color=1.0, thickness=2)

        # Return Normalized Start and End points for Regression
        return np.array([p0[0], p0[1], p2[0], p2[1]]) / self.img_size

    def create_dataset(self, num_samples=3000):
        X = []
        y_class = []
        y_bbox = []

        print(f"Generating {num_samples} High-Fidelity Samples...")

        for _ in range(num_samples):
            noise_level = np.random.uniform(0.1, 0.25)
            img = np.random.normal(0.5, noise_level, (self.img_size, self.img_size))

            label = np.random.randint(0, 2)
            bbox = [0, 0, 0, 0]

            if label == 1:
                bbox = self.generate_curve(img)

            num_stars = np.random.randint(0, 3)
            for _ in range(num_stars):
                rx, ry = np.random.randint(0, self.img_size, 2)
                img[rx, ry] = 1.0

            img = np.clip(img, 0, 1)
            X.append(img)
            y_class.append(label)
            y_bbox.append(bbox)

        X = np.array(X).reshape(-1, self.img_size, self.img_size, 1)
        return X, np.array(y_class), np.array(y_bbox)


# ==========================================
# 2. NOVEL ARCHITECTURE: SAR-Net
# ==========================================
def spatial_attention_module(x):
    avg_pool = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x)
    max_pool = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention_map = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    return layers.Multiply()([x, attention_map])


def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    if x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def build_SAR_Net(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Feature Extraction
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Residual + Attention Blocks
    x = residual_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)
    x = spatial_attention_module(x)

    x = residual_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.GlobalAveragePooling2D()(x)

    # Heads
    class_out = layers.Dense(64, activation='relu')(x)
    class_out = layers.Dropout(0.3)(class_out)
    class_output = layers.Dense(1, activation='sigmoid', name='class_output')(class_out)

    reg_out = layers.Dense(64, activation='relu')(x)
    reg_output = layers.Dense(4, activation='linear', name='reg_output')(reg_out)

    model = Model(inputs=inputs, outputs=[class_output, reg_output])
    return model


# ==========================================
# 3. EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    if not os.path.exists('results'):
        os.makedirs('results')

    # 1. Generate Data
    img_size = 64
    generator = AdvancedMissileGenerator(img_size)
    X, y_cls, y_box = generator.create_dataset(num_samples=3000)

    X_train, X_test, yc_train, yc_test, yb_train, yb_test = train_test_split(
        X, y_cls, y_box, test_size=0.2, random_state=42
    )

    # 2. Build & Compile
    print("Building SAR-Net Model...")
    model = build_SAR_Net((img_size, img_size, 1))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'class_output': 'binary_crossentropy', 'reg_output': 'mse'},
        loss_weights={'class_output': 1.0, 'reg_output': 2.0},
        metrics={'class_output': 'accuracy', 'reg_output': 'mse'}
    )

    # 3. Train
    print("\nTraining Started (20 Epochs)...")
    history = model.fit(
        X_train, {'class_output': yc_train, 'reg_output': yb_train},
        validation_data=(X_test, {'class_output': yc_test, 'reg_output': yb_test}),
        epochs=20,
        batch_size=32,
        verbose=1
    )

    # 4. Save Model (NEW ADDITION)
    model.save('missile_model.keras')
    print("Model saved as 'missile_model.keras'")

    # 5. Save Graphs
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['class_output_accuracy'], label='Train Acc')
    plt.plot(history.history['val_class_output_accuracy'], label='Val Acc')
    plt.title('Detection Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Total Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('results/training_metrics.png')

    # 6. Visualize Predictions
    preds = model.predict(X_test[:20])
    pred_class = (preds[0] > 0.5).astype(int)
    pred_box = preds[1]

    plt.figure(figsize=(20, 4))
    count = 0
    for i in range(20):
        if count >= 5: break
        if yc_test[i] == 1:
            img = X_test[i].reshape(img_size, img_size)
            img_rgb = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_GRAY2RGB)

            t = (yb_test[i] * img_size).astype(int)
            cv2.line(img_rgb, (t[0], t[1]), (t[2], t[3]), (0, 255, 0), 2)

            if pred_class[i] == 1:
                p = (pred_box[i] * img_size).astype(int)
                cv2.line(img_rgb, (p[0], p[1]), (p[2], p[3]), (255, 0, 0), 2)

            count += 1
            plt.subplot(1, 5, count)
            plt.imshow(img_rgb)
            plt.title(f"Green=True\nRed=Pred")
            plt.axis('off')

    plt.savefig('results/final_detection_output.png')
    plt.show()
    print("All tasks completed successfully.")