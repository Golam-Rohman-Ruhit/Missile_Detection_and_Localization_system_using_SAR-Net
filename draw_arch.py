import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.utils import plot_model
import os


# ==========================================
# 1. DEFINE THE EXACT MODEL (No Loading Error)
# ==========================================
def spatial_attention_module(x):
    # Same as your main code
    avg_pool = layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True), name="Avg_Pool")(x)
    max_pool = layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True), name="Max_Pool")(x)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention_map = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', name="Attn_Map")(concat)
    return layers.Multiply(name="Apply_Attn")([x, attention_map])


def residual_block(x, filters, block_id):
    prefix = f"ResBlock_{block_id}_"
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), padding='same', name=prefix + "Conv1")(x)
    x = layers.BatchNormalization(name=prefix + "BN1")(x)
    x = layers.Activation('relu', name=prefix + "ReLU1")(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', name=prefix + "Conv2")(x)
    x = layers.BatchNormalization(name=prefix + "BN2")(x)

    if x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same', name=prefix + "Shortcut")(shortcut)

    x = layers.Add(name=prefix + "Add")([x, shortcut])
    x = layers.Activation('relu', name=prefix + "ReLU_Out")(x)
    return x


def build_SAR_Net(input_shape):
    inputs = layers.Input(shape=input_shape, name="Input_Image")

    # Feature Extraction
    x = layers.Conv2D(32, (3, 3), padding='same', name="Init_Conv")(inputs)
    x = layers.BatchNormalization(name="Init_BN")(x)
    x = layers.Activation('relu', name="Init_ReLU")(x)

    # Residual + Attention Blocks
    x = residual_block(x, 64, block_id=1)
    x = layers.MaxPooling2D((2, 2), name="Pool_1")(x)
    x = spatial_attention_module(x)

    x = residual_block(x, 128, block_id=2)
    x = layers.MaxPooling2D((2, 2), name="Pool_2")(x)

    x = layers.GlobalAveragePooling2D(name="Global_Avg_Pool")(x)

    # Heads
    class_out = layers.Dense(64, activation='relu', name="Class_Dense")(x)
    class_out = layers.Dropout(0.3, name="Dropout")(class_out)
    class_output = layers.Dense(1, activation='sigmoid', name="class_output")(class_out)

    reg_out = layers.Dense(64, activation='relu', name="Reg_Dense")(x)
    reg_output = layers.Dense(4, activation='linear', name="reg_output")(reg_out)

    model = Model(inputs=inputs, outputs=[class_output, reg_output], name="SAR-Net")
    return model


# ==========================================
# 2. GENERATE STANDARD KERAS PLOT
# ==========================================
print("Building Model...")
model = build_SAR_Net((64, 64, 1))

print("Generating Standard Architecture Diagram...")
try:
    # এটি অফিশিয়াল ডায়াগ্রাম তৈরি করবে
    plot_model(
        model,
        to_file='official_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
        dpi=96
    )
    print("SUCCESS: Saved as 'official_architecture.png'")
except Exception as e:
    print("\nERROR: Could not create image using Graphviz.")
    print(f"Reason: {e}")
    print("\nSOLUTION: If Graphviz fails, please use the website: https://netron.app")