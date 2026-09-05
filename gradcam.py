import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def gradcam(img_path, model, last_conv_layer):

    img = Image.open(img_path).convert("RGB")
    img = img.resize((299,299))

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    base_model = model.layers[0]   # Xception

    # build gradient model
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(last_conv_layer).output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, x = grad_model(img_array)

        # pass through top classifier layers
        for layer in model.layers[1:]:
            x = layer(x)

        predictions = x
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # -------------------------------
    # Tumor Location Description
    # -------------------------------

    h, w = heatmap.shape
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    # Left / Right
    if x < w/2:
        side = "Left"
    else:
        side = "Right"

    # Brain region
    if y < h/3:
        region = "Frontal Lobe"
    elif y < 2*h/3:
        region = "Parietal/Temporal Region"
    else:
        region = "Occipital Lobe"

    tumor_location = f"{side} {region}"

    return heatmap, tumor_location

def show_gradcam(img, heatmap):
    img = cv2.resize(img, (299,299))
   # img = cv2.imread(img_path)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return superimposed_img

