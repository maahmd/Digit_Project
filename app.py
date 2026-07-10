import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ===== 1. تعريف بنية النموذج مع أسماء فريدة =====
def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv2d_1'),
        tf.keras.layers.MaxPooling2D((2, 2), name='maxpool_1'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
        tf.keras.layers.MaxPooling2D((2, 2), name='maxpool_2'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_3'),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(64, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(10, activation='softmax', name='dense_2')
    ])
    return model

model = build_model()
model.load_weights('model_weights.weights.h5')

# ===== 2. واجهة التطبيق =====
st.title("🤖 التعرف على الأرقام المكتوبة بخط اليد")
st.write("ارسم رقماً في المربع الأسود بالأسفل، وسأخمنه فوراً!")

# إنشاء لوحة الرسم
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype(np.uint8)
    img = Image.fromarray(img).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    
    st.markdown(f"## 🎯 هذا الرقم هو: **{predicted_digit}**")
    st.image(img, caption="الصورة التي رآها النموذج", width=150)

#إظهار نسبة ثقة النموذج
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)
confidence = np.max(prediction) * 100

st.markdown(f"## 🎯 هذا الرقم هو: **{predicted_digit}**")
st.markdown(f"### 🔍 درجة الثقة: **{confidence:.1f}%**")

if confidence < 70:
    st.warning("⚠️ لست متأكداً كثيراً... حاول رسم الرقم بشكل أوضح!")

st.image(img, caption="الصورة التي رآها النموذج", width=150)