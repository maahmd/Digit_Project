import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import time

# ===== 1. تعريف بنية النموذج =====
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

# ===== 3. إنشاء لوحة الرسم (هذا الجزء كان ناقصاً عندك) =====
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

# ===== 4. معالجة الرسم والتنبؤ =====
if canvas_result.image_data is not None:
    # تحويل بيانات الرسم إلى صورة
    img_array_raw = canvas_result.image_data.astype(np.uint8)
    original_img = Image.fromarray(img_array_raw)  # الصورة الأصلية (ملونة)
    
    # تحويل إلى أبيض وأسود وتغيير الحجم
    img_gray = Image.fromarray(img_array_raw).convert('L')
    img_resized = img_gray.resize((28, 28))
    
    # تحويل إلى مصفوفة أرقام وتطبيع
    img_input = np.array(img_resized) / 255.0
    img_input = img_input.reshape(1, 28, 28, 1)

    # التحقق من أن اللوحة ليست فارغة
    if img_input.mean() < 0.01:
        st.warning("✏️ ارسم رقماً أولاً")
    else:
        # قياس زمن التنبؤ
        start_time = time.time()
        prediction = model.predict(img_input)
        end_time = time.time()
        prediction_time_ms = (end_time - start_time) * 1000

        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # عرض النتيجة الرئيسية
        st.markdown(f"## 🎯 هذا الرقم هو: **{predicted_digit}**")
        st.markdown(f"### 🔍 درجة الثقة: **{confidence:.1f}%**")
        st.caption(f"⚡ زمن التنبؤ: {int(prediction_time_ms)} مللي ثانية")

        # عرض رسالة حسب مستوى الثقة
        if confidence >= 90:
            st.success("✅ النموذج واثق جداً من هذا التخمين")
        elif confidence >= 65:
            st.info("ℹ️ النموذج واثق بشكل معقول من هذا التخمين")
        else:
            st.warning("⚠️ النموذج غير واثق كثيراً. حاول رسم الرقم بشكل أوضح")

        # عرض الصور جنباً إلى جنب
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="🖍️ الرسم الأصلي", width=150)
        with col2:
            st.image(img_resized, caption="🔍 الصورة المعالجة (28×28)", width=150)
