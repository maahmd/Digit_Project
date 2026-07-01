import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ===== 1. تحميل النموذج =====
model = tf.keras.models.load_model('digit_recognition_model.h5')

# ===== 2. واجهة التطبيق =====
st.title("🤖 التعرف على الأرقام المكتوبة بخط اليد")
st.write("ارسم رقماً في المربع الأسود بالأسفل، وسأخمنه فوراً!")

# ===== 3. لوحة الرسم =====
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

# ===== 4. التحقق من الرسم =====
if canvas_result.image_data is not None:

    # تحويل الصورة إلى numpy
    img = canvas_result.image_data.astype(np.uint8)

    # تحويل إلى PIL ثم إعادة تجهيزها
    img_pil = Image.fromarray(img).convert('L')
    img_pil = img_pil.resize((28, 28))

    # تحويل إلى array للنموذج
    img_array = np.array(img_pil) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # ===== 5. التحقق من أن الرسم غير فارغ =====
    if img_array.mean() < 0.01:
        st.warning("✏️ ارسم رقماً أولاً")

    else:
        # ===== 6. التنبؤ =====
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # ===== 7. مستوى الثقة =====
        if confidence >= 90:
            st.success("The model is highly confident in this prediction.")
        elif confidence >= 65:
            st.info("The model is reasonably confident in this prediction.")
        else:
            st.warning("The model is not very confident. Try drawing more clearly.")

        # ===== 8. عرض النتيجة =====
        st.markdown(
            f"""
## 🎯 هذا الرقم هو: **{predicted_digit}**

📊 نسبة الثقة:
**{confidence:.2f}%**
"""
        )

        st.image(img_pil, caption="الصورة التي رآها النموذج", width=150)
