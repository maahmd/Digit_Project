if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype(np.uint8)
    original_img = Image.fromarray(img)               # original drawing
    img = Image.fromarray(img).convert('L')           # grayscale for model
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Check if canvas is nearly empty
    if img_array.mean() < 0.01:
        st.warning("✏️ ارسم رقماً أولاً")
    else:
        start_time = time.time()
        prediction = model.predict(img_array)
        end_time = time.time()
        prediction_time = (end_time - start_time) * 1000

        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Display prediction and confidence in Arabic
        st.markdown(f"## 🎯 هذا الرقم هو: **{predicted_digit}**")
        st.markdown(f"### 🔍 درجة الثقة: **{confidence:.1f}%**")

        # Show confidence level with a colour/icon
        if confidence >= 90:
            st.success("✅ النموذج واثق جداً من هذه التوقعات")
        elif confidence >= 65:
            st.info("ℹ️ النموذج واثق بشكل معقول من هذه التوقعات")
        else:
            st.warning("⚠️ النموذج غير واثق كثيراً. حاول رسم الرقم بشكل أوضح")

        # Optional: show prediction time
        st.caption(f"⚡ زمن التنبؤ: {int(prediction_time)} مللي ثانية")

        # Show original and processed images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="الرسم الأصلي", width=150)
        with col2:
            st.image(img, caption="الصورة المعالجة (28×28)", width=150)