import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import keras

st.title("Land Classification Project (EuroSAT)")

# 1. تحميل الموديل
@st.cache_resource
def load_my_model():
    return keras.models.load_model('EuroSAT_MobileNetV2_Final.keras')

model = load_my_model()

# 2. أسماء الكلاسات (لازم نفس ترتيب التدريب)
class_names = [
    'AnnualCrop (أرض زراعية)',
    'Forest (غابة)',
    'Herbaceous Vegetation (نباتات عشبية)',
    'Highway (طريق سريع)',
    'Industrial (منطقة صناعية)',
    'Pasture (مرعى)',
    'Permanent Crop (محاصيل دائمة)',
    'Residential (منطقة سكنية)',
    'River (نهر)',
    'SeaLake (بحر/بحيرة)'
]

# 3. رفع الصورة
uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    # عرض الصورة
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success("Image Uploaded Successfully")

    # 4. preprocessing الصحيح
    img_resized = image.resize((128, 128))

    img_array = np.array(img_resized)

    # مهم جدًا لـ MobileNetV2
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    # 5. prediction
    with st.spinner("Classifying image..."):
        predictions = model.predict(img_array)

        # لو الموديل فيه softmax آخر layer → ده الصحيح
        score = predictions[0]

        class_index = np.argmax(score)
        result = class_names[class_index]
        confidence = float(np.max(score)) * 100

        st.write("---")
        st.subheader(f"التصنيف المتوقع: **{result}**")
        st.metric("نسبة التأكد", f"{confidence:.2f}%")
