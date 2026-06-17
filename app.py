import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Land Classification Project (EuroSAT)")

# 1. بناء الهيكل يدوياً وشحن الأوزان لتخطي مشكلة الـ BatchNormalization تماماً
@st.cache_resource
def load_my_model():
    # بناء الـ Base Model القياسي بدون الأوزان الجاهزة
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights=None
    )
    base_model.trainable = False
    
    # بناء الـ Sequential بنفس الترتيب المكتوب جوه ملفك بالظبط
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2), # مأخوذة من الـ config المذكور في الـ logs
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # شحن الأوزان مباشرة من ملفك الحالي بدون فتح كاجل
    model.load_weights('EuroSAT_MobileNetV2_Final.keras')
    return model

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

        score = predictions[0]

        class_index = np.argmax(score)
        result = class_names[class_index]
        confidence = float(np.max(score)) * 100

        st.write("---")
        st.subheader(f"التصنيف المتوقع: **{result}**")
        st.metric("نسبة التأكد", f"{confidence:.2f}%")
