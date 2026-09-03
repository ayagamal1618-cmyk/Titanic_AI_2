import streamlit as st
import pandas as pd
import joblib

# ضبط إعدادات الصفحة
st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Prediction")
st.write("أدخل البيانات التالية لمعرفة توقع النجاة باستخدام موديل الـ Decision Tree:")

# تحميل الموديل المتدرب (غيري اسم الملف لو مختلف عندك)
@st.cache_resource
def load_model():
    return joblib.load("desicion_tree_titanic.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("لم يتم العثور على ملف الموديل `model.pkl`. تأكدي من رفعه في نفس المجلد.")
    st.stop()

# عمل نموذج إدخال البيانات
st.subheader("📋 البيانات المطلوبة")

pclass = st.selectbox("درجة التذكرة (Pclass):", [1, 2, 3], index=2)
sex = st.selectbox("النوع (Sex):", ["male", "female"])
age = st.slider("العمر (Age):", min_value=1, max_value=80, value=25)
sibsp = st.number_input("عدد الأقارب / الأزواج المرافقتين (SibSp):", min_value=0, max_value=10, value=0)
parch = st.number_input("عدد الوالدين / الأبناء المرافقتين (Parch):", min_value=0, max_value=10, value=0)
fare = st.number_input("سعر التذكرة (Fare):", min_value=0.0, max_value=600.0, value=32.2)
embarked = st.selectbox("ميناء الانطلاق (Embarked):", ["S", "C", "Q"])

# تجهيز البيانات للتوقع
# تنبيه: الموديل يتوقع تحويل البيانات النصية إلى أرقام إذا لم يتم ترميزها مسبقاً
sex_encoded = 1 if sex == "male" else 0
embarked_dict = {"S": 0, "C": 1, "Q": 2}
embarked_encoded = embarked_dict[embarked]

input_data = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex_encoded,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked_encoded
}])

# زر التوقع
st.markdown("---")
if st.button("🔮 احسب التوقع"):
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success("🎉 التوقع: الشخص ده غالباً **نجا (Survived)**!")
    else:
        st.error("⚠️ التوقع: الشخص ده غالباً **لم ينجُ (Not Survived)**.")