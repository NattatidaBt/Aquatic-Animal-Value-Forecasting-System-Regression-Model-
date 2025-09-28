import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


# --- 1. การโหลดทรัพยากร ---
# โหลดโมเดลที่ดีที่สุด (สมมติว่าเป็น Neural Network)
@st.cache_resource
def load_model(path):
    return joblib.load(path)


# โหลดข้อมูลที่ใช้ในการเทรนเพื่อนำมาหาค่า Mean/StdDev สำหรับ Scaler
# ต้องใช้ไฟล์ที่ไม่มีคอลัมน์ 'มูลค่า(พันบาท)' และ 'ปี' (ตามโค้ดที่คุณลบออกไปก่อน Scale)
@st.cache_data
def load_data_and_fit_scaler(file_path):
    df = pd.read_csv(file_path)
    # เตรียมข้อมูลเพื่อหา Scaler (ต้องมี Features เหมือน X_train)
    # ลบคอลัมน์ 'มูลค่า(พันบาท)'
    X = df.drop(columns=["มูลค่า(พันบาท)"])

    # สร้างและ Fit StandardScaler (ใช้กับ Features ทั้งหมด)
    scaler = StandardScaler()
    scaler.fit(X)

    # เก็บชื่อคอลัมน์และข้อมูลที่จำเป็นต่อการแปลง One-Hot Encoding
    # เนื่องจาก One-Hot ถูกทำมาก่อนหน้าแล้ว เราจึงจำเป็นต้องรู้ลำดับคอลัมน์เดิม
    original_cols = X.columns

    return scaler, original_cols


# ชื่อไฟล์ที่คุณบันทึก
MODEL_PATH = "NeuralNetwork_tuned.pkl"  # หรือ "xgb_tuned_model.pkl"
SCALED_PATH = "scaler_for_prediction.pkl"

# โหลดโมเดลและ scaler
try:
    model = load_model(MODEL_PATH)
    scaler, original_cols = load_data_and_fit_scaler(SCALED_DATA_PATH)

    # ดึงค่า unique ของ column categorical เดิมที่ถูกแปลงเป็น One-Hot แล้ว
    # เช่น 'เครื่องมือ', 'ขนาดเรือ', 'พื้นที่ทำการประมง', 'ชนิดสัตว์น้ำ'
    # ในการ deploy จริง, คุณควรจะ hardcode ค่า unique เหล่านี้จากตอนทำ One-Hot
    # หรือใช้วิธีการที่ซับซ้อนกว่านี้ (เช่น Pipeline)

except FileNotFoundError:
    st.error(
        "ไฟล์โมเดลหรือข้อมูลไม่พบ กรุณาตรวจสอบว่ามีไฟล์ 'afterScaler1.csv' และ 'NeuralNetwork_tuned.pkl' (หรือโมเดลอื่น ๆ) อยู่ในไดเรกทอรีเดียวกัน")
    st.stop()

# --- 2. การรับ Input จากผู้ใช้ ---
st.title("ระบบทำนายมูลค่าสัตว์น้ำจากการประมง")
st.markdown("กรุณาป้อนข้อมูลเพื่อทำนายมูลค่า (พันบาท):")

# **หมายเหตุสำคัญ**: การรับ Input สำหรับ 108 Features ต้องทำอย่างรัดกุม 
# เนื่องจากคุณมีการแปลง Label/One-Hot/Scaling

# --- ตัวอย่างการรับ Input สำหรับ Columns สำคัญ ---
# 1. ปริมาณ(ตัน)
st.subheader("1. ปริมาณ (ตัน)")
ปริมาณ_ตัน = st.number_input("ป้อนปริมาณ (ตัน)", min_value=0.01, value=10.0, step=1.0)

# 2. เดือน (Label Encoded)
st.subheader("2. เดือน")
month_mapping = {
    'มกราคม': 1, 'กุมภาพันธ์': 2, 'มีนาคม': 3, 'เมษายน': 4,
    'พฤษภาคม': 5, 'มิถุนายน': 6, 'กรกฎาคม': 7, 'สิงหาคม': 8,
    'กันยายน': 9, 'ตุลาคม': 10, 'พฤศจิกายน': 11, 'ธันวาคม': 12
}
month_names = list(month_mapping.keys())
selected_month_name = st.selectbox("เลือกเดือน", month_names)
เดือน_encoded = month_mapping[selected_month_name]

# **--- 3. ตัวแปรประเภท Categorical เดิมที่ถูก One-Hot Encoding ---**
# เนื่องจากมี Features จำนวนมาก (108) จะแสดงเฉพาะ Features สำคัญตาม Feature Importance
st.subheader("3. ข้อมูลประเภทการประมง, เครื่องมือ, ขนาดเรือ, และชนิดสัตว์น้ำ")

# ดึงค่า Unique ของคอลัมน์ Categorical เดิม (ในโค้ด Notebook มี แต่ในนี้ต้อง Hardcode)
# ตัวอย่าง:
ประเภทการทำการประมง_options = ['พาณิชย์', 'พื้นบ้าน']
ชนิดสัตว์น้ำ_top_options = ['ปลาน้ำดอกไม้', 'ปลาจวด', 'ปลาทรายแดง', 'ปลาปากคม', 'ปูม้า', 'หมึกกล้วย',
                            'ปลาเป็ด']  # เลือกมาบางส่วนตาม Importance
เครื่องมือ_top_options = ['เครื่องมือ_อวนจมปู', 'เครื่องมือ_อวนจมกุ้ง',
                          'เครื่องมือ_อวนครอบหมึก']  # เลือกมาบางส่วนตาม Importance
ขนาดเรือ_options = ['น้อยกว่า 30 ตันกรอส', 'ตั้งแต่ 30 ตันกรอส ถึงน้อยกว่า 60 ตันกรอส',
                    'ตั้งแต่ 60 ตันกรอส ถึงน้อยกว่า 150 ตันกรอส', 'ตั้งแต่ 150 ตันกรอสขึ้นไป', 'น้อยกว่า 10 ตันกรอส']

# รับ Input
selected_ประเภท = st.selectbox("เลือกประเภทการทำการประมง", ประเภทการทำการประมง_options)
selected_ชนิด = st.selectbox("เลือกชนิดสัตว์น้ำ (ที่สำคัญ)", ชนิดสัตว์น้ำ_top_options)
selected_เครื่องมือ = st.selectbox("เลือกเครื่องมือ (ที่สำคัญ)", เครื่องมือ_top_options)
selected_ขนาดเรือ = st.selectbox("เลือกขนาดเรือ", ขนาดเรือ_options)
selected_พื้นที่ = st.selectbox("เลือกพื้นที่ทำการประมง", ['อันดามัน', 'อ่าวไทย', 'นอกน่านน้ำ'])

# ปุ่มทำนาย
if st.button("ทำนายมูลค่า"):

    # --- 3. การทำนายผล: เตรียม Input Dataframe ให้ตรงกับ X_train ---

    # 1. สร้าง Row ที่เป็นศูนย์ทั้งหมดตามจำนวน Features (108)
    input_data = pd.DataFrame(0, index=[0], columns=original_cols)

    # 2. ใส่ค่าตัวเลขและ Label Encoded
    input_data['ปริมาณ(ตัน)'] = ปริมาณ_ตัน
    # เดือน_encoded: 1=ม.ค., 2=ก.พ.,... 12=ธ.ค. แต่ LabelEncoder ในโน้ตบุ๊กของคุณ
    # ได้แปลง 'มกราคม' (1) เป็น 0, 'กุมภาพันธ์' (2) เป็น 1, ... 'ธันวาคม' (12) เป็น 11
    # ดังนั้นเราต้องทำตามนั้น:
    le_month_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11}
    input_data['เดือน_encoded'] = le_month_map[เดือน_encoded]

    # 3. ใส่ค่า Label Encoded สำหรับ 'ประเภทการทำการประมง' (พาณิชย์=0, พื้นบ้าน=1)
    ประเภทการทำการประมง_label = 1 if selected_ประเภท == 'พื้นบ้าน' else 0
    input_data['ประเภทการทำการประมง_label'] = ประเภทการทำการประมง_label

    # 4. ใส่ค่า One-Hot Encoded (ตั้งค่าคอลัมน์ที่ถูกเลือกเป็น 1)

    # เครื่องมือ
    # ต้องจับคู่ Input กับชื่อคอลัมน์ One-Hot เช่น 'เครื่องมือ_อวนจมปู'
    if selected_เครื่องมือ:
        col_name = f'เครื่องมือ_{selected_เครื่องมือ.split("_")[1]}'  # เช่น 'อวนจมปู'
        if col_name in original_cols:
            input_data[col_name] = 1

    # ขนาดเรือ
    col_name = f'ขนาดเรือ_{selected_ขนาดเรือ}'
    if col_name in original_cols:
        input_data[col_name] = 1

    # พื้นที่ทำการประมง
    col_name = f'พื้นที่ทำการประมง_{selected_พื้นที่}'
    if col_name in original_cols:
        input_data[col_name] = 1

    # ชนิดสัตว์น้ำ
    col_name = f'ชนิดสัตว์น้ำ_{selected_ชนิด}'
    if col_name in original_cols:
        input_data[col_name] = 1

    # 5. ทำ Scaling
    scaled_input = scaler.transform(input_data)
    scaled_input_df = pd.DataFrame(scaled_input, columns=original_cols)

    # 6. ทำนาย
    prediction = model.predict(scaled_input_df)[0]

    # 7. แสดงผล
    st.success(f"มูลค่าที่ทำนายได้: **{prediction:,.2f} พันบาท**")

