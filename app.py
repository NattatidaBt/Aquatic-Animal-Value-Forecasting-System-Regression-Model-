import streamlit as st
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# ==============================================================================
# --- 0. Initial Configuration & Global Constants ---
# ==============================================================================

# กำหนดสถานะเริ่มต้นของ Session State ก่อนการเรียกใช้ฟังก์ชันใดๆ
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'th'

# URL รูปภาพพื้นหลังสำหรับกรอบ Input (ปลา/อวน)
BACKGROUND_IMAGE_URL = "https://ipdefenseforum.com/wp-content/uploads/2020/01/Fisheries_Lead-in-1024x653.jpg"
# URL รูปภาพพื้นหลังสำหรับพื้นหลังแอปทั้งหมด (ปลาทูน่า)
RESULT_IMAGE_URL = "https://scontent.furt3-1.fna.fbcdn.net/v/t1.6435-9/156313088_109358747878601_3235269419876796112_n.jpg?_nc_cat=109&ccb=1-7&_nc_sid=cc71e4&_nc_eui2=AeGkHS3XfHjCWeVppW6lkvokjTaDpmeYfgeNNoOmZ5h-BxAd49TIkJotoKALibPVQm9-hm4NB96wPjyZYZ8b3tlx&_nc_ohc=Qwr9GSnrDscQ7kNvwGLrY&_nc_oc=Adk0d8BtyjEr-UvZRBB38MkHEiklMVBKBzHKPu-o-ZWtiGV5vt2cDwQXSzUKECWc0n2j60sA9vsV8_QYtW9Z95Nw&_nc_zt=23&_nc_ht=scontent.furt3-1.fna&_nc_gid=PCLXdL9iFkiWkyr83uixnQ&oh=00_Afb8sP2lKVpAPGBC6ekeLyw-LEpOK1BqcztrjTiLRLXhHQ&oe=68FE3BEF"

# --- 0.1 Multilingual Dictionary & Translator Function ---
LANGUAGE_DICT = {
    "th": {
        "title": "ระบบคาดการณ์มูลค่าสัตว์น้ำ (Fisheries Price Predictor)",
        "input_header": "ข้อมูลการป้อนเข้า (Inputs)",
        "predict_button": "คาดการณ์มูลค่า",
        "output_header": "ผลการคาดการณ์",
        "output_label_thb": "มูลค่าคาดการณ์ (หน่วย: พันบาท)",
        "total_baht": "มูลค่ารวมทั้งหมดโดยประมาณ",
        "avg_actual": "มูลค่ารวมเฉลี่ยจริงใน Dataset",
        "error_model": "ไม่พบไฟล์ Model, Scaler หรือ Dataset ที่จำเป็น",
        "error_input": "เกิดข้อผิดพลาด: ข้อมูลที่เลือกไม่ถูกต้องหรือขาดหายไป",
        "error_prediction": "เกิดข้อผิดพลาดในการทำนาย",
        "input_fields": {
            "ปริมาณ(ตัน)": "1. ปริมาณ (ตัน)",
            "เดือน": "2. เดือน",
            "ประเภทการทำการประมง": "3. ประเภทการทำการประมง",
            "พื้นที่ทำการประมง": "4. พื้นที่ทำการประมง",
            "ขนาดเรือ": "5. ขนาดเรือ",
            "เครื่องมือ": "6. เครื่องมือ",
            "ชนิดสัตว์น้ำ": "7. ชนิดสัตว์น้ำ"
        }
    },
    "en": {
        "title": "Fisheries Price Prediction System (Regression Model)",
        "input_header": "Input Parameters",
        "predict_button": "Predict Value",
        "output_header": "Prediction Results",
        "output_label_thb": "Predicted Value (Unit: Thousand Baht)",
        "total_baht": "Estimated Total Value (Baht)",
        "avg_actual": "Average Actual Total Value in Dataset",
        "error_model": "Required Model, Scaler, or Dataset files not found",
        "error_input": "Error: Selected data is invalid or missing",
        "error_prediction": "An error occurred during prediction",
        "input_fields": {
            "ปริมาณ(ตัน)": "1. Volume (Tons)",
            "เดือน": "2. Month",
            "ประเภทการทำการประมง": "3. Fishery Type",
            "พื้นที่ทำการประมง": "4. Fishing Area",
            "ขนาดเรือ": "5. Boat Size",
            "เครื่องมือ": "6. Fishing Tool",
            "ชนิดสัตว์น้ำ": "7. Species Type"
        }
    }
}


def _T(key):
    return LANGUAGE_DICT[st.session_state['lang']][key]


# --- 2. กำหนดรายการคอลัมน์และ Maps ที่สมบูรณ์ (108 Features) ---

MONTH_OPTIONS = ['มกราคม', 'กุมภาพันธ์', 'มีนาคม', 'เมษายน', 'พฤษภาคม', 'มิถุนายน',
                 'กรกฎาคม', 'สิงหาคม', 'กันยายน', 'ตุลาคม', 'พฤศจิกายน', 'ธันวาคม']
EN_MONTH_OPTIONS = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
]

MONTH_MAP = {name: i for i, name in enumerate(MONTH_OPTIONS)}
TYPE_MAP = {'พาณิชย์': 0, 'พื้นบ้าน': 1}
TYPE_OPTIONS = list(TYPE_MAP.keys())
EN_TYPE_OPTIONS = ['Commercial', 'Artisanal']

TOOL_OPTIONS = [
    'อวนลากแผ่นตะเฆ่', 'อวนลากคู่', 'อวนลากคานถ่าง', 'อวนล้อมจับ', 'อวนล้อมจับปลากะตัก',
    'อวนครอบปลากะตัก', 'อวนครอบหมึก', 'อวนช้อนปลาจะละเม็ด', 'อวนช้อน/ยกปลากะตัก',
    'ลอบหมึก', 'ลอบหมึกสาย', 'ลอบปลา', 'ลอบปู', 'คราดหอยลาย', 'คราดหอยอื่น',
    'อวนลอยปลา', 'อวนจมปู', 'อวนติดตาอื่นๆ', 'แผงยกปูจักจั่น', 'เบ็ดมือ', 'เบ็ดราว',
    'อวนรุนเคย', 'อวนจมกุ้ง', 'เครื่องมืออื่นๆ', 'สวิงช้อนแมงกะพรุน',
    'อวนติดตาปลาทู', 'อวนติดตาปลาหลังเขียว'
]
EN_TOOL_OPTIONS = [
    'Otter Trawl', 'Pair Trawl', 'Beam Trawl', 'Purse Seine', 'Anchovy Purse Seine',
    'Anchovy Seine (Cover)', 'Squid Seine (Cover)', 'Threadfin Bream Seine', 'Anchovy Scoop Net',
    'Squid Trap', 'Octopus Trap', 'Fish Trap', 'Crab Trap', 'Mussel Dredge', 'Other Shellfish Dredge',
    'Gillnet (Fish)', 'Crab Gillnet', 'Other Gillnets', 'Jingling Crab Lift Net', 'Handline', 'Longline',
    'Acetes Push Net', 'Shrimp Gillnet', 'Other Gears', 'Jellyfish Dip Net',
    'Mackerel Gillnet', 'Round Scad Gillnet'
]

BOAT_SIZE_OPTIONS = [
    'น้อยกว่า 30 ตันกรอส', 'ตั้งแต่ 30 ตันกรอส ถึงน้อยกว่า 60 ตันกรอส',
    'ตั้งแต่ 60 ตันกรอส ถึงน้อยกว่า 150 ตันกรอส', 'ตั้งแต่ 150 ตันกรอสขึ้นไป',
    'น้อยกว่า 10 ตันกรอส'
]
EN_BOAT_SIZE_OPTIONS = [
    'Less than 30 GT', '30 GT to Less than 60 GT',
    '60 GT to Less than 150 GT', '150 GT and Over',
    'Less than 10 GT'
]

AREA_OPTIONS = ['อันดามัน', 'อ่าวไทย', 'นอกน่านน้ำ']
EN_AREA_OPTIONS = ['Andaman Sea', 'Gulf of Thailand', 'International Waters']

SPECIES_OPTIONS = [
    'ปลาน้ำดอกไม้', 'ปลาจวด', 'ปลาทรายแดง', 'ปลาทรายขาว', 'ปลาปากคม', 'ปลาดาบเงิน',
    'ปลาลิ้นหมา', 'ปลาเลยหน้าดิน', 'ปลาเป็ด', 'กุ้งกุลาลาย', 'กุ้งโอคัก', 'กุ้งอื่นๆ',
    'ปูม้า', 'ปูอื่นๆ', 'หมึกกล้วย', 'หมึกกระดอง', 'หมึกสาย', 'ปลาสีกุน',
    'ปลากะพงแดง', 'ปลาตาโต', 'ปลากดทะเล', 'ปลากระเบน', 'ปลายอดจาก', 'ปลาเก๋า',
    'กุ้งแชบ๊วย', 'กั้งไข่', 'หอยเชลล์', 'หอยอื่นๆ', 'ปลาแข้งไก่', 'ปลาเห็ดโคน',
    'ปลาดุกทะเล', 'ปลาทูแขก', 'ปลาหลังเขียว', 'ปลาเลยผิวน้ำ', 'ปลาสีกุนตาโต',
    'ปลาทู', 'ปลาดาบลาว', 'ปลาสำลี', 'ปลาจะละเม็ดดำ', 'สัตว์น้ำอื่นๆ', 'ปลาลัง',
    'ปลาอินทรี', 'ปลากะตัก', 'ปลากุเรา', 'ปลากระบอก', 'หมึกหอม', 'ปลาโอลาย',
    'ปลากระโทงแทง', 'ปลาจะละเม็ดขาว', 'ปลาฉลาม', 'ปลาจักรผาน', 'กุ้งกุลาดำ',
    'หอยแครง', 'ปลาโอดำ', 'ปลาโอหลอด', 'ปลากระโทงแทงร่ม', 'ปลาทูน่าท้องแถบ',
    'ปลาโอแกลบ', 'ปลาทูน่า', 'หอยลาย', 'กั้งกระดาน', 'ปูจักจั่น', 'ปูทะเล', 'เคย',
    'กุ้งเหลือง', 'ปลากะพงขาว', 'หอยแมลงภู่', 'แมงกะพรุน', 'หอยกะพง', 'หอยนางรม'
]
EN_SPECIES_OPTIONS = [
    'Threadfin Bream', 'Croaker', 'Red Snapper', 'White Snapper', 'Ribbon Fish', 'Hairtail',
    'Sole/Flounder', 'Demersal Fish (Other)', 'Trash Fish', 'Tiger Prawn', 'Mantis Shrimp', 'Other Shrimp',
    'Blue Swimming Crab', 'Other Crabs', 'Squid (Loligo)', 'Cuttlefish', 'Octopus', 'Trevally',
    'Red Grouper', 'Bigeye Scad', 'Sea Catfish', 'Ray/Skate', 'Kingfish', 'Grouper',
    'White Shrimp', 'Mantis Shrimp (Roe)', 'Scallop', 'Other Shellfish', 'King Mackerel', 'Threadfin',
    'Sea Catfish (Other)', 'Indian Mackerel', 'Round Scad', 'Pelagic Fish (Other)', 'Bigeye Tuna',
    'Indian Mackerel (Small)', 'Mackerel', 'Silver Pomfret', 'Black Pomfret', 'Other Aquatic Animals',
    'Skipjack Tuna', 'King Mackerel (Other)', 'Anchovy', 'Threadfin (Other)', 'Mullet', 'Squid (Cuttlefish)',
    'Tuna (Skipjack)', 'Sailfish', 'White Pomfret', 'Shark', 'Jackfish',
    'Black Tiger Prawn', 'Cockle', 'Black Tuna', 'Longtail Tuna', 'Swordfish/Marlin',
    'Skipjack Tuna (Stripe)', 'Other Tuna', 'Tuna (Other)', 'Clam', 'Slipper Lobster',
    'Jingling Crab', 'Mud Crab', 'Krill', 'Yellow Shrimp', 'White Snapper (Other)',
    'Mussel', 'Jellyfish', 'Whelk', 'Oyster'
]

EN_TO_TH_MAPS = {
    "TYPE": dict(zip(EN_TYPE_OPTIONS, TYPE_OPTIONS)),
    "TOOL": dict(zip(EN_TOOL_OPTIONS, TOOL_OPTIONS)),
    "BOAT_SIZE": dict(zip(EN_BOAT_SIZE_OPTIONS, BOAT_SIZE_OPTIONS)),
    "AREA": dict(zip(EN_AREA_OPTIONS, AREA_OPTIONS)),
    "SPECIES": dict(zip(EN_SPECIES_OPTIONS, SPECIES_OPTIONS)),
}

BASE_FEATURES = [
    'ปริมาณ(ตัน)', 'เดือน_encoded', 'ประเภทการทำการประมง_label'
]

OHE_TOOL = [
    'เครื่องมือ_คราดหอยลาย', 'เครื่องมือ_คราดหอยอื่น', 'เครื่องมือ_ลอบปลา', 'เครื่องมือ_ลอบปู',
    'เครื่องมือ_ลอบหมึก', 'เครื่องมือ_ลอบหมึกสาย', 'เครื่องมือ_สวิงช้อนแมงกะพรุน',
    'เครื่องมือ_อวนครอบปลากะตัก', 'เครื่องมือ_อวนครอบหมึก', 'เครื่องมือ_อวนจมกุ้ง',
    'เครื่องมือ_อวนจมปู', 'เครื่องมือ_อวนช้อน/ยกปลากะตัก', 'เครื่องมือ_อวนช้อนปลาจะละเม็ด',
    'เครื่องมือ_อวนติดตาปลาทู', 'เครื่องมือ_อวนติดตาปลาหลังเขียว', 'เครื่องมือ_อวนติดตาอื่นๆ',
    'เครื่องมือ_อวนรุนเคย', 'เครื่องมือ_อวนลอยปลา', 'เครื่องมือ_อวนลากคานถ่าง',
    'เครื่องมือ_อวนลากคู่', 'เครื่องมือ_อวนลากแผ่นตะเฆ่', 'เครื่องมือ_อวนล้อมจับ',
    'เครื่องมือ_อวนล้อมจับปลากะตัก', 'เครื่องมือ_เครื่องมืออื่นๆ', 'เครื่องมือ_เบ็ดมือ',
    'เครื่องมือ_เบ็ดราว', 'เครื่องมือ_แผงยกปูจักจั่น'
]

OHE_BOAT_SIZE = [
    'ขนาดเรือ_ตั้งแต่ 150 ตันกรอสขึ้นไป', 'ขนาดเรือ_ตั้งแต่ 30 ตันกรอส ถึงน้อยกว่า 60 ตันกรอส',
    'ขนาดเรือ_ตั้งแต่ 60 ตันกรอส ถึงน้อยกว่า 150 ตันกรอส', 'ขนาดเรือ_น้อยกว่า 10 ตันกรอส',
    'ขนาดเรือ_น้อยกว่า 30 ตันกรอส'
]

OHE_AREA = [
    'พื้นที่ทำการประมง_นอกน่านน้ำ', 'พื้นที่ทำการประมง_อันดามัน', 'พื้นที่ทำการประมง_อ่าวไทย'
]

OHE_SPECIES = [
    'ชนิดสัตว์น้ำ_กั้งกระดาน', 'ชนิดสัตว์น้ำ_กั้งไข่', 'ชนิดสัตว์น้ำ_กุ้งกุลาดำ',
    'ชนิดสัตว์น้ำ_กุ้งกุลาลาย', 'ชนิดสัตว์น้ำ_กุ้งอื่นๆ', 'ชนิดสัตว์น้ำ_กุ้งเหลือง',
    'ชนิดสัตว์น้ำ_กุ้งแชบ๊วย', 'ชนิดสัตว์น้ำ_กุ้งโอคัก', 'ชนิดสัตว์น้ำ_ปลากดทะเล',
    'ชนิดสัตว์น้ำ_ปลากระบอก', 'ชนิดสัตว์น้ำ_ปลากระเบน', 'ชนิดสัตว์น้ำ_ปลากระโทงแทง',
    'ชนิดสัตว์น้ำ_ปลากะโทงแทร่ม', 'ชนิดสัตว์น้ำ_ปลากะตัก', 'ชนิดสัตว์น้ำ_ปลากะพงขาว',
    'ชนิดสัตว์น้ำ_ปลากะพงแดง', 'ชนิดสัตว์น้ำ_ปลากุเรา', 'ชนิดสัตว์น้ำ_ปลาจวด',
    'ชนิดสัตว์น้ำ_ปลาจะละเม็ดขาว', 'ชนิดสัตว์น้ำ_ปลาจะละเม็ดดำ', 'ชนิดสัตว์น้ำ_ปลาจักรผาน',
    'ชนิดสัตว์น้ำ_ปลาฉลาม', 'ชนิดสัตว์น้ำ_ปลาดาบลาว', 'ชนิดสัตว์น้ำ_ปลาดาบเงิน',
    'ชนิดสัตว์น้ำ_ปลาดุกทะเล', 'ชนิดสัตว์น้ำ_ปลาตาโต', 'ชนิดสัตว์น้ำ_ปลาทรายขาว',
    'ชนิดสัตว์น้ำ_ปลาทรายแดง', 'ชนิดสัตว์น้ำ_ปลาทู', 'ชนิดสัตว์น้ำ_ปลาทูน่า',
    'ชนิดสัตว์น้ำ_ปลาทูน่าท้องแถบ', 'ชนิดสัตว์น้ำ_ปลาทูแขก', 'ชนิดสัตว์น้ำ_ปลาน้ำดอกไม้',
    'ชนิดสัตว์น้ำ_ปลาปากคม', 'ชนิดสัตว์น้ำ_ปลายอดจาก', 'ชนิดสัตว์น้ำ_ปลาลัง',
    'ชนิดสัตว์น้ำ_ปลาลิ้นหมา', 'ชนิดสัตว์น้ำ_ปลาสำลี', 'ชนิดสัตว์น้ำ_ปลาสีกุน',
    'ชนิดสัตว์น้ำ_ปลาสีกุนตาโต', 'ชนิดสัตว์น้ำ_ปลาหลังเขียว', 'ชนิดสัตว์น้ำ_ปลาอินทรี',
    'ชนิดสัตว์น้ำ_ปลาเก๋า', 'ชนิดสัตว์น้ำ_ปลาเป็ด', 'ชนิดสัตว์น้ำ_ปลาเลยผิวน้ำ',
    'ชนิดสัตว์น้ำ_ปลาเลยหน้าดิน', 'ชนิดสัตว์น้ำ_ปลาเห็ดโคน', 'ชนิดสัตว์น้ำ_ปลาแข้งไก่',
    'ชนิดสัตว์น้ำ_ปลาโอดำ', 'ชนิดสัตว์น้ำ_ปลาโอลาย', 'ชนิดสัตว์น้ำ_ปลาโอหลอด',
    'ชนิดสัตว์น้ำ_ปลาโอแกลบ', 'ชนิดสัตว์น้ำ_ปูจักจั่น', 'ชนิดสัตว์น้ำ_ปูทะเล',
    'ชนิดสัตว์น้ำ_ปูม้า', 'ชนิดสัตว์น้ำ_ปูอื่นๆ', 'ชนิดสัตว์น้ำ_สัตว์น้ำอื่นๆ',
    'ชนิดสัตว์น้ำ_หมึกกระดอง', 'ชนิดสัตว์น้ำ_หมึกกล้วย', 'ชนิดสัตว์น้ำ_หมึกสาย',
    'ชนิดสัตว์น้ำ_หมึกหอม', 'ชนิดสัตว์น้ำ_หอยกะพง', 'ชนิดสัตว์น้ำ_หอยนางรม',
    'ชนิดสัตว์น้ำ_หอยลาย', 'ชนิดสัตว์น้ำ_หอยอื่นๆ', 'ชนิดสัตว์น้ำ_หอยเชลล์',
    'ชนิดสัตว์น้ำ_หอยแครง', 'ชนิดสัตว์น้ำ_หอยแมลงภู่', 'ชนิดสัตว์น้ำ_เคย',
    'ชนิดสัตว์น้ำ_แมงกะพรุน'
]

FEATURE_COLUMNS = BASE_FEATURES + OHE_TOOL + OHE_BOAT_SIZE + OHE_AREA + OHE_SPECIES

# --- 0. Page Configuration ---
st.set_page_config(
    page_title="ระบบคาดการณ์มูลค่าสัตว์น้ำ | Fisheries Price Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- 0.2 CSS Injection ---
st.markdown(
    f"""
    <style>
    /* 1. สไตล์พื้นหลังหลัก */
    .stApp {{
        background-image: linear-gradient(rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.4)), 
                          url("{RESULT_IMAGE_URL}"); 
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
        min-height: 100vh;
    }}

    /* 2. สไตล์ Title */
    h1 {{
        color: #004d40; 
        text-align: center;
        text-shadow: 1px 1px 2px #b2dfdb;
        background-color: rgba(255, 255, 255, 0.95); 
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
    }}

    /* 3. ปรับปรุงพื้นหลังของกล่อง Input Form */
    div[data-testid="stForm"] {{
        background-image: linear-gradient(rgba(255, 255, 255, 0.7), rgba(255, 255, 255, 0.7)),
                          url("{BACKGROUND_IMAGE_URL}"); 
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;

        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3); 
    }}

    /* 4. ปรับสีตัวอักษรและหัวข้อภายใน Form */
    div[data-testid="stForm"] label {{
        background-color: rgba(255, 255, 255, 0.7); 
        padding: 5px 10px;
        border-radius: 10px;
        display: block;
        margin-bottom: 5px; 
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }}

    div[data-testid="stForm"] label p {{
        color: #004d40 !important; 
        text-shadow: none;
        font-weight: bold;
    }}

    div[data-testid="stForm"] h3 {{
        color: #004d40 !important; 
        text-shadow: none;
    }}

    /* 5. ปรับสไตล์ Input Widgets */
    div[data-testid^="stTextInput"] input, 
    div[data-testid^="stSelectbox"] div[data-baseweb="select"] {{
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #004d40 !important; 
        border-radius: 10px;
    }}

    /* 5c: สำหรับปุ่มคาดการณ์ */
    div[data-testid="stFormSubmitButton"] button {{
        border-radius: 10px;
        background-color: #008080;
        color: white;
    }}


    /* 7. สไตล์ของกล่องผลลัพธ์หลัก */
    .prediction-container {{
        background-image: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)),
                          url("{RESULT_IMAGE_URL}"); 
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        border-radius: 10px;
        border: 1px solid #ddd; 
        padding: 15px;
        margin-top: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }}

    /* 8. ซ่อนพื้นที่ว่างเปล่าเหนือ Metric */
    div[data-testid="stMetric"] {{
        background-color: transparent !important; 
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    /* 9. สไตล์กล่องผลรวม (Predicted Total Value) */
    .total-value-green-box {{
        background-color: #f0fff0; 
        border-radius: 10px;
        border-left: 8px solid #004d40; 
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        margin-top: 15px;
    }}
    .total-value-green-box .total-baht-text,
    .total-value-green-box .total-baht-value {{
        color: #004d40 !important; 
    }}
    .total-value-green-box .total-baht-value {{
        color: #008080 !important; 
        font-size: 32px;
        font-weight: bold;
    }}

    /* สไตล์สำหรับกล่องมูลค่าเฉลี่ยจริง */
    .actual-avg-box {{
        background-color: #fff8e1; 
        border-radius: 10px;
        border-left: 8px solid #ffb300; 
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        margin-top: 15px;
    }}
    .actual-avg-box .actual-avg-text {{
        color: #ff8f00 !important; 
        font-size: 16px;
        margin: 0;
    }}
    .actual-avg-box .actual-avg-value {{
        color: #ff6f00 !important; 
        font-size: 28px;
        font-weight: bold;
        margin: 0;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# --- 1. โหลดไฟล์ Model, Scaler และ Dataset ---

MODEL_PATH = 'NeuralNetwork_tuned.pkl'
SCALER_PATH = 'scaler.pkl'
# เปลี่ยนไปใช้ไฟล์ต้นฉบับ
COMPARISON_DATA_PATH = r'C:\Users\Win11\OneDrive - Walailak University\Documents\JupyterProject\machine_ประมง.csv'

try:
    # โหลดโมเดล (MLPRegressor/NN) และ Scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    # โหลดข้อมูลสำหรับเปรียบเทียบ (ไฟล์ต้นฉบับ)
    df_actual = pd.read_csv(COMPARISON_DATA_PATH)

except FileNotFoundError:
    st.error(
        _T("error_model") +
        "\n\n**ข้อแนะนำ:** โปรดตรวจสอบว่าไฟล์ `NeuralNetwork_tuned.pkl`, `scaler.pkl`, และ **`machine_ประมง.csv`** อยู่ในโฟลเดอร์เดียวกันกับ `app.py`")
    st.stop()
except Exception as e:
    st.error(f"{_T('error_model')}: {e}")
    st.stop()


# --- 3. ฟังก์ชัน Preprocessing (จำลองขั้นตอนใน Notebook) ---
def preprocess_input(raw_data, scaler, feature_columns):
    """แปลงข้อมูลดิบให้เป็น 108 Features ที่ Scale แล้วตามที่โมเดลต้องการ"""

    # 1. สร้าง DataFrame ที่มี 108 คอลัมน์ (ตั้งค่าเริ่มต้นเป็น 0.0 ทั้งหมด)
    df = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    # 2. ใส่ค่า Numerical และ Label Encoding
    df['ปริมาณ(ตัน)'] = raw_data['ปริมาณ(ตัน)']
    df['เดือน_encoded'] = MONTH_MAP[raw_data['เดือน']]
    df['ประเภทการทำการประมง_label'] = TYPE_MAP[raw_data['ประเภทการทำการประมง']]

    # 3. ใส่ค่า One-Hot Encoding (OHE) โดยตั้งค่าคอลัมน์ที่ตรงกับ Input เป็น 1

    # OHE: เครื่องมือ (Prefix: เครื่องมือ_)
    col_name_tool = f"เครื่องมือ_{raw_data['เครื่องมือ']}"
    if col_name_tool in df.columns:
        df[col_name_tool] = 1.0

    # OHE: ขนาดเรือ (Prefix: ขนาดเรือ_)
    col_name_size = f"ขนาดเรือ_{raw_data['ขนาดเรือ']}"
    if col_name_size in df.columns:
        df[col_name_size] = 1.0

    # OHE: พื้นที่ทำการประมง (Prefix: พื้นที่ทำการประมง_)
    col_name_area = f"พื้นที่ทำการประมง_{raw_data['พื้นที่ทำการประมง']}"
    if col_name_area in df.columns:
        df[col_name_area] = 1.0

    # OHE: ชนิดสัตว์น้ำ (Prefix: ชนิดสัตว์น้ำ_)
    col_name_species = f"ชนิดสัตว์น้ำ_{raw_data['ชนิดสัตว์น้ำ']}"
    # Handling known typo/variant for 'ปลากระโทงแทงร่ม'
    if raw_data['ชนิดสัตว์น้ำ'] == 'ปลากระโทงแทงร่ม':
        col_name_species = 'ชนิดสัตว์น้ำ_ปลากะโทงแทร่ม'  # Use the exact column name from the notebook

    if col_name_species in df.columns:
        df[col_name_species] = 1.0

    # 4. Scaling ข้อมูล (ต้องเรียงคอลัมน์ให้ตรงกันเป๊ะ!)
    X_final = df[feature_columns]

    # แปลงเป็น array และ Scale
    scaled_data = scaler.transform(X_final.values)
    return scaled_data


# --- 4. ฟังก์ชันสำหรับค้นหามูลค่าเฉลี่ยจริงใน Dataset (ปรับใช้กับ DF ต้นฉบับ) ---
def get_actual_avg_value(df_actual, raw_data):
    """
    ค้นหามูลค่ารวมและปริมาณรวมของข้อมูลประวัติที่มีเงื่อนไขจัดหมวดหมู่ตรงกัน
    (ใช้ DF ต้นฉบับ)
    """

    # 1. Clean data (crucial fix for comparison failure)
    # Apply strip() to remove unwanted leading/trailing spaces in categorical columns
    df_actual_clean = df_actual.copy()
    for col in ['เดือน', 'ประเภทการทำการประมง', 'เครื่องมือ', 'ขนาดเรือ', 'พื้นที่ทำการประมง', 'ชนิดสัตว์น้ำ']:
        if col in df_actual_clean.columns and df_actual_clean[col].dtype == 'object':
            df_actual_clean[col] = df_actual_clean[col].str.strip()

    # 2. Set filter conditions based on original categorical column values
    conditions = (df_actual_clean['เดือน'] == raw_data['เดือน']) & \
                 (df_actual_clean['ประเภทการทำการประมง'] == raw_data['ประเภทการทำการประมง']) & \
                 (df_actual_clean['เครื่องมือ'] == raw_data['เครื่องมือ']) & \
                 (df_actual_clean['ขนาดเรือ'] == raw_data['ขนาดเรือ']) & \
                 (df_actual_clean['พื้นที่ทำการประมง'] == raw_data['พื้นที่ทำการประมง']) & \
                 (df_actual_clean['ชนิดสัตว์น้ำ'] == raw_data['ชนิดสัตว์น้ำ'])

    # Filter matching rows
    matching_rows = df_actual_clean[conditions]

    if not matching_rows.empty:
        total_historical_value = matching_rows['มูลค่า(พันบาท)'].sum()
        total_historical_volume = matching_rows['ปริมาณ(ตัน)'].sum()

        num_records = len(matching_rows)
        current_input_volume = raw_data['ปริมาณ(ตัน)']

        if total_historical_volume > 0:
            # Calculate Average Price per Ton (Historical)
            avg_price_per_ton = total_historical_value / total_historical_volume
            # Calculate Estimated Average Value for the current input volume
            estimated_actual_avg_value = avg_price_per_ton * current_input_volume

            return estimated_actual_avg_value, avg_price_per_ton, num_records
        else:
            # Fallback if historical volume is 0
            return matching_rows['มูลค่า(พันบาท)'].mean(), 0.0, num_records
    else:
        # Returns 0.0, 0.0, 0 if no matching categorical record is found
        return 0.0, 0.0, 0


# --- 5. Streamlit UI/Logic (Main Execution) ---

# 4.1 Header and Title
st.markdown(f"<h1 style='color: #004d40;'>{_T('title')}</h1>",
            unsafe_allow_html=True)
st.markdown("---")

# --- Language Selector ---
lang_col1, lang_col2 = st.columns([0.8, 0.2])
with lang_col2:
    current_lang = st.radio(
        "",
        ['TH', 'EN'],
        index=0 if st.session_state['lang'] == 'th' else 1,
        horizontal=True
    )
    if current_lang == 'TH' and st.session_state['lang'] != 'th':
        st.session_state['lang'] = 'th'
        st.rerun()
    elif current_lang == 'EN' and st.session_state['lang'] != 'en':
        st.session_state['lang'] = 'en'
        st.rerun()

with st.form("prediction_form", clear_on_submit=False):
    # 4.3 Input Form Layout - Grouped with container
    with st.container(border=True):
        st.markdown(f"### {_T('input_header')}")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Input 1: ปริมาณ (Numerical) - ตั้งค่าเริ่มต้น 1.0 ตัน (เพื่อให้ใกล้เคียง 56.0)
            ปริมาณ_ตัน = st.number_input(_T("input_fields")["ปริมาณ(ตัน)"], min_value=0.01, max_value=20000.0,
                                         value=1.0, step=1.0)

            # เดือน (Default: มกราคม)
            display_months = MONTH_OPTIONS if st.session_state['lang'] == 'th' else EN_MONTH_OPTIONS
            selected_month_display = st.selectbox(_T("input_fields")["เดือน"], display_months,
                                                  index=0)  # Index 0 คือ มกราคม

            if st.session_state['lang'] == 'en':
                month_index = display_months.index(selected_month_display)
                เดือน = MONTH_OPTIONS[month_index]
            else:
                เดือน = selected_month_display

        with col2:
            # ประเภทการทำการประมง (Default: พาณิชย์)
            display_type = TYPE_OPTIONS if st.session_state['lang'] == 'th' else EN_TYPE_OPTIONS
            selected_type_display = st.selectbox(_T("input_fields")["ประเภทการทำการประมง"], display_type,
                                                 index=0)  # Index 0 คือ พาณิชย์
            if st.session_state['lang'] == 'en':
                ประเภท = EN_TO_TH_MAPS["TYPE"][selected_type_display]
            else:
                ประเภท = selected_type_display

            # พื้นที่ทำการประมง (Default: อันดามัน)
            display_area = AREA_OPTIONS if st.session_state['lang'] == 'th' else EN_AREA_OPTIONS
            selected_area_display = st.selectbox(_T("input_fields")["พื้นที่ทำการประมง"], display_area,
                                                 index=0)  # Index 0 คือ อันดามัน
            if st.session_state['lang'] == 'en':
                พื้นที่ = EN_TO_TH_MAPS["AREA"][selected_area_display]
            else:
                พื้นที่ = selected_area_display

        with col3:
            # ขนาดเรือ (Default: น้อยกว่า 30 ตันกรอส)
            display_boat_size = BOAT_SIZE_OPTIONS if st.session_state['lang'] == 'th' else EN_BOAT_SIZE_OPTIONS
            selected_boat_size_display = st.selectbox(_T("input_fields")["ขนาดเรือ"], display_boat_size,
                                                      index=4)  # Index 4 คือ น้อยกว่า 30 ตันกรอส
            if st.session_state['lang'] == 'en':
                ขนาดเรือ = EN_TO_TH_MAPS["BOAT_SIZE"][selected_boat_size_display]
            else:
                ขนาดเรือ = selected_boat_size_display

            # เครื่องมือ (Default: อวนลากแผ่นตะเฆ่)
            display_tool = TOOL_OPTIONS if st.session_state['lang'] == 'th' else EN_TOOL_OPTIONS
            selected_tool_display = st.selectbox(_T("input_fields")["เครื่องมือ"], display_tool,
                                                 index=0)  # Index 0 คือ อวนลากแผ่นตะเฆ่
            if st.session_state['lang'] == 'en':
                เครื่องมือ = EN_TO_TH_MAPS["TOOL"][selected_tool_display]
            else:
                เครื่องมือ = selected_tool_display

            # ชนิดสัตว์น้ำ (Default: ปลาน้ำดอกไม้)
            display_species = SPECIES_OPTIONS if st.session_state['lang'] == 'th' else EN_SPECIES_OPTIONS
            selected_species_display = st.selectbox(_T("input_fields")["ชนิดสัตว์น้ำ"], display_species,
                                                    index=0)  # Index 0 คือ ปลาน้ำดอกไม้
            if st.session_state['lang'] == 'en':
                ชนิดสัตว์น้ำ = EN_TO_TH_MAPS["SPECIES"][selected_species_display]
            else:
                ชนิดสัตว์น้ำ = selected_species_display

    st.markdown("---")
    submitted = st.form_submit_button(_T("predict_button"), type="primary")

if submitted:
    # รวบรวมข้อมูลดิบ (7 Features)
    raw_input = {
        'ปริมาณ(ตัน)': ปริมาณ_ตัน,
        'เดือน': เดือน,
        'ประเภทการทำการประมง': ประเภท,
        'เครื่องมือ': เครื่องมือ,
        'ขนาดเรือ': ขนาดเรือ,
        'พื้นที่ทำการประมง': พื้นที่,
        'ชนิดสัตว์น้ำ': ชนิดสัตว์น้ำ,
    }

    try:
        # 1. Preprocess และ Scale (สร้าง 108 Features)
        processed_data = preprocess_input(raw_input, scaler, FEATURE_COLUMNS)

        # 2. Predict
        prediction = model.predict(processed_data)
        prediction_value = prediction[0]
        total_baht = prediction_value * 1000

        # 3. Get Actual Comparison Value
        avg_actual_value, avg_actual_price_per_ton, num_records = get_actual_avg_value(df_actual, raw_input)

        # Calculate Value per Ton for the Prediction
        predicted_price_per_ton = prediction_value / ปริมาณ_ตัน if ปริมาณ_ตัน > 0 else 0.0

        # Calculate difference percentage
        delta_percent = 0.0
        if avg_actual_value > 0 and prediction_value is not None:
            delta_percent = (prediction_value - avg_actual_value) / avg_actual_value * 100

        # --- Display Results ---

        with st.container():
            st.markdown('<div class="prediction-container">', unsafe_allow_html=True)

            st.markdown(f"## {_T('output_header')}")

            col_pred, col_actual = st.columns(2)

            with col_pred:
                # Predicted Total Value (Key Result)
                st.markdown(f"""
                    <div class="total-value-green-box">
                        <p class='total-baht-text'>{_T('total_baht')} (สำหรับ {ปริมาณ_ตัน:,.2f} ตัน)</p>
                        <p class='total-baht-value'>
                            {total_baht:,.2f} {("Baht" if st.session_state['lang'] == 'en' else "บาท")}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            with col_actual:
                # Actual Average Value Comparison
                if num_records > 0:

                    st.markdown(f"""
                        <div class="actual-avg-box">
                            <p class='actual-avg-text'>{_T('avg_actual')} (จาก {num_records:,} รายการ)</p>
                            <p class='actual-avg-value'>
                                {avg_actual_value:,.2f} พันบาท
                            </p>
                            <p style='font-size: 14px; margin-top: 10px; color: {"#004d40" if abs(delta_percent) < 10 else "#dc3545"};'>
                            (ทำนายต่างจากค่าเฉลี่ยจริง: {delta_percent:,.2f}%)
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="actual-avg-box">
                            <p class='actual-avg-text'>ไม่พบข้อมูลจริงใน Dataset</p>
                            <p style='font-size: 14px; margin-top: 10px; color: #dc3545;'>
                            (ไม่สามารถเปรียบเทียบกับค่าเฉลี่ยจริงใน Dataset ได้)
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

            # --- Row 3: Price per Ton Metric ---
            st.markdown("---")
            st.markdown("### การวิเคราะห์ราคาต่อตัน (พันบาท/ตัน)")
            price_col1, price_col2 = st.columns(2)

            with price_col1:
                st.metric(
                    label="ราคาต่อตัน (คาดการณ์โดยโมเดล)",
                    value=f"{predicted_price_per_ton:,.2f}",
                    help="ราคาต่อตันที่คาดการณ์ตามโมเดล (สะท้อนผลจากปริมาณ)"
                )
            with price_col2:
                # Determine delta for price per ton
                delta_price_per_ton = predicted_price_per_ton - avg_actual_price_per_ton
                delta_color_price = "off"
                if avg_actual_price_per_ton > 0:
                    delta_percent_price = (delta_price_per_ton / avg_actual_price_per_ton) * 100
                    if abs(delta_percent_price) > 5:  # Highlight if difference > 5%
                        delta_color_price = "inverse" if delta_price_per_ton < 0 else "normal"

                st.metric(
                    label="ราคาต่อตัน (จากค่าเฉลี่ยจริง)",
                    value=f"{avg_actual_price_per_ton:,.2f}",
                    delta=f"{delta_price_per_ton:,.2f} (Difference)",
                    delta_color=delta_color_price,
                    help="ราคาต่อตันเฉลี่ยของข้อมูลประวัติจริงที่มีเงื่อนไข (เดือน, เครื่องมือ, ชนิดสัตว์น้ำ, ฯลฯ) เหมือนกัน"
                )

            st.markdown('</div>', unsafe_allow_html=True)  # ปิด div.prediction-container

        st.balloons()

    except KeyError as e:
        st.error(f"{_T('error_input')}: {e}")
        st.warning("โปรดตรวจสอบว่าทุกช่องถูกเลือกถูกต้องตามข้อมูลที่คุณใช้เทรน")
    except Exception as e:
        st.exception(f"{_T('error_prediction')}: {e}")
