import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import numpy as np
import pandas as pd
import plotly.express as px

# 1. CẤU HÌNH TRANG

st.set_page_config(
    page_title="AI Toxic Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom: Nút Làm mới sẽ có màu khác để phân biệt
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1 { color: #00FF99 !important; text-shadow: 0 0 10px #00FF99; }
    .stTextArea textarea { background-color: #262730; color: white; border-radius: 10px; }
    
    /* Style cho nút Phân tích (Màu Đỏ) */
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        border-radius: 20px;
        height: 50px;
        font-weight: bold;
        font-size: 18px;
        border: 1px solid #FF4B4B;
    }
    div.stButton > button:first-child:hover {
        background-color: #FF0000;
        border: 1px solid white;
    }

    /* Style riêng cho nút Làm mới (Màu Trắng/Xám) - Cần trick CSS một chút hoặc để mặc định */
</style>
""", unsafe_allow_html=True)


# 2. LOGIC MODEL (GIỮ NGUYÊN)


LABELS = ["Độc hại (Toxic)", "Cực kỳ độc hại (Severe)", "Tục tĩu (Obscene)", 
          "Đe dọa (Threat)", "Xúc phạm (Insult)", "Thù ghét (Hate)"]

class ToxicLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.3):
        super(ToxicLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        embed = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embed)
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.dropout(hidden_concat)
        out = self.fc(out)
        return out

def clean_text(text):
    text = str(text).lower()           
    text = re.sub(r'\n', ' ', text)          
    text = re.sub(r'[^a-z0-9\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip() 
    return text

@st.cache_resource
def load_resources():
    with open('vocab.pkl', 'rb') as f:
        word_to_idx = pickle.load(f)
    VOCAB_SIZE = len(word_to_idx) + 1
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 128
    OUTPUT_DIM = 6 
    model = ToxicLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model.load_state_dict(torch.load('saved_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model, word_to_idx

def preprocess_input(text, word_to_idx, max_len=100):
    text = clean_text(text)
    tokens = text.split()
    vec = [word_to_idx.get(w, 0) for w in tokens]
    if len(vec) < max_len:
        vec = vec + [0] * (max_len - len(vec))
    else:
        vec = vec[:max_len]
    return torch.tensor([vec], dtype=torch.long)


# 3. GIAO DIỆN WEB & CHỨC NĂNG LÀM MỚI


# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1085/1085442.png", width=100)    
    st.title("⚙️ Control Panel")
    st.info("Hệ thống sử dụng mô hình **Bi-LSTM** để phân tích cảm xúc và phát hiện ngôn ngữ độc hại.")
    st.markdown("---")
    st.write("Authored by: **Trung Ngoe (OnsraNz)**")
    st.write("Version: **Super promax**")

# --- Main Content ---
st.title("🛡️ AI TOXIC GUARD SYSTEM")
st.caption("🚀 Hệ thống giám sát nội dung bình luận tự động")

try:
    model, word_to_idx = load_resources()
except Exception as e:
    st.error(f"Lỗi hệ thống: {e}")
    st.stop()

#  HÀM CALLBACK ĐỂ XÓA TEXT 
def clear_text():
    st.session_state["user_input_key"] = ""

# Layout nhập liệu
col1, col2 = st.columns([2, 1])

with col1:
   
    user_input = st.text_area("📡 Nhập dữ liệu đầu vào (Input):", 
                              height=150, 
                              placeholder="Type your comment here...",
                              key="user_input_key")
    
    # Chia cột cho 2 nút bấm nằm ngang hàng
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        analyze_btn = st.button("🚀 KÍCH HOẠT PHÂN TÍCH", use_container_width=True)
    with btn_col2:
        # Nút làm mới gọi hàm clear_text
        st.button("🔄 LÀM MỚI (RESET)", on_click=clear_text, use_container_width=True)

with col2:
    st.markdown("### 📝 Hướng dẫn")
    st.markdown("""
    1. Nhập bình luận tiếng Anh.
    2. Nhấn **Kích hoạt** để xem kết quả.
    3. Nhấn **Làm mới** để xóa nhanh nội dung cũ.
    """)

# Xử lý khi bấm nút Phân tích
if analyze_btn:
    if not user_input.strip():
        st.warning("⚠️ Cảnh báo: Dữ liệu đầu vào trống!")
    else:
        # Xử lý
        tensor_input = preprocess_input(user_input, word_to_idx)
        with torch.no_grad():
            outputs = model(tensor_input)
            probs = torch.sigmoid(outputs).squeeze().numpy()

        # --- HIỂN THỊ KẾT QUẢ ---
        st.markdown("---")
        st.subheader("📊 KẾT QUẢ PHÂN TÍCH THỜI GIAN THỰC")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            # Biểu đồ Radar
            df = pd.DataFrame(dict(r=probs, theta=LABELS))
            fig = px.line_polar(df, r='r', theta='theta', line_close=True)
            fig.update_traces(fill='toself', line_color='#00FF99')
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title="Vùng Phủ Độc Hại"
            )
            st.plotly_chart(fig, use_container_width=True)

        with res_col2:
            st.write("#### 🔍 Chi tiết chỉ số:")
            if np.any(probs > 0.5):
                st.error("🚨 PHÁT HIỆN: NỘI DUNG ĐỘC HẠI!")
            else:
                st.success("✅ TRẠNG THÁI: AN TOÀN")

            metric_cols = st.columns(2)
            for i, label in enumerate(LABELS):
                score = probs[i]
                with metric_cols[i % 2]:
                    st.metric(
                        label=label, 
                        value=f"{score*100:.2f}%", 
                        delta="Nguy hiểm" if score > 0.5 else "Ổn định",
                        delta_color="inverse" if score > 0.5 else "normal"
                    )