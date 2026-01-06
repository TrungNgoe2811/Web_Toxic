import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# ==========================================
# 1. CẤU HÌNH & CSS
# ==========================================
# Cấu hình địa chỉ API Server (Backend)
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Toxic Guard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Custom (Theo code bạn gửi)
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
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR & LOGIC
# ==========================================

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2562/2562186.png", width=100)    
    st.title("⚙️ Control Panel")
    st.info("Client-Server Mode (API)")
    st.markdown("---")
    st.write("Authored by: **Trung Ngoe (OnsraNz)**")
    st.write("Version: **Super promax**")
    
    st.markdown("---")
    # Tính năng xem Log từ Server (Giữ lại để debug)
    if st.checkbox("Show Server Logs"):
        st.subheader("System Logs")
        try:
            res = requests.get(f"{API_URL}/logs", timeout=2)
            if res.status_code == 200:
                logs = res.json()["logs"]
                for line in logs:
                    st.text(line.strip())
            else:
                st.error("Không lấy được log")
        except:
            st.error("⚠️ Server chưa bật!")

# --- Main Content ---
st.title("🛡️ AI TOXIC GUARD SYSTEM")
st.caption("🚀 Hệ thống giám sát nội dung bình luận tự động (API Version)")

# HÀM CALLBACK ĐỂ XÓA TEXT 
def clear_text():
    st.session_state["user_input_key"] = ""

# Layout nhập liệu
col1, col2 = st.columns([2, 1])

with col1:
    # Ô nhập liệu có gắn key để xóa được
    user_input = st.text_area("📡 Nhập dữ liệu đầu vào (Input):", 
                              height=150, 
                              placeholder="Type your comment here...",
                              key="user_input_key")
    
    # Chia cột cho 2 nút bấm
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        analyze_btn = st.button("🚀 KÍCH HOẠT PHÂN TÍCH", use_container_width=True)
    with btn_col2:
        st.button("🔄 LÀM MỚI (RESET)", on_click=clear_text, use_container_width=True)

with col2:
    st.markdown("### 📝 Hướng dẫn")
    st.markdown("""
    1. Đảm bảo file `api.py` đang chạy.
    2. Nhập bình luận tiếng Anh.
    3. Nhấn **Kích hoạt** để gửi tới Server.
    """)

# ==========================================
# 3. XỬ LÝ KHI BẤM NÚT
# ==========================================
if analyze_btn:
    if not user_input.strip():
        st.warning("⚠️ Cảnh báo: Dữ liệu đầu vào trống!")
    else:
        try:
            with st.spinner("Đang gửi dữ liệu tới Server..."):
                # --- GỌI API (Thay thế cho model chạy trực tiếp) ---
                payload = {"text": user_input}
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    preds = data["predictions"] # Lấy danh sách xác suất
                    
                    # --- HIỂN THỊ KẾT QUẢ ---
                    st.markdown("---")
                    st.subheader("📊 KẾT QUẢ PHÂN TÍCH TỪ SERVER")

                    res_col1, res_col2 = st.columns([1, 1])

                    with res_col1:
                        # Biểu đồ Radar
                        df = pd.DataFrame(dict(
                            r=list(preds.values()), 
                            theta=list(preds.keys())
                        ))
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
                        # Kiểm tra xem có nhãn nào > 50% không
                        is_toxic = any(score > 0.5 for score in preds.values())
                        
                        if is_toxic:
                            st.error("🚨 SERVER: NỘI DUNG ĐỘC HẠI!")
                        else:
                            st.success("✅ SERVER: AN TOÀN")

                        metric_cols = st.columns(2)
                        for i, (label, score) in enumerate(preds.items()):
                            with metric_cols[i % 2]:
                                st.metric(
                                    label=label, 
                                    value=f"{score*100:.2f}%", 
                                    delta="Nguy hiểm" if score > 0.5 else "Ổn định",
                                    delta_color="inverse" if score > 0.5 else "normal"
                                )
                else:
                    st.error(f"Lỗi API Server: {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("❌ LỖI KẾT NỐI: Không tìm thấy Server! Hãy chắc chắn bạn đã chạy lệnh 'uvicorn api:app --reload' ở cửa sổ Terminal kia.")