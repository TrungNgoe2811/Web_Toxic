# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import pickle
import re
import logging
import datetime

# 1. CẤU HÌNH LOGGING
# Ghi log vào file 'system.log' và cả màn hình console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 2. ĐỊNH NGHĨA MODEL & HÀM XỬ LÝ (Copy từ notebook sang)
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

# 3. KHỞI TẠO FASTAPI & LOAD MODEL
app = FastAPI(title="Toxic Comment API", description="API phát hiện bình luận độc hại")

# Biến toàn cục để lưu model
model = None
word_to_idx = None
LABELS = [
    "Độc hại (Toxic)", 
    "Cực kỳ độc hại (Severe)", 
    "Tục tĩu (Obscene)", 
    "Đe dọa (Threat)", 
    "Xúc phạm (Insult)", 
    "Thù ghét & Kỳ thị (Hate)"
]

@app.on_event("startup")
def load_model():
    global model, word_to_idx
    try:
        # Load Vocab
        with open('vocab.pkl', 'rb') as f:
            word_to_idx = pickle.load(f)
        
        # Init Model
        VOCAB_SIZE = len(word_to_idx) + 1
        model = ToxicLSTM(VOCAB_SIZE, 128, 128, 6)
        model.load_state_dict(torch.load('saved_model.pth', map_location=torch.device('cpu')))
        model.eval()
        logger.info("✅ Model & Vocab loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")

# 4. ĐỊNH NGHĨA DATA MODEL (Pydantic)
class CommentRequest(BaseModel):
    text: str

# 5. API ENDPOINT
@app.post("/predict")
async def predict(request: CommentRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Preprocess
    clean_txt = clean_text(request.text)
    tokens = clean_txt.split()
    vec = [word_to_idx.get(w, 0) for w in tokens]
    
    # Padding (max_len=100)
    max_len = 100
    if len(vec) < max_len:
        vec = vec + [0] * (max_len - len(vec))
    else:
        vec = vec[:max_len]
        
    tensor_input = torch.tensor([vec], dtype=torch.long)
    
    # Predict
    with torch.no_grad():
        outputs = model(tensor_input)
        probs = torch.sigmoid(outputs).squeeze().tolist() # Convert to list for JSON
    
    # Tạo kết quả trả về
    result = {label: round(prob, 4) for label, prob in zip(LABELS, probs)}
    is_toxic = any(p > 0.5 for p in probs)
    
    # GHI LOG
    log_msg = f"Input: '{request.text[:30]}...' | Toxic: {is_toxic} | Scores: {result}"
    logger.info(log_msg)
    
    return {
        "text": request.text,
        "is_toxic": is_toxic,
        "predictions": result
    }

# Endpoint để đọc log (cho UI hiển thị)
@app.get("/logs")
def get_logs():
    try:
        with open("system.log", "r") as f:
            lines = f.readlines()
        return {"logs": lines[-20:]} # Trả về 20 dòng log mới nhất
    except FileNotFoundError:
        return {"logs": ["Chưa có log nào."]}