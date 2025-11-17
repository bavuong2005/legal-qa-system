#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Demo App for Vietnamese Law QA System - Chatbot Interface
"""
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import time
# Giả định rằng bạn đã clone repo và các file này nằm trong thư mục 'backend'
# (Nếu file của bạn tên khác, hãy sửa lại đường dẫn import)
from backend.retriever_custom import retrieve
from backend.generator import generate_answer

# Page config
st.set_page_config(
    page_title="RoadLawQA",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chat interface
st.markdown("""
<style>
/* ... (các class .main, .chat-header, .chat-title, .chat-subtitle giữ nguyên) ... */
.main { background-color: #f7f7f8; }
.chat-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1.5rem; border-radius: 10px; text-align: center; margin-bottom: 1rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
.chat-title { font-size: 1.8rem; font-weight: bold; margin: 0; }
.chat-subtitle { font-size: 0.9rem; opacity: 0.9; margin-top: 0.3rem; }

/* Tin nhắn user*/
.user-message {
    background-color: #e5e5ea;
    color: #111;
    padding: 0.6rem 1rem; /* giảm padding trên/dưới */
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0 0.5rem auto;
    white-space: pre-line;
    max-width: 80%;
    width: fit-content;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    word-wrap: break-word;
    animation: fadeInUp 0.3s ease-out;
    text-align: left;
}


/* --- ĐÃ SỬA TỪ ĐÂY --- */

/* 1. Đây là KHUNG BAO NGOÀI MỚI cho toàn bộ phản hồi của Bot */
.bot-response-container {
    margin: 0.5rem auto 0.5rem 0; /* Căn lề trái */
    max-width: 85%;              /* Giới hạn chiều rộng tối đa */
    width: fit-content;          /* Tự co dãn theo nội dung */
    animation: fadeInUp 0.3s ease-out; /* Áp dụng animation cho cả khối */
}

/* 2. Tin nhắn bot (Đã BỎ CÁC THUỘC TÍNH LAYOUT) */
.bot-message {
    background-color: #e5e5ea; /* xám nhạt */
    color: #111;
    padding: 1rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    word-wrap: break-word;
    /* Đã BỎ: margin, max-width, width, animation */
}

/* 3. Chỉnh expander (Nguồn tham khảo) để nó khớp với style */
.bot-response-container .stExpander {
    border: none;
    box-shadow: none;
    margin-top: 0.5rem;
    border-radius: 10px;
    background-color: #f0f0f0; /* Một màu xám nhạt khác biệt */
}
.bot-response-container .stExpander header {
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
    border-radius: 10px;
}

/* --- KẾT THÚC PHẦN SỬA --- */


/* Bot icon */
.bot-icon {
    display: inline-block;
    background: #4f46e5;
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    text-align: center;
    line-height: 32px;
    margin-right: 0.5rem;
    font-weight: bold;
}

/* Nguồn tham khảo */
.source-item {
    background: #fff8dc;
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    margin: 0.3rem 0;
    border-left: 3px solid #ffc107;
    font-size: 0.85rem;
    color: #555;
}

/* ... (Phần còn lại của CSS giữ nguyên) ... */
.time-badge { background: #e3f2fd; color: #1976d2; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.75rem; display: inline-block; margin-top: 0.5rem; }
.stChatInputContainer { border-top: 2px solid #e0e0e0; background: #f2f2f2; padding: 1rem 0; }
.welcome-card { background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; margin: 2rem 0; }
.sample-question { background: #f8f9fa; padding: 0.8rem 1rem; border-radius: 10px; margin: 0.5rem 0; cursor: pointer; border: 1px solid #dee2e6; transition: all 0.3s; }
.sample-question:hover { background: #e9ecef; border-color: #667eea; transform: translateY(-2px); }
.metric-inline { display: inline-block; background: #f0f0f0; padding: 0.3rem 0.8rem; border-radius: 8px; margin: 0.2rem; font-size: 0.8rem; color: #666; }
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'running_rag' not in st.session_state:
    st.session_state.running_rag = False
if 'chat_started' not in st.session_state:
    st.session_state.chat_started = False


# ===== WELCOME SCREEN (Trang giới thiệu) =====
if not st.session_state.chat_started:
    # Hiển thị HOÀN TOÀN trang giới thiệu
    st.markdown("""
    <div style='text-align: center; padding: 0.5rem 2rem 2rem 2rem;'>
        <h1 style='color: #667eea; font-size: 3rem; margin: 0; font-weight: bold;'>RoadLawQA</h1>
        <h2 style='color: #666; font-size: 1.3rem; margin: 0.8rem 0 1.5rem 0; font-weight: 400;'>Hệ thống Hỏi Đáp về Luật An Toàn Giao Thông Đường Bộ Việt Nam</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Nút "Bắt đầu trò chuyện" - căn giữa
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("🚀 Bắt đầu trò chuyện", use_container_width=True, key="start_chat_btn"):
            st.session_state.chat_started = True
            st.rerun()
    
    # Thêm khoảng trống
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3 ảnh minh họa với chiều cao cố định
    col1, col2, col3 = st.columns(3, gap="small")
    
    with col1:
        st.image("assets/law.png", use_container_width=True)
    
    with col2:
        st.image("assets/traffic.png", use_container_width=True)
    
    with col3:
        st.image("assets/legal.png", use_container_width=True)
    
    st.stop()  # Dừng execution tại đây, không hiển thị gì khác


# ===== CHAT INTERFACE (Chỉ hiển thị khi đã bấm "Bắt đầu") =====

# Header
st.markdown("""
<div class="chat-header">
    <div class="chat-title">RoadLawQA - Hỏi Đáp Luật An Toàn Giao Thông Đường Bộ</div>
    <div class="chat-subtitle">Hỏi đáp tức thì về Luật Giao Thông Việt Nam</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")
    
    k_value = st.slider(
        "Số lượng chunks",
        min_value=1,
        max_value=10,
        value=5,
        help="Số đoạn văn bản được tìm kiếm"
    )
    
    st.markdown("---")
    
    st.markdown("""
    **📚 Dữ liệu:**
    - Nghị định 168/2024/NĐ-CP
    - Luật 36/2024/QH15
    - Luật 35/2024/QH15
    """)
    
    st.markdown("---")
    
    if st.button("🗑️ Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_started = False
        st.session_state.running_rag = False
        st.rerun()

# Biến tạm để lưu input
user_input = None

# Hiển thị câu hỏi mẫu nếu chưa có tin nhắn
sample_questions = [
    "Kết cấu hạ tầng đường bộ bao gồm những gì?",
    "Người đi xe dàn hàng ba bị xử phạt như thế nào?",
    "Làn đường được định nghĩa là gì?",
    "Xe máy chở 2 người trở lên có bị phạt không?",
]

if len(st.session_state.messages) == 0:
    st.markdown("""
    <div style='text-align: center; margin: 2rem 0;'>
        <h4 style='color: #666;'>💡 Bạn có thể bắt đầu với các câu hỏi này:</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    for idx, question in enumerate(sample_questions):
        with col1 if idx % 2 == 0 else col2:
            if st.button(f"💡 {question}", key=f"sample_{idx}", use_container_width=True):
                user_input = question

# Chat input
chat_input = st.chat_input("Nhập câu hỏi của bạn...")
if chat_input:
    user_input = chat_input.strip()

# Display messages
for idx, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        # Xử lý text: loại bỏ khoảng trắng thừa ở đầu/cuối mỗi dòng, loại bỏ dòng trống và escape HTML
        user_text = message['content'].strip()
        # Loại bỏ khoảng trắng thừa ở đầu/cuối mỗi dòng và bỏ dòng trống
        user_text = '\n'.join(line.strip() for line in user_text.split('\n') if line.strip())
        user_text = user_text.replace('<', '&lt;').replace('>', '&gt;')
        # Chuyển newline thành <br> để hiển thị đúng
        user_text = user_text.replace('\n', '<br>')
        st.markdown(f'<div style="text-align: right;"><div class="user-message">{user_text}</div></div>', unsafe_allow_html=True)
    else:
        with st.container():
            st.markdown('<div class="bot-response-container">', unsafe_allow_html=True)
            
            st.markdown(f"""
                <div class="bot-message">
                    <span class="bot-icon">⚖️</span>
                    <strong>RoadLawQA</strong><br><br>
                    {message['content']}
                </div>
            """, unsafe_allow_html=True)
            
            if 'sources' in message and message['sources']:
                with st.expander("📚 Nguồn tham khảo", expanded=False):
                    for i, src in enumerate(message['sources'], 1):
                        if src:
                            st.markdown(f'<div class="source-item">[{i}] {src}</div>', unsafe_allow_html=True)
            
            if 'metrics' in message:
                m = message['metrics']
                st.markdown(f"""
                <div style='text-align: left; margin-top: 0.5rem;'>
                    <span class="metric-inline">⏱️ {m['total']:.2f}s</span>
                    <span class="metric-inline">🔍 {m['retrieval']:.2f}s</span>
                    <span class="metric-inline">🤖 {m['generation']:.2f}s</span>
                    <span class="metric-inline">📄 {m['chunks']} chunks</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

# Handle RAG logic
if user_input and not st.session_state.running_rag:
    st.session_state.running_rag = True 
    st.session_state.messages.append({
        "role": "user",
        "content": user_input.strip()
    })
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and st.session_state.running_rag:
    last_user_question = st.session_state.messages[-1]["content"]
    
    with st.spinner("Đang suy nghĩ..."):
        start_time = time.time()
        
        try:
            t0 = time.time()
            context, sources = retrieve(last_user_question, k=k_value)
            retrieval_time = time.time() - t0
            
            t1 = time.time()
            answer, sources = generate_answer(last_user_question, context, sources)
            generation_time = time.time() - t1
            
            total_time = time.time() - start_time
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "metrics": {
                    "total": total_time,
                    "retrieval": retrieval_time,
                    "generation": generation_time,
                    "chunks": len(sources)
                }
            })
            
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ Xin lỗi, đã xảy ra lỗi: {str(e)}"
            })
    
    st.session_state.running_rag = False
    st.rerun()