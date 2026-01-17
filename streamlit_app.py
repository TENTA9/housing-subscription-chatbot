import streamlit as st
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# housing_rag 모듈 import
from housing_rag import EnhancedRAGSystem, initialize_rag_system

# 페이지 설정
st.set_page_config(
    page_title="주택청약 챗봇",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    /* 메인 컨테이너 */
    .main {
        padding: 0rem 1rem;
    }
    
    /* 사용자 메시지 */
    .user-message {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        margin-left: 20%;
        text-align: left;
        color: #000000;  /* 검정색 */
    }
    
    /* AI 메시지 */
    .ai-message {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        margin-right: 20%;
        text-align: left;
        color: #000000;  /* 검정색 */
    }
    
    /* 출처 스타일 */
    .source-badge {
        display: inline-block;
        background-color: #1976D2;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        margin: 0.2rem;
        font-size: 0.8rem;
    }
    
    /* 타임스탬프 */
    .timestamp {
        color: #666;
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }
    
    /* 사이드바 */
    .sidebar .sidebar-content {
        background-color: #FAFAFA;
    }
    
    /* 입력창 */
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# RAG 시스템 초기화 (캐싱)
@st.cache_resource(show_spinner=False)
def load_rag_system():
    """RAG 시스템을 로드하고 캐싱합니다."""
    base_path = r"C:\Users\user\Desktop\bkms"
    pdf_path = os.path.join(base_path, "pdfs")
    cache_dir = os.path.join(base_path, "cache")
    
    with st.spinner("🔄 RAG 시스템 초기화 중... (처음만 시간이 걸립니다)"):
        rag_system, documents = initialize_rag_system(pdf_path, cache_dir)
    
    return rag_system, documents

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.documents = None

# RAG 시스템 로드
if st.session_state.rag_system is None:
    try:
        st.session_state.rag_system, st.session_state.documents = load_rag_system()
    except Exception as e:
        st.error(f"❌ 시스템 초기화 실패: {str(e)}")
        st.stop()

# 사이드바
with st.sidebar:
    st.title("🏠 주택청약 챗봇")
    st.markdown("---")
    
    # 문서 목록
    st.subheader("📁 처리된 문서")
    if st.session_state.documents:
        for doc_name, info in st.session_state.documents.items():
            with st.expander(f"📄 {doc_name}"):
                st.write(f"**청크 수:** {info['chunk_count']}개")
                st.write(f"**경로:** {info['path']}")
                
                # PDF 파일 열기 버튼
                if st.button(f"파일 열기", key=f"open_{doc_name}"):
                    try:
                        pdf_path = Path(info['path'])
                        if pdf_path.exists():
                            os.startfile(str(pdf_path))
                            st.success(f"✅ {doc_name} 열기 완료")
                        else:
                            st.error(f"❌ 파일을 찾을 수 없습니다")
                    except Exception as e:
                        st.error(f"❌ 파일 열기 실패: {str(e)}")
    
    st.markdown("---")
    
    # 대화 초기화 버튼
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # 사용 가이드
    st.subheader("💡 사용 가이드")
    st.markdown("""
    1. 하단 입력창에 질문 입력
    2. Enter 또는 전송 버튼 클릭
    3. 답변과 출처 확인
    4. 출처 클릭 시 상세 내용 표시
    
    **예시 질문:**
    - 청약 가점이 낮은 20대가 당첨 가능성을 높이는 방법은?
    - 특별공급 신청 조건은?
    - 재당첨 제한이란?
    """)
    
    st.markdown("---")
    st.caption(f"📊 문서: {len(st.session_state.documents)}개")
    st.caption(f"💬 대화: {len(st.session_state.messages)//2}개")

# 메인 화면
st.title("💬 주택청약 상담 챗봇")
st.markdown("주택청약 관련 궁금한 점을 질문해주세요!")
st.markdown("---")

# 대화 기록 표시
chat_container = st.container()

with chat_container:
    if len(st.session_state.messages) == 0:
        st.info("👋 안녕하세요! 주택청약에 대해 무엇이든 물어보세요.")
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong>👤 You</strong><br/>
                {message["content"]}
                <div class="timestamp">{message["timestamp"]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="ai-message">
                <strong>🤖 AI Assistant</strong><br/>
                {message["content"]}
                <div class="timestamp">{message["timestamp"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 출처 표시
            if "sources" in message and message["sources"]:
                with st.expander("📚 참고 문서 및 출처 보기"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**[{i}] {source['doc_name']}**")
                        if source.get('page'):
                            st.caption(f"📄 페이지: {source['page']}")
                        
                        # 원문 내용
                        with st.expander(f"📖 원문 내용 {i}"):
                            st.text(source['content'])
                        
                        st.markdown("---")

# 질문 처리 함수
st.markdown("---")

def process_question(question):
    """질문을 처리하고 답변 생성"""
    # 사용자 메시지 추가
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # 답변 생성
    with st.spinner("🤔 답변을 생성하는 중..."):
        try:
            # 컨텍스트 가져오기
            context = st.session_state.rag_system._get_relevant_context(question)
            
            # 답변 생성
            answer = st.session_state.rag_system.answer_question(
                question=question,
                show_metadata=False
            )
            
            # 출처 정보 추출
            sources = []
            if context:
                contexts = context.split("="*50)
                for doc in contexts:
                    if not doc.strip() or "[Document:" not in doc:
                        continue
                    
                    try:
                        # 문서명 추출
                        doc_name = doc.split("[Document:")[1].split("]")[0].strip()
                        
                        # 페이지 정보 추출 (있다면)
                        page = None
                        if "[Page" in doc:
                            try:
                                page = doc.split("[Page")[1].split("]")[0].strip()
                            except:
                                pass
                        
                        # 원문 내용 추출
                        if "Content:" in doc:
                            content = doc.split("Content:")[1].strip()
                            # 너무 긴 경우 잘라내기
                            if len(content) > 500:
                                content = content[:500] + "..."
                        else:
                            content = "내용을 불러올 수 없습니다."
                        
                        sources.append({
                            "doc_name": doc_name,
                            "page": page,
                            "content": content
                        })
                    except Exception as e:
                        continue
            
            # 타이핑 애니메이션을 위한 placeholder
            response_placeholder = st.empty()
            displayed_text = ""
            
            # 타이핑 애니메이션 효과
            for char in answer:
                displayed_text += char
                response_placeholder.markdown(f"""
                <div class="ai-message">
                    <strong>🤖 AI Assistant</strong><br/>
                    {displayed_text}▌
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.01)  # 타이핑 속도 조절
            
            # 최종 답변
            response_placeholder.empty()
            
            # AI 메시지 추가
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
        except Exception as e:
            st.error(f"❌ 오류 발생: {str(e)}")
            
            # 재시도 버튼
            if st.button("🔄 다시 시도"):
                # 마지막 사용자 메시지 가져오기
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    last_question = st.session_state.messages[-1]["content"]
                    st.session_state.messages.pop()  # 실패한 질문 제거
                    process_question(last_question)  # 재시도
                    st.rerun()

# 입력창
with st.form(key='question_form', clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "질문을 입력하세요",
            key="user_input",
            placeholder="예: 청약 가점이 낮은 20대가 당첨 가능성을 높이는 방법은?",
            label_visibility="collapsed"
        )
    
    with col2:
        submit = st.form_submit_button("📤 전송", use_container_width=True)
    
    if submit and user_input:
        process_question(user_input)
        st.rerun()

# 하단 여백
st.markdown("<br/>" * 3, unsafe_allow_html=True)
