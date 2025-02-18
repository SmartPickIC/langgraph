import streamlit as st
from datetime import datetime
from videoextractor import survice
import threading
import time
from queue_manager import get_queue
from queue import Empty

firstrun = True

def main(videoextractor):
   
    log_queue = get_queue() 
    
    st.set_page_config(
        page_title="Search UI",
        layout="wide"
    )

    # CSS 스타일 적용
    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            padding: 12px 20px;
            border-radius: 24px;
            border: 1px solid #dfe1e5;
            font-size: 16px;
            width: 100%;
            margin-bottom: 20px;
        }
        .stTextInput > div > div > input:hover,
        .stTextInput > div > div > input:focus {
            box-shadow: 0 1px 6px rgba(32,33,36,.28);
            border-color: rgba(223,225,229,0);
        }
        </style>
    """, unsafe_allow_html=True)

    # 세션 상태 초기화
    if 'logs' not in st.session_state:
        st.session_state.logs = []

    # 검색창과 버튼을 한 줄에 배치
    col1, col2 = st.columns([6, 1])
    
    with col1:
        # label 추가 및 숨김 처리
        query = st.text_input(
            label="Search Input",
            label_visibility="collapsed",
            placeholder="검색어를 입력하세요",
            key="search_input"
        )
    
    with col2:
        search_button = st.button("검색", key="search_button")

    # 검색 버튼 클릭 또는 엔터키 입력 시
    if search_button or (query and st.session_state.get('last_query') != query):
        if query:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_log = f"{current_time}: {query}"
            st.session_state.logs.append(new_log)
            st.session_state.last_query = query
            videoextractor.extract_from_query(user_query_raw=query)


    # 외부 로그 처리
    process_external_logs()

    # 로그 영역
    st.markdown("### 검색 기록")
    log_area = st.empty()
    
    # 자동 갱신을 위한 placeholder
    if 'placeholder' not in st.session_state:
        st.session_state.placeholder = st.empty()
    
    # 로그를 역순으로 표시하여 최신 로그가 항상 상단에 표시
    log_text = "\n".join(reversed(st.session_state.logs))
    st.session_state.placeholder.text_area(
        label="Log Area",
        value=log_text,
        height=300,
        disabled=True,
        label_visibility="collapsed"
    )
    
    #if not log_queue.empty(): # 이 조건을 정의해야 합니다
    #    time.sleep(0.1)
    #    st.rerun()

def process_external_logs():
    """외부 로그 큐에서 로그를 가져와 화면에 추가"""
    queue = get_queue()
    try:
        while True:
            log = queue.get_nowait()
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_log = f"{current_time}: {log}"
            if formatted_log not in st.session_state.logs:
                st.session_state.logs.append(formatted_log)
    except Empty:  # queue.Empty 대신 Empty 사용
        pass

def add_external_log(log_message):  # log_queue 파라미터 제거
    """외부에서 로그를 추가하기 위한 함수"""
    queue = get_queue()
    queue.put(log_message)
    
    
if __name__ == "__main__":
    if 'videoextractor' not in st.session_state:
        st.session_state.videoextractor = survice()
    main(st.session_state.videoextractor)