#!/usr/bin/env python3
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import tiktoken
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma 
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI as OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.utils import filter_complex_metadata
from dotenv import load_dotenv
import json
from langchain.prompts import PromptTemplate
from time import sleep
import time
import pickle
from utility import Node
from queue_manager import add_log
import re

globalist=[]

def log_wrapper(log_message):
    globalist.append(log_message)
    print(log_message)
    add_log(log_message)  
def truncate_text_by_tokens(text, max_tokens, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        return encoding.decode(tokens)
    return text
def token_bool(text, model="gpt-4o-mini",target=1500):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    length=len(tokens)
    return length>target
def cal_token(text, model="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)
def simple_filter(metadata):
    simple_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            simple_metadata[k] = v
        else:
            # 복잡한 타입은 JSON 문자열로 변환하여 저장
            simple_metadata[k] = json.dumps(v, ensure_ascii=False)
    return simple_metadata


def compress_subtitles(text, chunk_size=500, overlap=50):  # overlap이 +/-
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=abs(overlap) if overlap > 0 else 0  # 양수면 overlap 적용
    )
    chunks = text_splitter.split_text(text)
    
    if overlap < 0:  # 음수면 각 청크를 자름
        trimmed_chunks = []
        for chunk in chunks:
            if len(chunk) > abs(overlap) * 2:
                trimmed = chunk[abs(overlap):-abs(overlap)]
                trimmed_chunks.append(trimmed)
        return trimmed_chunks
    
    return chunks 

def setting_tockens(text,target=1600,model="gpt-4o-mini",chunk_size=500):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    chunck_overrap= int(round((target-len(tokens))/2/target*chunk_size,0))
    if -chunck_overrap > chunk_size/4:
        #n=1
        while -chunck_overrap > chunk_size/4:
            words = text.split(" ")
            # 홀수 인덱스만 가져오기 (1,3,5...)
            filtered_words = words[1::2]
            text = "".join(filtered_words)
            tokens = encoding.encode(text)
            chunck_overrap= int(round((target-len(tokens))/2/target*chunk_size,0))
            #print (f"chunck_overrap:{n}번째 청크 제거")
            #n+=1
    compresed_txt=compress_subtitles(text, chunk_size=chunk_size, overlap=chunck_overrap)
    return compresed_txt, chunck_overrap









def compress_text(subtitle_text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(subtitle_text)

def count_tokens(text, model="gpt-4o-"):
    # 모델에 맞는 인코딩 방식을 가져옵니다.
    encoding = tiktoken.encoding_for_model(model)
    # 텍스트를 토큰으로 인코딩합니다.
    tokens = encoding.encode(text)
    # 토큰의 개수를 반환합니다.
    return len(tokens)
def load_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        # splitlines()는 각 줄을 리스트로 반환하면서 줄바꿈 문자는 제거합니다.
        file = f.read().splitlines()
    return file


class Dataprocessor:
    def __init__(self, target_dir="youtube", ref_file="YTref.txt",pickle_file="data.pkl"):
        self.tocken_count=0
        current_dir = os.getcwd()
        self.keywords={}
        self.keywords["samsung"]=load_file('samsung_keyword')
        self.keywords["apple"]=load_file('apple_keyword')
        self.keywords["xiaomi"]=load_file('Xiaomi_ketword')
        self.keywords["art"]=load_file('art_keyword')
        self.keywords["game"]=load_file('game_keyword')
        self.keywords["study"]=load_file('study_keyword')
        self.keywords["performance"]=load_file('high_performance_keyword')
        self.keywords["effectiveness"]=load_file('cost-effectiveness_keyword')
        self.k_value=10
        self.buffer=None
        self.lln = None
        self.qa_video=None
        self.vectorstore = None
        self.retriever = None
        self.target_dir = os.path.join(current_dir, "youtube")
        self.pickle_file = pickle_file
        self.ytref_list = self.load_ytref( ref_file)
        self.youtube_contents = self.load_youtube_folder()
        self.keyword_llm=Node("""
                                    아래 두 종류의 키워드 리스트를 참고하세요.

                                    [1] 회사 관련 키워드: 
                                    - samsung 
                                    - apple 
                                    - xiaomi

                                    [2] 특성 관련 키워드:
                                    - effectiveness 
                                    - performance 
                                    - study 
                                    - game 
                                    - art

                                    사용자의 질문을 분석한 후, 위 리스트에서 가장 관련성 높은 키워드를 영문으로 선택해 주세요.  
                                    응답은 반드시 아래의 키워드들(각각 또는 조합)만 사용해야 하며, 다른 단어나 문장은 절대 포함하면 안 됩니다.  
                                    만약 질문과 관련된 키워드를 찾지 못하면, 아무런 응답도 하지 마세요.

                                    [중요]
                                     가장 중요한 임무는 응답에 art, game, study, performance, effectiveness, samsung, apple, xiaomi " 외에 어떠한 글자 어떠한 말도 포함하지 않고 입력해서는 안됩니다"
                                    
                                    예시:
                                    - "그림 연습용 아이패드를 추천해달라" → 응답: apple, art  
                                    - "고성능 태블릿을 추천해달라" → 응답: apple, samsung, performance  
                                    - "가성비 필기용 패드를 추천해달라" → 응답: xiaomi, effectiveness
                                    - "게임용 스마트폰 추천해주세요" → 응답: game, samsung, xiaomi, apple
                                    
                              """
                              ,context= "samsung={}, apple={}, xiaomi={},effectiveness={},performance={}, study={}, game={}, art={}".format(self.keywords["samsung"], 
                            self.keywords["apple"], self.keywords["xiaomi"],self.keywords["effectiveness"],self.keywords["performance"],self.keywords["study"],self.keywords["game"],self.keywords["art"]),gptmodel='chatgpt-4o-latest')
        if os.path.exists(self.pickle_file):
            print("피클 파일에서 데이터 로드 중...")
            self.data = self.load_data_from_pickle(self.pickle_file)
        else:
            print("디렉토리에서 데이터 스캔 중...")
            self.data={}
            self.load_csv_in_folder()
            self.load_srt_in_folder()
            self.save_data_to_pickle(self.data, self.pickle_file)
        self.summary_list=None
        self.qa=None
        self.videometadata=[]
        self.videosummary=[]
        self.vectorstore_video = None
        self.retriever_video = None
        self.llm_video=None
        # 커스텀 템플릿 정의 (예시)
        self.custom_template = """
        당신은 RAG 시스템의 인덱싱/검색 담당 AI입니다.
        당신이 가진 자료는 "영상 자막"과 "영상 요약 정보"입니다.

        당신의 목표:
        - 아래 사용자 질문(Question)에 제시된 요구사항(제품군, 스펙, 가격대, 제조사 등)을 꼼꼼히 파악합니다.
        - 이 요구사항과 가장 잘 부합하는 영상(또는 클립)을, 아래 제공된 자막/요약 자료({context})에서 찾아서 안내합니다.

        [행동 규약]
        1. 사용자 질문에서 언급된 제품군(카테고리)에 맞는 영상만 우선 필터링합니다.
        2. 사용자 질문에서 언급된 스펙(성능, 기능 등)에 맞는 영상을 필터링합니다.
        3. 사용자 질문에서 언급된 가격대에 부합하는 영상을 필터링합니다.
        4. 사용자 질문에서 특정 제조사를 언급했다면, 그 제조사의 영상만 선택합니다.
        5. '주요 제조사 라인업 <-->' 자료를 참고하여, 적절히 매칭되는지 검토합니다.
        6. 질문에 대한 직접적인 해설이나 구매 조언을 하지 말고, 오직 관련 영상 정보만 찾아 제시하세요.

        [출력 방식]
        - 조건에 부합하는 영상이 여러 개인 경우, 가장 적합하다고 판단되는 상위 3개까지만 보여주세요.
        - 각 영상마다 다음 정보를 간단히 표기:
        1) 영상(클립) 식별 정보(제목, ID 등)
        2) (가능하면) 영상 URL
        3) 질문의 요구사항과 어떻게 연관되는지 1-2줄로 설명
        - 만약 관련 영상을 전혀 찾을 수 없으면, "해당 조건에 부합하는 영상이 없습니다."라고만 답변하세요.

        아래는 현재 사용할 수 있는 자료(키워드/자막/요약 등)입니다:
        {context}

        ---

        질문:
        {question}
        """
        # 프롬프트 템플릿 객체 생성
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.custom_template
        )
        # RetrievalQA 체인 생성 시 custom prompt 전달
        #qa_chain = RetrievalQA.from_chain_type(
        #    llm=llm, 
        #    chain_type="stuff", 
        #    retriever=retriever,
        #    chain_type_kwargs={"prompt": prompt},
        #    return_source_documents=True
        #)
    def get_keywords(self,query):
        response=self.keyword_llm.get_response("유저 요청사항 : " + query+"""
                                               \n\n 사장님 지시사항 : 그리고 절대 다른 말하지 말고 요구한 대로 키워드만 입력해 
                                               뒤의 AI 까지 가는 로직에서 치명적인 오류가 발생해 추가적인 필터 알고리즘이 없으니 알겟습니다 안녕하세요 이런 언급 금지야
                                               """)
        log_wrapper(f'GPT-4o-mini의 선택: {response}')
        normalized = response.lower().replace(" ", "")
        # 각 브랜드가 포함되어 있는지 검사
        contains_xiaomi = "xiaomi" in normalized
        contains_apple = "apple" in normalized
        contains_samsung = "samsung" in normalized
        contains_effectiveness = "effectiveness" in normalized
        contains_performance = "performance" in normalized
        contains_study = "study" in normalized
        contains_game = "game" in normalized
        contains_art = "art" in normalized
        # 각 브랜드에 해당하는 키워드 리스트를 totalkeywords에 추가
        self.totalkeywords=[]
        if contains_xiaomi:
            self.totalkeywords+=self.keywords["xiaomi"]
        if contains_apple:
            self.totalkeywords+=self.keywords["apple"]
        if contains_samsung:
            self.totalkeywords+=self.keywords["samsung"]
        if contains_effectiveness:
            self.totalkeywords+=self.keywords["effectiveness"]
        if contains_performance:
            self.totalkeywords+=self.keywords["performance"]
        if contains_study:
            self.totalkeywords+=self.keywords["study"]
        if contains_game:
            self.totalkeywords+=self.keywords["game"]
        if contains_art:
            self.totalkeywords+=self.keywords["art"]
        # 각 브랜드에 해당하는 키워드 리스트 반환
    def enhance_query(self,query):
        system_message=f"""
            roles:system
            당신은 벡터 검색(RAG) 시스템에 들어갈 자연어 쿼리를 개선하는 전문가입니다. 저희 RAG에는 영상의 자막과 설명이 저장되어 있습니다.
            원본 쿼리 :{query}
            해당 쿼리만으로는 RAG에 저장된 자막 자료로 원하는 영상을 제대로 찾지 못할것으로 예상됩니다.
            목표:
            쿼리의 목적을 보다 명확히 표현해야 합니다.또한 벡터 공간 상에서 더욱 강력한 방향성을 가져 검색이 용이하도록 해주세요.
            RAG가 실제로 ‘{query}’ 와 관련된 정보를 담은 영상을 잘 찾을 수 있도록 관련 키워드를 추가해 주세요.
            사용자가 궁극적으로 원하는 것은 “{query}와 관련된 특징을 요약하거나 인상적인 부분을 보여주는 클립 영상”임을 반영하세요.
            이후에는 context에 태그를 추가하는 llm도 있습니다 그에 도움이되게 제조사나 목적을 명확히 파악 할 수 있는 문구를 추가해주세요
        """
        user_message= f"""
            [원본 쿼리]
            {query}

            [실패 예측]
            - 쿼리가 너무 짧고 맥락이 부족하여, RAG가 원하는 영상을 제대로 찾지 못합니다.
            - 영상의 어떤부분을 원하는지 명확한 구체화가 필요합니다.

            [목표]
            1. 쿼리가 목표를 명확히 표현하되, 어떤 영상을 원하는지(기능 요약, 리뷰, etc.) 추가 설명
            2. RAG가 더 정확히 '{query}' 관련 클립을 찾도록, 적절한 키워드를 보강
            3. 사용자가 원하는 것은 '{query}'의 핵심적·인상적인 장면을 담은 영상임을 반영
            4. 원하는 장면을 구체적으로 머리에 그려지듯이 설명
            5. 확실한 제조사 정보와 정량적인 스펙을 제시
            
            위 사항을 고려하여, 개선된 자연어 쿼리를 작성해 주세요.
        """
        if isinstance(system_message, list):
            system_message = " ".join(map(str, system_message)).replace("}","").replace("{","")  # 리스트를 문자열로 변환
        if isinstance(user_message, list):
            user_message = " ".join(map(str, user_message)).replace("}","").replace("{","")  # 리스트를 문자열로 변환
        message=[("system",system_message+"{context}"),("user",user_message+"{input}")]
        rellm=Node("",gptmodel="chatgpt-4o-latest")
        rellm.change_raw_prompt(message)
        rellm.change_context("""
                            [스마트폰]
                            삼성전자
                            - 갤럭시 S 시리즈: 기본형 플래그십
                            - 갤럭시 S+ 시리즈: 대화면 프리미엄
                            - 갤럭시 S Ultra 시리즈: 최상급 카메라/S펜
                            - 갤럭시 A 시리즈: 중저가 실속형
                            - 갤럭시 Z Fold: 메인 폴더블
                            - 갤럭시 Z Flip: 컴팩트 폴더블

                            애플
                            - 아이폰 Pro Max: 최상급 플래그십
                            - 아이폰 Pro: 프리미엄 컴팩트
                            - 아이폰 기본형: 보급형 플래그십
                            - 아이폰 Plus: 대화면 보급형
                            - 아이폰 SE: 실속형 컴팩트

                            [노트북]
                            애플
                            - 맥북 Pro: 전문가용 고성능
                            - 맥북 Air: 휴대성 중심 일반용

                            삼성
                            - 갤럭시 북 Pro: 프리미엄 비즈니스
                            - 갤럭시 북: 일반 사무용
                            - 갤럭시 Book2/3 시리즈: 360도 회전형

                            LG
                            - 그램: 초경량 장시간 배터리
                            - 울트라PC: 일반 사무용
                            - 울트라기어: 게이밍 특화

                            [태블릿]
                            애플
                            - 아이패드 Pro: 전문가용 고성능
                            - 아이패드 Air: 중급형 범용
                            - 아이패드: 보급형 기본
                            - 아이패드 미니: 소형 휴대용

                            삼성
                            - 갤럭시 탭 S: 프리미엄 안드로이드
                            - 갤럭시 탭 A: 보급형 실속

                            [무선이어버드]
                            애플
                            - 에어팟 Pro: 프리미엄 ANC
                            - 에어팟: 기본형
                            - 에어팟 맥스: 오버이어 최상급

                            삼성
                            - 갤럭시 버즈 Pro: 프리미엄 ANC
                            - 갤럭시 버즈: 기본형
                            - 갤럭시 버즈 Live: 오픈형

                            소니
                            - WF 시리즈: 프리미엄 사운드
                            - WH 시리즈: 오버이어 프리미엄

                            [제품 라인별 공통 특징]
                            - Ultra/Pro/Max: 최고급 성능/기능 집중
                            - Air/Plus: 보급형 프리미엄
                            - 기본형: 핵심 기능 중심
                            - SE/A/Lite: 실속형 entry
                            [스마트폰]
                            샤오미/레드미
                            - 샤오미 시리즈: 플래그십
                            - 레드미 노트: 중급형 베스트셀러
                            - POCO: 성능특화 중저가
                            - 레드미: 보급형 실속

                            OPPO/원플러스
                            - 파인드 시리즈: 최상급 플래그십
                            - 레노 시리즈: 중상급 
                            - 원플러스: 성능특화 플래그십

                            구글
                            - 픽셀 Pro: 카메라특화 플래그십
                            - 픽셀: 준플래그십
                            - 픽셀 a: 중급형

                            [노트북]
                            레노버
                            - ThinkPad X1: 프리미엄 비즈니스
                            - ThinkPad T: 정통 비즈니스
                            - ThinkPad E: 보급형 비즈니스
                            - Legion: 게이밍 전문
                            - Yoga: 컨버터블 프리미엄
                            - IdeaPad: 일반 소비자용

                            HP
                            - Spectre: 프리미엄 컨버터블
                            - ENVY: 준프리미엄
                            - Pavilion: 일반 소비자용
                            - Omen: 게이밍 라인
                            - EliteBook: 비즈니스 프리미엄
                            - ProBook: 비즈니스 보급형

                            Dell
                            - XPS: 프리미엄 컨버터블
                            - Latitude: 비즈니스용
                            - Inspiron: 일반 소비자용
                            - Alienware: 고급 게이밍
                            - G시리즈: 보급형 게이밍

                            MSI
                            - Stealth: 슬림 게이밍
                            - Raider: 고성능 게이밍
                            - Creator: 크리에이터용
                            - Modern: 일반 사무용

                            [이어폰/헤드폰]
                            Bose
                            - QuietComfort: 프리미엄 노이즈캔슬링
                            - Sport: 운동특화
                            - SoundLink: 범용 무선

                            젠하이저
                            - Momentum: 프리미엄 사운드
                            - HD/HD Pro: 스튜디오용
                            - CX: 일반 소비자용

                            [스마트워치]
                            애플
                            - 워치 Ultra: 아웃도어/프로
                            - 워치: 일반형
                            - 워치 SE: 보급형

                            삼성
                            - 갤럭시 워치 프로: 프리미엄
                            - 갤럭시 워치: 일반형
                            - 갤럭시 핏: 피트니스 밴드

                            가민
                            - Fenix: 프리미엄 아웃도어
                            - Forerunner: 러닝특화
                            - Venu: 일반 스마트워치
                            - Instinct: 견고성 강화

                            [게이밍 모니터]
                            LG
                            - UltraGear: 게이밍 프리미엄
                            - UltraWide: 울트라와이드
                            - UltraFine: 전문가용

                            삼성
                            - 오디세이: 게이밍 프리미엄
                            - 뷰피트: 사무용
                            - 스마트 모니터: 올인원형

                            [공통 특성]
                            - Pro/Ultra/Premium: 최상급 라인
                            - Plus/Advanced: 업그레이드 모델
                            - Lite/SE/Neo: 실속형 라인
                            - Gaming/Creator: 용도특화 라인
                             """)
        developer_query="""
            개발자 요청 : 나는 개발자로써 첨언을 할게 유저들은 사용법을 잘 모르니까 좀더 쿼리를 구체화 해서 좋은 답변을 받을 수 있도록 도와줘 부탁할게 그리고 뒤에는 작은 모델들도 많으니 동작을 잘 할 수있도록 하는 너의 역활이 매우 중요하단다
        """
        return rellm.get_response(developer_query)
        
    def retry(self,sidelog,query,indump):
        #self.create_qa_chain_from_store()
        self.buffer=globalist
        self.buffer.append(sidelog)
        system_message=f"""
            roles:system
            당신은 벡터 검색(RAG) 시스템에 들어갈 자연어 쿼리를 개선하는 전문가입니다. 저희 RAG에는 영상의 자막과 설명이 저장되어 있습니다.
            원본 쿼리 :{query}
            해당 쿼리만으로는 RAG에 저장된 자막 자료로 원하는 영상을 제대로 찾지 못해 결과 추출에 실패했습니다.
            목표:
            쿼리가 “{query}”임을 보다 명확히 표현해야 합니다.
            RAG가 실제로 ‘{query}’ 와 관련된 정보를 담은 영상을 잘 찾을 수 있도록 관련 키워드를 추가해 주세요.
            사용자가 궁극적으로 원하는 것은 “{query}와 관련된 특징을 요약하거나 인상적인 부분을 보여주는 클립 영상”임을 반영하세요. 아래의 실패 로그를 참고하세요.
            실패로그 : {self.buffer}
        """
        user_message= f"""
            [원본 쿼리]
            {query}

            [실패 로그]
            {self.buffer}

            [실패 원인]
            - 쿼리가 너무 짧고 맥락이 부족하여, RAG가 원하는 영상을 제대로 찾지 못했음
            - '하이라이트 클립'이 무엇을 의미하는지 구체화가 부족함

            [목표]
            1. 쿼리가 '{query}'임을 명확히 표현하되, 어떤 영상을 원하는지(기능 요약, 리뷰, etc.) 추가 설명
            2. RAG가 더 정확히 '{query}' 관련 클립을 찾도록, 적절한 키워드를 보강
            3. 사용자가 원하는 것은 '{query}'의 핵심적·인상적인 장면을 담은 영상임을 반영

            위 사항을 고려하여, 개선된 자연어 쿼리를 작성해 주세요.
        """
        if isinstance(system_message, list):
            system_message = " ".join(map(str, system_message)).replace("}","").replace("{","")  # 리스트를 문자열로 변환
        if isinstance(user_message, list):
            user_message = " ".join(map(str, user_message)).replace("}","").replace("{","")  # 리스트를 문자열로 변환
        if isinstance(self.totalkeywords, list):
            self.totalkeywords = " ".join(map(str, self.totalkeywords)).replace("}","").replace("{","")  # 리스트를 문자열로 변환
        if isinstance(indump, list):
            indump = " ".join(map(str, indump)).replace("}","").replace("{","")  # 리스트를 문자열로 변환


        message=[("system",system_message+"{context}"),("user",user_message+"{input}")]
        rellm=Node("",gptmodel="chatgpt-4o-latest")
        rellm.change_raw_prompt(message)
        rellm.change_context(f"이전 쿼리를 기준으로 llm이 키워드라고 판단했던 단어들 (없는 경우도 존재함): {self.totalkeywords}, 덤프된 json 파일: {indump}")
        developer_query="""
            나는 개발자로써 첨언을 할게 유저들은 사용법을 잘 모르니까 좀더 쿼리를 구체화 해서 좋은 답변을 받을 수 있도록 도와줘 부탁할게
        """
        return rellm.get_response(developer_query)
           
        
    def load_data_from_pickle(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data

    def save_data_to_pickle(self, data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"데이터가 {filename} 파일로 저장되었습니다.")

    @staticmethod
    def load_ytref(ref_file):
        """
        YTref.txt 파일을 로드하여 각 줄을 리스트로 반환합니다.
        """
        try:
            with open(ref_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            return lines
        except Exception as e:
            return None    
        
        
    def load_youtube_folder(self):
        """
        youtube 폴더의 상위 디렉토리 목록(폴더 내 파일 및 폴더 이름)을 반환합니다.
        """
        try:
            if not os.path.isdir( self.target_dir):
                raise Exception(f"폴더 '{self.target_dir}'가 존재하지 않습니다.")
            contents = os.listdir( self.target_dir)
            return contents
        except Exception as e:
            return None
    def load_csv_in_folder(self):
        """
        youtube 폴더 내의 모든 파일을 재귀적으로 순회하면서,
        텍스트 파일(.csv, .srt)만 로드하고, 로드한 파일의 경로와 내용을 리스트로 반환합니다.
        """
        loaded_files = []
        for root, dirs, files in os.walk(self.target_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # .csv와 .srt 파일만 로드 (txt는 제외)
                if file.endswith((".csv")):
                    try:
                        content = pd.read_csv(file_path)
                        content = content.set_index("Unnamed: 0")
                        #with open(file_path, "r", encoding="utf-8") as f:
                        #    content_tocken = f.read()
                        #self.tocken_count+=count_tokens(content_tocken.replace("\n", " "))
                        content["설명"] = content["설명"].str.replace("\n", "", regex=False)
                        content["자막"]="0"
                        content["유튜버"]=root.split('/')[-2]
                        #print (root.split('/')[-2])
                        self.data[root.split('/')[-2]]=[file_path,content]
                    except Exception as e:
                        pass
                else:
                    pass
            
    def load_srt_in_folder(self):
        """
        youtube 폴더 내의 모든 파일을 재귀적으로 순회하면서,
        텍스트 파일(.csv, .srt)만 로드하고, 로드한 파일의 경로와 내용을 리스트로 반환합니다.
        """
        loaded_files = []
        for root, dirs, files in os.walk(self.target_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # .csv와 .srt 파일만 로드 (txt는 제외)
                if file.endswith((".srt")):
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        #self.tocken_count+=count_tokens(content)
                        self.data[root.split('/')[-1]][1].loc[int(file_path.split('/')[-1].split('.')[-2])-1,"자막"]=content    
                    except Exception as e:
                        pass
                else:
                    pass
    def create_summary_dicts(self):
        """
        self.data에 저장된 각 영상의 원본 DataFrame을 순회하여,
        요약본 딕셔너리 목록을 생성합니다.
        
        각 요약 딕셔너리는 다음과 같은 구조로 생성됩니다.
        {
            "metadata": { 
                "유튜버": "잇섭",
                "제목": "영상 제목",
                "조회수": "218K views",
                "업로드일": "1 day ago",
                "링크": "https://www.youtube.com/..."
            },
            "embedding_text": "영상 설명과 자막 내용을 합친 텍스트 (최대 1000자)",
            "요약": "영상 설명과 자막 내용을 합친 텍스트 (최대 1000자)"
        }
        """
        summary_list = []
        n=0
        tottoken=0
        for channel, (csv_path, df) in self.data.items():
            for idx, row in df.iterrows():
                # 메타데이터는 별도로 저장 (단어 단위이므로 임베딩에 큰 영향을 주지 않음)

                metadata = {
                    "인덱스": idx,
                    "유튜버": channel,
                }
                # 임베딩에 사용할 텍스트: 설명과 자막을 결합
                description = row.get("설명", "")
                subtitle_text = row.get("자막", "")
                # 자막이 길 경우, 청크 분할을 통해 압축

                
                buff=subtitle_text.replace("\n","")
                buff2=re.sub(r'[-:\d>]', '', buff)
                buff3=buff2.replace(" ,"," ").replace(", ","")
                cunck_text = f"{description} {buff3}".strip()
                
                

                
                #target=1400
                #chunck_overrap=int(0)
                #roop_con, chunck_overrap=setting_tockens(cunck_text,target=target)
                #
                #while token_bool("".join(roop_con)):
                #    roop_con, chunck_overrap=setting_tockens(cunck_text,target=target)
                #    if token_bool("".join(roop_con)):
                #        break
                #    target=target-10

                roop_con=compress_text(cunck_text, chunk_size=500, chunk_overlap=50)
                embedding_text = "".join(roop_con)
                metadata['embedding_text']=embedding_text

                metadata['token']=cal_token(embedding_text, model="gpt-4o-mini")
                n+=1
                tottoken+=metadata['token']
                avg=tottoken/n
                #print(f"총 토큰수 {tottoken},평균 토큰수:{avg}")
                #metadata['chunck_overrap']=chunck_overrap
                summary = {
                    "metadata": metadata,
                    "요약": embedding_text  # 추후 필요 시 표시용 요약문으로 사용
                }
                summary_list.append(summary)
                # 루프에 걸린 시간 측정 및 남은 시간이 있으면 sleep
        self.summary_list = summary_list
    def get_combined_context(self, query,custom_context):
        retrieved_docs = self.retriever.invoke(query)
        retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])
        return custom_context + "\n" + retrieved_text
            
        
    def create_vector_store(self, persist_directory="chroma_db"):
        """
        생성된 요약 딕셔너리 목록을 기반으로 벡터스토어(Chroma)를 생성하고 저장합니다.
        각 요약 딕셔너리에서 "요약"은 문서 본문, 나머지는 메타데이터로 사용합니다.
        """
        if self.summary_list is None:
            raise Exception("요약 목록이 생성되지 않았습니다. 먼저 create_summary_dicts()를 실행하세요.")
        mininal_chunck=self.summary_list[0]
        metadata=mininal_chunck["metadata"]
        #metadata =simple_filter( {k: v for k, v in mininal_chunck.items() if k != "요약"})
        docs = []
        docs.append(Document(page_content=mininal_chunck["요약"], metadata=metadata))
        embeddings = OpenAIEmbeddings() 
        vectorstore=Chroma.from_documents(docs, embeddings, persist_directory=persist_directory) 
        removed=self.summary_list.pop(0)
        chunks_size=100
        total_chunks = (len(self.summary_list) + chunks_size-1) // chunks_size   
        nowtoken=0    
        for i in range(total_chunks):
            start_time = time.time()
            try:
                chunck=self.summary_list[i*chunks_size:(i+1)*chunks_size]
            except:
                chunck=self.summary_list[i*chunks_size:]
            docs = []
            for summary in chunck:
                #metadata =simple_filter( {k: v for k, v in summary.items() if k != "요약"})
                metadata = summary["metadata"]
                docs.append(Document(page_content=summary["요약"], metadata=metadata))
                #doc=Document(page_content=summary["요약"], metadata=metadata)
                nowtoken+=metadata['token']
                print(f"write mwtadata:{metadata['인덱스']},{metadata['유튜버']},토큰 합!!{nowtoken}")
                #vectorstore.add_documents([doc])
            vectorstore.add_documents(docs)
            #for doc in docs:
            #    vectorstore.add_texts([doc.page_content], metadatas=[doc.metadata])
            elapsed_time = time.time() - start_time
            remaining_time = 60 - elapsed_time
            if remaining_time > 0 and i<total_chunks-1:
                print(f"Chunk {i+1}/{total_chunks} 완료, {elapsed_time:.2f}초 걸림. {remaining_time:.2f}초 대기합니다.")
                time.sleep(remaining_time)
            else:
                print(f"Chunk {i+1}/{total_chunks} 완료, {elapsed_time:.2f}초 걸림. 대기 시간 없이 다음 청크 진행합니다.")
        self.summary_list.insert(0,removed)      
        return vectorstore
    def create_vector_store_active(self, persist_directory="chroma_db"):
        """
        생성된 요약 딕셔너리 목록을 기반으로 벡터스토어(Chroma)를 생성하고 저장합니다.
        각 요약 딕셔너리에서 "요약"은 문서 본문, 나머지는 메타데이터로 사용합니다.
        """
        if self.summary_list is None:
            raise Exception("요약 목록이 생성되지 않았습니다. 먼저 create_summary_dicts()를 실행하세요.")
        mininal_chunck=self.summary_list[0]
        metadata=mininal_chunck["metadata"]
        #metadata =simple_filter( {k: v for k, v in mininal_chunck.items() if k != "요약"})
        docs = []
        docs.append(Document(page_content=mininal_chunck["요약"], metadata=metadata))
        embeddings = OpenAIEmbeddings() 
        vectorstore=Chroma.from_documents(docs, embeddings, persist_directory=persist_directory) 
        removed=self.summary_list.pop(0)
        chunks_size=100
        total_chunks = (len(self.summary_list) + chunks_size-1) // chunks_size   
        nowtoken=0 
        i=0   
        for summary in self.summary_list:
            if summary:
                #metadata =simple_filter( {k: v for k, v in summary.items() if k != "요약"})
                metadata = summary["metadata"]
                docs.append(Document(page_content=summary["요약"], metadata=metadata))
                #doc=Document(page_content=summary["요약"], metadata=metadata)
                nowtoken+=metadata['token']
                print(f"write mwtadata:{metadata['인덱스']},{metadata['유튜버']},토큰 합!!{nowtoken}")
                #vectorstore.add_documents([doc])
        lenthD=len(docs)
        lenthO=len(docs)


        
        while len(docs)>0:
            nowtoken=0
            in_doc=[]
            while nowtoken<500000:
                if len(docs)==0:
                    break
                in_doc.append(docs.pop(0))
                nowtoken+=in_doc[-1].metadata['token']
                lenthD-=1
                print(f"남은 문서수:{lenthD},남은 청크 토큰수:{500000-nowtoken}")
            start_time = time.time()
            vectorstore.add_documents(in_doc)
            #for doc in docs:
            #    vectorstore.add_texts([doc.page_content], metadatas=[doc.metadata])
            elapsed_time = time.time() - start_time
            remaining_time = 60 - elapsed_time
            if remaining_time > 0 and lenthD>1:
                print(f"Chunk {lenthD}/{lenthO} 완료, {elapsed_time:.2f}초 걸림. {remaining_time:.2f}초 대기합니다.")
                time.sleep(remaining_time)
            else:
                print(f"Chunk {lenthD}/{lenthO} 완료, {elapsed_time:.2f}초 걸림. 대기 시간 없이 다음 청크 진행합니다.")
            self.summary_list.insert(0,removed)      
        return vectorstore    

    def create_video_vector_store(self):
        """
        생성된 요약 딕셔너리 목록을 기반으로 벡터스토어(Chroma)를 생성하고 저장합니다.
        각 요약 딕셔너리에서 "요약"은 문서 본문, 나머지는 메타데이터로 사용합니다.
        """
        if self.videometadata is None:
            raise Exception("요약 목록이 생성되지 않았습니다. 먼저 self.videosummary를 생성 하세요.")
        mininal_chunck=self.videometadata[0].to_dict()
        metadata=self.videometadata[0].to_dict()
        buff=str(mininal_chunck["자막"]).replace("\n","")
        buff2=re.sub(r'[-:\d>]', '', buff)
        buff3=buff2.replace(" ,"," ").replace(", ","")
        cunck_text = f"{str(mininal_chunck['설명']).replace("\n","")} {buff3}".strip()
        roop_con=compress_text(cunck_text, chunk_size=500, chunk_overlap=200)
        metadata['token']=cal_token("".join(roop_con), model="gpt-4o-mini")
        docs = []
        docs.append(Document(page_content="".join(roop_con), metadata=metadata))
        embeddings = OpenAIEmbeddings() 
        vectorstore=Chroma.from_documents(docs, embeddings) 
        removed=self.videometadata.pop(0)
        docs = []
        for metadata in self.videometadata:
            metadata=metadata.to_dict()
            buff=str(metadata["자막"]).replace("\n","")
            buff2=re.sub(r'[-:\d>]', '', buff)
            buff3=buff2.replace(" ,"," ").replace(", ","")
            cunck_text = f"{str(metadata['설명']).replace("\n","")} {buff3}".strip()
            roop_con=compress_text(cunck_text, chunk_size=500, chunk_overlap=50)
            metadata['token']=cal_token("".join(roop_con), model="gpt-4o-mini")
            docs.append(Document(page_content="".join(roop_con), metadata=metadata))
            #print(f"write mwtadata:{metadata["제목"]}")
        lenthD=len(docs)
        lenthO=len(docs)
        while len(docs)>0:
            nowtoken=0
            in_doc=[]
            while nowtoken<500000:
                if len(docs)==0:
                    break
                in_doc.append(docs.pop(0))
                nowtoken+=in_doc[-1].metadata['token']
                lenthD-=1
                print(f"남은 문서수:{lenthD},남은 청크 토큰수:{500000-nowtoken}")
            start_time = time.time()
            vectorstore.add_documents(in_doc)
            #for doc in docs:
            #    vectorstore.add_texts([doc.page_content], metadatas=[doc.metadata])
            elapsed_time = time.time() - start_time
            remaining_time = 60 - elapsed_time
            if remaining_time > 0 and lenthD>0:
                print(f"Chunk {lenthD}/{lenthO} 완료, {elapsed_time:.2f}초 걸림. {remaining_time:.2f}초 대기합니다.")
                time.sleep(remaining_time)
            else:
                print(f"Chunk {lenthD}/{lenthO} 완료, {elapsed_time:.2f}초 걸림. 대기 시간 없이 다음 청크 진행합니다.")
        self.videometadata.insert(0,removed)   
        return vectorstore
    
    def create_qa_video_chain_from_llm(self, model_name="gpt-4o-mini", temperature=0):
        """
        생성된 벡터스토어를 기반으로 RetrievalQA 체인을 생성합니다.
        LLM 호출 시 model_name과 temperature를 지정할 수 있습니다.
        """
        load_dotenv()
        self.vectorstore_video = self.create_video_vector_store()
        self.retriever_video = self.vectorstore_video.as_retriever(search_type="mmr", search_kwargs={'k': 1})
        self.llm_video = OpenAI(model_name=model_name, temperature=temperature)
        qa = RetrievalQA.from_chain_type(llm=self.llm_video, chain_type="stuff", retriever=self.retriever_video,chain_type_kwargs={"prompt": self.prompt},return_source_documents=True)  # 이 옵션 추가!)
        self.qa_video=qa
    
    def create_qa_chain_from_llm(self, model_name="gpt-4o-mini", temperature=0, persist_directory="chroma_db"):
        """
        생성된 벡터스토어를 기반으로 RetrievalQA 체인을 생성합니다.
        LLM 호출 시 model_name과 temperature를 지정할 수 있습니다.
        """
        model_name="chatgpt-4o-latest"
        load_dotenv()
        #self.vectorstore = self.create_vector_store(persist_directory=persist_directory)
        self.vectorstore = self.create_vector_store_active(persist_directory=persist_directory)
        self.retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': self.k_value})
        self.llm = OpenAI(model_name=model_name, temperature=temperature)
        qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever,chain_type_kwargs={"prompt": self.prompt},return_source_documents=True)  # 이 옵션 추가!)
        self.qa=qa
    def create_qa_chain_from_store(self,model_name="gpt-4o-mini", temperature=0, persist_directory="chroma_db"):
        load_dotenv()
        self.vectorstore = self.load_vector_store(persist_directory=persist_directory)
        self.retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': self.k_value})
        self.llm = OpenAI(model_name=model_name, temperature=temperature)
        qa = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff", 
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True  # 선택 사항: 소스 문서까지 반환
        )
        self.qa=qa
    def get_video_data(self,query,mode="database"):
        if mode=="video":
            self.create_qa_video_chain_from_llm(model_name="gpt-4o-mini", temperature=0)


        self.get_keywords(query)
        custom_context="핵심 kewwords:{}".format(self.totalkeywords)+"""
        주요 제조사 라인업<IOS
        [애플 {
            스마트폰:
                아이폰 프로 시리즈: 최상위 플래그십(iPhone 15 Pro, 15 Pro Max)
                아이폰 기본 시리즈: 준프리미엄(iPhone 15, 15 Plus)
                아이폰 SE: 실용적인 보급형 모델(iPhone SE 3세대)
            태블릿:
                아이패드 프로: 최고사양 프로용 태블릿(12.9인치, 11인치)
                아이패드 에어: 준프리미엄 태블릿
                아이패드: 기본형 태블릿
                아이패드 미니: 소형 태블릿
            노트북:
                맥북 프로: 전문가용 고성능(14인치, 16인치, M3/M3 Pro/M3 Max)
                맥북 에어: 일반용 슬림(13인치, 15인치, M2/M3)
                맥 미니: 데스크톱 미니PC
                맥 스튜디오: 전문가용 고성능 데스크톱
                맥 프로: 최상위 워크스테이션
            모니터:
                프로 디스플레이 XDR: 최고급 전문가용 모니터
                스튜디오 디스플레이: 준프리미엄 모니터
            웨어러블:
                애플워치: 스마트워치(Series 9, Ultra 2, SE 2세대)
                에어팟: 무선이어폰(AirPods Pro 2, AirPods 3, AirPods 2)
                에어팟 맥스: 오버이어 헤드폰
                비전 프로: 혼합현실 헤드셋 (2024년 출시)
            }]
        안드로이드    
        [삼성{ 
            스마트폰:
                갤럭시 S 시리즈: 최상위 플래그십 라인(S24, S24+, S24 Ultra)
                갤럭시 Z 시리즈: 폴더블 스마트폰(Z Fold5, Z Flip5)
                갤럭시 A 시리즈: 중저가 라인(A54, A34 등)
                갤럭시 M 시리즈: 실용적인 가성비 라인(M34, M14 등)
            태블릿:
                갤럭시 탭 S 시리즈: 프리미엄 태블릿(Tab S9, S9+, S9 Ultra)
                갤럭시 탭 A 시리즈: 중저가 태블릿(Tab A9, A8 등)
                갤럭시 탭 Active: 견고성 강화 비즈니스용 태블릿
            노트북:
                갤럭시 북4 시리즈: 프리미엄 노트북(Book4 Pro, Book4 Pro 360)
                갤럭시 북3 시리즈: 일반 사무용/학생용 노트북
                갤럭시 Book2 Business: 비즈니스용 노트북
            모니터:
                오디세이 시리즈: 게이밍 모니터(G9, G7, G5 등)
                뷰피니티 시리즈: 전문가용 고해상도 모니터
                스마트 모니터: 일체형 스마트 디스플레이
            웨어러블:
                갤럭시 워치: 스마트워치(Watch6, Watch6 Classic)
                갤럭시 버즈: 무선이어폰(Buds3, Buds3 Pro)
                갤럭시 링: 스마트 반지(신제품)
            }
        샤오미{
            스마트폰:
                샤오미 시리즈: 플래그십 라인(Xiaomi 14, 14 Pro, 14 Ultra)
                레드미 노트 시리즈: 중급형(Redmi Note 13 Pro+, Note 13 Pro, Note 13)
                레드미 시리즈: 보급형(Redmi 13C, 12C 등)
                POCO 시리즈: 성능특화 중저가(POCO F5, X5, M5 등)
            태블릿:
                샤오미 패드: 프리미엄 태블릿(Pad 6, Pad 6 Pro)
                레드미 패드: 보급형 태블릿(Redmi Pad SE)
            노트북:
                샤오미북: 프리미엄 노트북(RedmiBook Pro, Mi Notebook Pro)
                레드미북: 일반 사무용/학생용 노트북(RedmiBook 15)
            모니터:
                Mi 모니터: 일반용 모니터
                Mi 게이밍 모니터: 게이밍용 모니터
                Mi 커브드 모니터: 커브드 디스플레이
            웨어러블:
                샤오미 워치: 스마트워치(Watch S3, Smart Band 8)
                레드미 워치: 보급형 스마트워치(Redmi Watch 3)
                샤오미 버즈: 무선이어폰(Buds 4 Pro, Buds 4)
                레드미 버즈: 보급형 무선이어폰(Redmi Buds 4)
            }]>    
        """
        combined_context = self.get_combined_context(query,custom_context)
        if token_bool(combined_context, model="gpt-4o-mini",target=120000):
            combined_context=setting_tockens(combined_context,target=115000,model="gpt-4o-mini",chunk_size=500)
            combined_context="".join(combined_context)
        if mode=="video":
            answer = self.qa_video.invoke({"context": combined_context, "question": query,"query": query})
            return answer["source_documents"][0].metadata
        elif mode=="database":
            answer = self.qa.invoke({"context": combined_context, "question": query,"query": query})
            keys=[]
            for doc in answer["source_documents"]:
            # metadata에 포함된 유튜버, 제목, 링크 등의 정보를 출력합니다.
                keys.append(doc.metadata)
            self.videometadata=[]
            self.videosummary=[]
            for key in keys:
                ap,embedding_text=self.get_original_row(key)
                self.videometadata.append(ap)
                self.videosummary.append(embedding_text)
            return self.videometadata
        else:
            raise Exception("mode는 video 또는 database 중 하나여야 합니다.")

    @staticmethod
    def load_vector_store(persist_directory="chroma_db"):
        embeddings = OpenAIEmbeddings()  # API 키가 설정되어 있어야 합니다.
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectorstore
    
    def get_original_row(self, metadata):
        """
        벡터스토어에서 반환된 메타데이터(문자열 또는 dict)를 기반으로 원본 DataFrame에서 해당 행을 찾습니다.
        
        :param metadata: {"인덱스": idx, "유튜버": channel} 형태의 메타데이터 (문자열일 경우 JSON으로 변환)
        :return: 해당 행 (pandas Series) 또는 None (찾을 수 없을 경우)
        """
        # metadata가 문자열이면 dict로 변환합니다.
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError as e:
                print(f"메타데이터 변환 실패: {e}")
                return None, None

        channel = metadata.get("유튜버")
        idx = metadata.get("인덱스")
        embedding_text = metadata.get("embedding_text")
        if channel in self.data:
            _, df = self.data[channel]
            try:
                row = df.loc[idx].copy()
                row.loc["인덱스"]=idx
                return row, embedding_text
            except KeyError:
                print(f"인덱스 {idx}가 채널 {channel}의 DataFrame에 없습니다.")
                return None, None
        else:
            print(f"채널 {channel}이 존재하지 않습니다.")
            return None, None



if __name__ == "__main__":
    # 환경변수(.env 파일) 로드: OPENAI_API_KEY 등이 설정되어 있어야 합니다.
    load_dotenv()
    query="초 고성능 앤드스펙 태블릿좀 추천해줘 아이패드 계열로"
    search_instance = Dataprocessor()
    search_instance.create_summary_dicts()
    search_instance.create_qa_chain_from_store(model_name="gpt-4o-mini", temperature=0)
    #search_instance.create_qa_chain_from_llm(model_name="gpt-4o-mini", temperature=0)
    result=search_instance.get_video_data(query)
    print(f'{query}')
    for res in result:
        print(f'{res}')
        print('----------------------')
        
    result2=search_instance.get_video_data(query,mode="video")
    print(f"{result2['제목']}")
    print(f"{result2['링크']}")
    print(f"{result2['인덱스']}")
    print(f"{result2['유튜버']}")
    print(f"{result2['조회수']}")
    
    
