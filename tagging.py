
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
    def load_data_from_pickle(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        return data

    def save_data_to_pickle(self, data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"데이터가 {filename} 파일로 저장되었습니다.")
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