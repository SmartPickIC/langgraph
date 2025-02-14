import os
import getpass
import warnings
import requests
from openai import OpenAI
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pprint import pprint
warnings.filterwarnings("ignore")
class UPstageAPUcontroller:
    def __init__(self):
        load_dotenv()
        self.UPSTAGE_API_KEY = os.environ.get("UPSTAGE_API_KEY")
        self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

        if not self.UPSTAGE_API_KEY:
            self.UPSTAGE_API_KEY = None
            print("Upstage API is None")
        else:
            
            print("Upstage API key is successfully set.")
        if not self.OPENAI_API_KEY:
            self.OPENAI_API_KEY = None
            print("OpenAi API is None")
        else:
            print("OpenAi API key is successfully set.")
    
    def get_upstage(self,model='openai'):
        if model == 'solar-pro' and self.UPSTAGE_API_KEY:
            if self.UPSTAGE_API_KEY:
                llm = ChatUpstage(model='solar-pro')
                return llm
            else:
                return None
            
        elif model == 'openai' and self.OPENAI_API_KEY:
            if self.OPENAI_API_KEY:
                client = OpenAI()
                #llm = client.chat.completions.create(  model="gpt-4o-mini", messages=["{context}"])
                return llm
            else:
                return None
        else:
            print("Not avilable any model")
            return None
    def get_prompt(self,input_prompt) -> ChatPromptTemplate:
        """
        시스템과 사용자 메시지를 포함한 ChatPromptTemplate을 생성하여 반환합니다.
        
        시스템 메시지에는 {context} 플레이스홀더가 있고,
        인간 메시지에는 {input} 플레이스홀더가 존재합니다.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",input_prompt+" {context}"),("human", "{input}"),
            ]
        )
        return prompt 
    def get_answer(self, llm:ChatUpstage ,prompt:ChatPromptTemplate ,query: str,context="") -> str:
        """
        주어진 query(사용자 질문)와 context(문맥 또는 문서 내용)를 바탕으로
        최종 답변을 반환하는 함수입니다.
        
        내부적으로 get_prompt()와 get_llm()를 사용하여 체인을 구성한 후, 
        StrOutputParser()를 통해 문자열 형태의 응답을 추출합니다.
        """

        # 3. 체인 구성: 프롬프트 → LLM → 출력 파서
        chain = prompt | llm | StrOutputParser()
        
        # 4. 체인 실행: 플레이스홀더에 실제 값 전달
        response = chain.invoke({'context': context, 'input': query})
        return response
    def Node_generation(self,input_prompt,context=""):
        llm = self.get_upstage()
        prompt = self.get_prompt(input_prompt)
        return Node(prompt,llm,context,self)


solar_pro=UPstageAPUcontroller()

class Node:
    def __init__(self, prompt,model='solar-pro',context=""):
        if model == 'solar-pro':
              self.controller = solar_pro
        self.llm = self.controller.get_upstage(model)
        self.prompt = self.controller.get_prompt(prompt)
        self.context = context

    def get_response(self,query):
        return self.controller.get_answer(self.llm,self.prompt,query,self.context)
    def change_context(self,context):
        self.context = context
    def change_prompt(self,prompt):
        self.prompt = self.controller.get_prompt(prompt)
    def change_llm(self,llm):
        self.llm = llm
    def get_prompt(self):
        return self.prompt
    def get_llm(self):
        return self.llm
    def get_context(self):
        return self.context
    def get_controller(self):
        return self.controller