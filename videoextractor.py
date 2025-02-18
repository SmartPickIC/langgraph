from utility import Node,Link,iterator
from pprint import pprint
from dataloader import Dataprocessor
from dotenv import load_dotenv
import re
import srt
import pandas as pd
from queue_manager import add_log
import json
loglist=[]
def get_srt_text(srt_filename):
    with open(srt_filename, "r", encoding="utf-8") as file:
        srt_content = file.read()
    # srt.parse()를 사용하여 SRT 데이터를 파싱합니다.
    subtitles = list(srt.parse(srt_content))
    combined_text = "\n\n".join([
        f"{subtitle.start} --> {subtitle.end}\n{subtitle.content}"
        for subtitle in subtitles
    ])
    return combined_text
def log_wrapper(log_message):
    loglist.append(log_message)
    print(log_message)
    add_log(log_message)  
def run_response(code_response):
    match = re.search(r"python\n(.*?)\n", code_response, re.DOTALL)
    if match:
        code_str = match.group(1)
    else:
        code_str = code_response  # 코드 블록이 없으면 전체 응답 사용

    print("실행할 코드:")
    print(code_str)

    # 주의: exec()는 신뢰할 수 있는 코드에 한해서만 사용하세요!
    exec(code_str)
    return "Success"
def feedback_context():

    srt_filename = "test_set/4.srt"
    input = get_srt_text(srt_filename)
    initial_prompt = "각 글자별로 datetime.timedelta 포멧으로 milisecond 단위로 딕셔너리 포멧으로 반환해줘 바로 코드로 쓸거니까 다른말이 섞여선 안되"
    feedback_prompt = "너는 이 결과를 보고 더 좋은 결과를 얻기위해 프롬프트를 수정해야해"
    slave_node = Node(initial_prompt)
    master_node = Node(feedback_prompt)
    last_prompt = None
    code_sniptet = """
    loop_con=True
    Maxtry=50
    trynum=0
    while loop_con:
        if last_prompt:
            slave_node.change_prompt(last_prompt)
        response = slave_node.get_response(input)
        try:
            out =run_response(response)
        except Exception as e:
            out= e
        combined_context = (
            f"initial_prompt:\n{initial_prompt}\n\n"
            f"last_prompt:\n{last_prompt}\n\n"
            f"result:\n{out}\n\n"
            f"raw_input:\n{input}"
        )
        master_node.change_context(combined_context)
        last_prompt = master_node.get_response(response+" 이게 결과물인데 확인하고 프롬프트를 수정해줘 만약 충분하다면 그냥 아무말 없이 False를 반환해서 코드 스니펫을 끝내줘")
        if last_prompt==False:
            loop_con=False
        trynum+=1
        if trynum>Maxtry:
            loop_con=False
    """
    loop_con=True
    Maxtry=50
    trynum=0
    while loop_con:
        if last_prompt:
            slave_node.change_prompt(last_prompt)
        response = slave_node.get_response(input)
        try:
            out =run_response(response)
        except Exception as e:
            out= e
        combined_context = (
            f"initial_prompt:\n{initial_prompt}\n\n"
            f"last_prompt:\n{last_prompt}\n\n"
            f"result:\n{out}\n\n"
            f"code sniptet:\n{code_sniptet}\n\n"
            f"raw_input:\n{input}"
        )
        master_node.change_context(combined_context)
        last_prompt = master_node.get_response(response+" 이게 결과물인데 확인하고 프롬프트를 수정해줘 만약 충분하다면 그냥 아무말 없이 False를 반환해서 코드 스니펫을 끝내줘")
        if last_prompt==False:
            loop_con=False
        trynum+=1
        if trynum>Maxtry:
            loop_con=False
def select_one_viedo(video_list):
    
    
    select_prompt = "5개의 영상 정보 중 요청자의 질문과 가장 관련있는 하나를 선택해줘"
    select_node=Node(select_prompt)
    video_list[0]
    

def extract_yt_str(user_query_abstract,youtuber_name,input_srt):
    user_query_abstract.replace("\n","").replace("개선된 쿼리"," ")
    abckupresopons={}
    
    buff=input_srt.replace("\n","")
    buff2=re.sub(r'[-:\d>]', '', buff)
    input_txt=buff2.replace(" ,"," ").replace(", ","")
    ueer_query_prompt = """
                            너의 역할: 
                            - 유저 요청에서 **핵심 조건**을 뽑아내고, 이를 LLM이 이해하기 좋은 형태로 요약/정리하는 것.

                            작업 지시:
                            - 유저 요청(abstract)을 분석하고, 그 안에 담긴 요구사항(조건)들을 우선순위대로 정리한다.
                            - 최종 출력은 **우선순위대로 번호를 붙여** 상세 설명 형태로 작성한다. 
                            - 중간 단계(조건 개수 파악, 우선순위 결정 등)는 절대 출력하지 말 것.
                            - 오직 “우선순위별 조건 설명”만 출력하라.
                      
                        """
    user_query_extractor = Node(ueer_query_prompt)
    user_query_extractor_response = user_query_extractor.get_response(user_query_abstract)
    abckupresopons["1차"]=user_query_extractor_response
    log_wrapper(user_query_extractor_response)
    slave_prompt = """
                        너의 역할:
                        - 제공된 '리뷰 스크립트'에서, 사용자(구매 고려자)가 중요하게 여길 만한 문구(중요사항과 관련)에 주목하고, 
                        해당 문구가 몇 번째 라인에 있는지 찾는다.
                        - 최종적으로 다음 세 가지를 **한 번에** 출력하라:
                        1) 원문(스크립트 그대로),
                        2) 몇 번째 라인인지,
                        3) 이를 표준어로 교정한 문장.

                        주의 사항:
                        - 리뷰 스크립트는 실제 발화 내용 그대로라서 문법이 다소 불규칙할 수 있음. 
                        - 중간 과정(1번, 2번 단계 탐색)은 절대 출력하지 말고, 최종 결과만 출력할 것.
                        - 만약 중요사항과 직접 관련 없는 문구라면, 답변하지 말고 다시 스크립트를 확인해(내부적으로) 
                        ‘직접 관련이 있는 문구’만 최종 결과에 포함하라.
                    """
    slave_prompt_srt = """
                            너의 역할:
                            - 자막 스크립트(시간 정보 포함)를 바탕으로, '요약 정보(중요사항)'와 가장 밀접하게 연결된 장면을 찾는다.
                            - 찾아낸 장면의 시작~끝 시간을 결정하고, 다음 포맷으로 한 번에 출력한다.

                            출력 형식(4줄 고정):
                            1) "사유 내용"
                            2) [;HH:MM:SS,HH:MM:SS;]
                            3) [NNNs]
                            4) /?;'원문 내용';?/
                            세부 규칙:
                            - 1번 줄: 큰따옴표("") 안에 들어갈 내용(사유).
                            - 2번 줄: 시작~끝 시간, 예: `[;00:01:23,00:01:45;]`. (밀리초가 필요하면 `00:01:23.456`처럼 추가 가능)
                            - 3번 줄: 시작 시간을 초단위로 환산 + 's' 붙인 형태 예: `[83s]`, `[135s]` 등.
                            - 4번 줄: /?;'   ';?/ 사이에 원문 내용. 예: `/?;'이 제품은 가성비가 뛰어나요';?/`.
                            주의 사항:
                            - 절대 이 4줄 이외의 내용(문장, 기호, 해설)을 추가 출력하지 마라.
                            - 단 하나의 시간대(클립)만 결정해야 한다.
                            - 만약 ‘중요사항’과 정확히 연결된 구간이 없다고 판단되면, 답변을 하지 말거나 “연관 구간이 없습니다.”라고만 출력.
                            - 중간 단계(0,1,2,3 등)는 절대 출력하지 말고, 요구된 최종 포맷만 반환할 것.
                        """    
    context = {f"중요사항:\n{user_query_extractor_response}"}
    slave_node = Node(slave_prompt)
    slave_node.change_context(context)
    slave_node_srt = Node(slave_prompt_srt,gptmodel='chatgpt-4o-latest')
    respons=slave_node.get_response(input_txt)
    log_wrapper(respons)
    abckupresopons["2차"]=respons
    context_srt = {f"자막:\n{input_srt}\n\n"\
                    f"요약:\n{respons}\n\n"\
                    f"유튜버:\n{youtuber_name}"}
    slave_node_srt.change_context(context_srt)
    respons_srt=slave_node_srt.get_response(user_query_extractor_response)
    abckupresopons["3차"]=respons_srt
    #print(respons_srt)
    return respons_srt,abckupresopons


def getoutput(text,base_link):

    # 1. 원문: 큰따옴표("") 사이의 내용
    original_match = re.search(r'"(.*?)"', text, re.DOTALL)
    original_text = original_match.group(1) if original_match else ""

    # 2. 시간: [; ;] 사이의 내용
    time_match = re.search(r'\[\;(.*?)\;\]', text, re.DOTALL)
    time_text = time_match.group(1) if time_match else ""

    # 3. 링크: 단, [] 안에 있지만 시간 형식([; ;])은 제외 (여기서는 간단히 [로 시작하고 ;가 없는 경우)
    link_match = re.search(r'\[(?!;)([^\[\]]*?)\]', text)
    link_text = link_match.group(1) if link_match else ""
    if link_text:
        link_text_f=re.sub(r'[^0-9s]', '', link_text)
    else:
        link_text_f=""
    # 4. 사유: /??/ 사이의 내용  
    # 예시에서는 "/?" 로 시작하고 "/"로 끝남. 내부의 ?는 고정이 아니라 필요에 따라 변동 가능하므로
    reason_match = re.search(r"/\?;'(.*?)';\?/", text, re.DOTALL)
    reason_text = reason_match.group(1) if reason_match else ""
    # 결과를 딕셔너리로 저장
    result = {
        "사유": original_text,
        "시간": time_text,
        "링크": base_link+"&t="+link_text_f,
        "원문": reason_text
    }
    if not link_text_f:
        retry=True
    elif len(link_text_f)<1:
        retry=True
    else:
        retry=False
    return result, retry
class survice:
    def __init__(self):
        self.source=None
        self.initial_context()

    def initial_context(self):
        load_dotenv()
        #user_query_abstract = "나는 ios만 10년을 써왔어 그런데 이번엔 갤럭시를 써보려고 하는데 적당히 카메라 성능 잘 나오고 괜찮은 갤럭시 모델 있을까?"
        self.source=Dataprocessor()
        self.source.create_summary_dicts()
        self.source.create_qa_chain_from_store(model_name="gpt-4o-mini", temperature=0)
        

    def extract_from_query(self,user_query_raw="그림 그리기 좋은 ios계열 패드 추천좀"):
        user_query_abstract=self.source.enhance_query(user_query_raw)
        log_wrapper(f"개선된 요청자 질문: {user_query_abstract}")
        result=self.source.get_video_data(user_query_abstract)
        for inputV in result:
            log_wrapper(f"{inputV['제목']}")
            log_wrapper(f"{inputV['유튜버']}")
        result2=self.source.get_video_data(user_query_abstract,mode="video")
        respons_srt,abckupresopons=extract_yt_str(user_query_abstract,result2["유튜버"],result2["자막"])
        abckupresopons["원본"]=user_query_abstract
        log_wrapper(f"response: {respons_srt}")
        out,retry=getoutput(respons_srt,result2["링크"])
        if not retry:
            for key, value in out.items():
                log_wrapper(f"{key}: {value}")
            log_wrapper(f'영상정보 : 유튜버: {result2["유튜버"]} 제목: {result2["제목"]} 조회수 :{result2["조회수"]} ')
            return out
        else:
            log_wrapper("링크 획득 실패 2차시도 시작")
            query_string = json.dumps(abckupresopons)
            newquery=self.source.retry(loglist,user_query_abstract,query_string)
            
            log_wrapper(f"새로운 쿼리: {newquery}")
            secondtry=self.source.get_video_data(newquery)
            for inputV in secondtry:
                log_wrapper(f"{inputV['제목']}")
                log_wrapper(f"{inputV['유튜버']}")
            secondtry_2=self.source.get_video_data(newquery,mode="video")
            respons_srt_2,abckupresopons_2=extract_yt_str(newquery,secondtry_2["유튜버"],secondtry_2["자막"])
            abckupresopons_2["원본"]=newquery
            log_wrapper(f"response: {respons_srt_2}")
            out_2,retry=getoutput(respons_srt_2,secondtry_2["링크"])
            for key, value in out_2.items():
                log_wrapper(f"{key}: {value}")
            log_wrapper(f'영상정보 : 유튜버: {secondtry_2["유튜버"]} 제목: {secondtry_2["제목"]} 조회수 :{secondtry_2["조회수"]} ')
            return out_2           
 
 
if __name__ == "__main__":
    Q=survice()
    Q.extract_from_query()
