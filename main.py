from utility import Node
from pprint import pprint
import re
import srt
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


def extract_yt_str(user_query_abstract,youtuber_name,txt_filename,srt_filename):

    input_srt = get_srt_text(srt_filename)
    input_txt = open(txt_filename, "r", encoding="utf-8").read()
    ueer_query_prompt = """
                        너는 유저의 요청에서 가장 중요하게 생각해야할 내용을 항목화해서 LLM이 이해하기 좋은 context화를 하는게 역할이야 단계별로 시작하자
                        1. 먼저 유저의 요청을 확인하고 원하는 바가 몇가지인지 정리해
                        2. 각 조건들의 가짓수와 처음의 요청의 뉘양스를 보아 우선순위를 결정해
                        3. 우선순위대로 번호를 지정하고 LLM에게 조건을 상세히 설명하는 문장을 만들어
                        4. 최종으로 출력할땐 너가 앞에 생각한 과정은 나열하지 말고 3번 결과만 출력해 단순히 3번결과만 출력하고 그대로 LLM에 넣을거야 다른부분이 들어가서 LLM이 혼란스러워하지 않게 
                        [ 중요 ]1,2번은 절대 출력하지마 LLM이 자꾸 잘못된 결과를출력해 반드시 3번의 결과만을 출력해
                      
    """
    user_query_extractor = Node(ueer_query_prompt)
    user_query_extractor_response = user_query_extractor.get_response(user_query_abstract)
    print(user_query_extractor_response)
    slave_prompt = """
                    너의 목적은 제공되는 리뷰 스크립트를 보고 구매를 고려하는 사람이 생각하는 '중요사항'과 비교하여 핵심 내용이 리뷰 스크립트에서 어디에 위치하는지 찾아내는거야 그래서 그 문구를 정확히 출력하고 몇번째 라인인지 알려주는거야 그걸 나와 단계별로 진행하자 그리고 중요사항을 고려해서 선정해야해
                    1. 먼저 리뷰 스크립트를 확인하고 중요사항과 관련되어 의미있는 문구들을 찾아내
                    2. 그중 중요사항을 고려할때 가장 필수적으로 확인해야 할 문구를 골라
                    3. 그 문구가 몇번째 라인에 위치하는지 알아내서 중요사항과 연관있는지 확인하고 연관이 있으면 최종 페이즈를 실행 아니면 1번 부터 다시 시작해
                    최종 페이즈 : 이 스크립트는 소리를 그대로 자막화 한거라 표준 문법이아니야 너의 최종아웃은 원문과 몇번째 라인인지 위치 그리고 표준어 교정본 세가지야
    """
    slave_prompt_srt = """
                    너의 목적은 제공되는 자막 스크립트 요약을 보고 영상에서 어떤 시간이 "중요사항"과 연관되어 중요한지 찾아내고 해당 클립을 따는 링크를 제공하는거야 단계별로 시작하자
                    0. 먼저 요약과 자막을 참고해서 시간을 결정해 자막의 시간은  HH:MM:SS,밀리초 형식이야 단 하나의 시간만을 결정해야해
                    1. 중요사항과 결정한 자막이 연관되어있는지 고찰하고 연관되어있다면 이유를 설명해 만약 아니라면 다시 0번부터 시작해 중요 시간과 자막이 일치하는지 확인해서 일치시켜
                    2. 중요 시간을[; ;](형식은 [;HH:MM:SS:밀리초,HH:MM:SS:밀리초;])안에 출력, 중요시간의 앞부분을 별도로 [HH*3600+MM*60+SS]로 계산해서 [SSSSs] 포멧으로 출력하고, 자막원문은 /?' '?/안에, 그리고 중요한 이유도 "" 안에 서술해줘 만약 중요사항과 연관이 없다면 0번부터 다시 시작해
                    3. [], [; ;], /?' '?/, " " 출력이 모두 생성되었는지 확인하고 모자라다면 0번부터 다시 검토해서 모자란 출력을 생성해 특히 []출력과 포멧확인은 필수야 그리고 10번이상 반복된다면 []출력과 맞는 포멧인지를 최우선으로 고려해서 빠르게 출력해
    """    
    context = {f"중요사항:\n{user_query_extractor_response}"}
    
    slave_node = Node(slave_prompt)
    slave_node.change_context(context)
    slave_node_srt = Node(slave_prompt_srt)
    respons=slave_node.get_response(input_txt)
    context_srt = {f"자막:\n{input_srt}\n\n"\
                    f"요약:\n{respons}\n\n"\
                    f"유튜버:\n{youtuber_name}"}
    slave_node_srt.change_context(context_srt)
    respons_srt=slave_node_srt.get_response(user_query_extractor_response)
    #print(respons_srt)
    return respons_srt

if __name__ == "__main__":
    youtuber_name='잇섭'
    user_query_abstract = "나는 ios만 10년을 써왔어 그런데 이번엔 갤럭시를 써보려고 하는데 적당히 카메라 성능 잘 나오고 괜찮은 갤럭시 모델 있을까?"
    txt_filename = "2.txt"
    srt_filename = "2.srt"
    respons_srt=extract_yt_str(user_query_abstract,youtuber_name,txt_filename,srt_filename)
    print(respons_srt)