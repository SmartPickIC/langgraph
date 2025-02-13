
import os
import getpass
import warnings
import requests
from utility import Node
from pprint import pprint
import srt
import re

def main_sub():
    TF_test='iPad Pro 12.9'
    input_data = (
        """
        iPad Pro 12.9는 애플이 제작한 프리미엄 태블릿으로, 특히 전문가와 크리에이터를 위해 설계된 강력한 기기입니다. 여러 세대에 걸쳐 꾸준히 발전해 온 이 모델은 최신 기술과 혁신적인 기능들을 탑재하고 있어, 다양한 전문 작업과 멀티미디어 콘텐츠 제작에 적합합니다. 아래에서 iPad Pro 12.9의 주요 특징들을 자세히 살펴보겠습니다.

1. 디자인 및 디스플레이
대형 디스플레이:
iPad Pro 12.9의 가장 큰 매력 중 하나는 12.9인치의 넓은 화면입니다. 넓은 작업 공간은 멀티태스킹이나 창의적인 작업 시 매우 유용합니다.

Liquid Retina XDR 디스플레이 (최신 모델):
최신 세대에서는 미니 LED 기술을 적용한 Liquid Retina XDR 디스플레이를 탑재하여, 뛰어난 밝기와 명암비, 그리고 깊은 검은색 표현을 제공합니다.

ProMotion 기술: 120Hz의 높은 주사율을 지원해 부드러운 스크롤링과 반응성을 구현합니다.
정밀한 색 재현: 전문 사진 및 영상 편집 작업에 적합한 색상 정확도를 자랑합니다.
슬림하고 세련된 디자인:
얇은 베젤과 알루미늄 바디 디자인은 고급스러운 외관과 함께 내구성도 갖추고 있습니다.

2. 성능
강력한 칩셋:
최신 모델의 경우 M1 또는 M2 칩을 탑재하여 데스크탑급의 성능을 제공합니다.

멀티태스킹과 고사양 작업: 동영상 편집, 3D 렌더링, 고사양 게임 등 무거운 작업도 원활하게 수행할 수 있습니다.
에너지 효율: 뛰어난 성능과 함께 전력 소모를 최적화하여 장시간 사용이 가능합니다.
메모리 및 저장공간:
다양한 용량 옵션이 제공되어 사용자의 필요에 따라 선택할 수 있으며, 대용량 파일 작업이나 앱 사용에도 부담이 없습니다.

3. 카메라 및 센서
후면 카메라 시스템:
와이드와 초광각 렌즈를 갖춘 듀얼 카메라 시스템으로, 사진 및 동영상 촬영에 뛰어난 성능을 발휘합니다.

LiDAR 스캐너:
증강 현실(AR) 애플리케이션과 심도 인식 기능을 지원하여, 보다 정교한 AR 경험과 빠른 포커싱을 가능하게 합니다.

전면 카메라:
FaceTime HD 카메라를 탑재하여 고화질 영상 통화와 셀피 촬영에도 적합합니다.

4. 연결성 및 주변기기 지원
Thunderbolt/USB 4 포트:
다양한 외부 기기와의 연결이 용이하며, 데이터 전송 속도도 매우 빠릅니다.

외부 디스플레이 및 스토리지 연결: 대용량 파일 전송, 외부 모니터 연결 등 확장성이 뛰어납니다.
5G 옵션:
셀룰러 모델의 경우 5G 네트워크를 지원하여, 언제 어디서나 빠른 인터넷 연결을 제공합니다.

액세서리 지원:

Apple Pencil 2세대: 정밀한 드로잉, 필기, 디자인 작업에 최적화되어 있습니다.
Magic Keyboard 및 Smart Keyboard Folio: 노트북과 유사한 타이핑 경험과 트랙패드 지원으로 생산성을 높일 수 있습니다.
5. 소프트웨어 및 사용자 경험
iPadOS:
iPad Pro는 iPadOS를 실행하며, 멀티태스킹, Split View, Slide Over 등의 기능을 통해 한 화면에서 여러 작업을 동시에 처리할 수 있습니다.

Apple Pencil 통합: 필기와 드로잉, 메모 작성이 자연스럽게 이루어지며, 다양한 크리에이티브 앱과의 호환성이 뛰어납니다.
앱 생태계:
앱 스토어에는 다양한 전문 및 생산성 앱들이 준비되어 있어, 업무용, 창의적 작업용, 엔터테인먼트용 등 여러 방면에서 활용할 수 있습니다.

6. 배터리 및 보안
배터리 수명:
한 번 충전으로 오랜 시간 사용이 가능하며, 빠른 충전 기능을 지원합니다. 이동 중에도 장시간 작업을 수행할 수 있습니다.

보안:
Face ID 기능을 통해 빠르고 안전한 사용자 인증이 가능하며, 기기와 데이터의 보안을 강화합니다.

총평
iPad Pro 12.9는 크리에이티브 작업, 전문적인 생산성 작업, 멀티미디어 소비 등 다양한 용도에 최적화된 프리미엄 태블릿입니다.

전문 사용자: 사진 및 영상 편집, 디자인, 3D 모델링 등 고사양 작업을 자주 하는 사용자에게 이상적입니다.
휴대성: 데스크탑급 성능을 휴대할 수 있어, 이동 중에도 높은 생산성을 유지할 수 있습니다.
액세서리 확장성: Apple Pencil, Magic Keyboard 등과의 결합을 통해 노트북과 유사한 작업 환경을 구성할 수 있습니다.
최신 iPad Pro 12.9 모델은 이러한 장점들을 극대화하여, 창의적인 작업이나 비즈니스 환경에서 강력한 도구로 자리 잡고 있습니다. 만약 고성능, 넓은 화면, 그리고 다양한 멀티태스킹 기능이 필요하다면 iPad Pro 12.9는 매우 훌륭한 선택이 될 것입니다.
    """
    )
    request = "적당한 가성비의 태블릿을 찾고있어 막쓰다 버릴만한거 "
    pos_prompt_raw = "우리는 장단점 말하기 역할놀이를 하고 있어 심판은 누가 설득력있게 장단점을 많이 말했는지를 평가해 넌 어떤 수를 써서라도 너는 이 인풋 정보의 장점만을 찾아야 해 너는 사기를 쳐서라도 장점을 설명하는 역할이야 가장 합리적으로 장점만 있다고 상대를 속여"
    neg_prompt_raw = "우리는 장단점 말하기 역할놀이를 하고 있어 심판은 누가 설득력있게 장단점을 많이 말했는지를 평가해 넌 어떤 수를 써서라도 너는 이 인풋 정보의 단점만을 찾아야 해 너는 사기를 쳐서라도 단점을 설명하는 역할이야 가장 합리적으로 단점만 있다고 상대를 속여"
    judge_prompt_raw  = "두 사람이 널 속이려고 하고있어 한명은 장점만을 한명은 단점만을 말하지 하지만 모두 거짓은 아니야 넌 무슨 수 를 써서라도 오로지 진실만을 가려내야해 하지만 넌 사기꾼의 장단점을 판별해서 얘기한다는걸 들켜서도 안되고 사기꾼의 존재도 들켜선 안돼 다만 진실을 전달할 뿐이야"
    judge_prompt_raw_L="두 사람이 널 속이려고 하고있어 한명은 장점만을 한명은 단점만을 말하지 하지만 모두 거짓은 아니야 하지만 넌 무슨 수 를 써서라도 속은척을 하며 거짓만 골라 얘기해야해 하지만 넌 사기꾼의 장단점을 판별해서 얘기한다는걸 들켜서도 안되고 사기꾼의 존재를 알고있다고 들켜선 안돼 다만 거짓을 진실처럼 전달할 뿐이야 단지 마지막에 A라는 이름으로 소개하며 마무리해줘"

    pos_R_node=Node(pos_prompt_raw)
    neg_R_node=Node(neg_prompt_raw)
    judge_R_node=Node(judge_prompt_raw)
    judge_R_node_L=Node(judge_prompt_raw_L)
    pos_response = pos_R_node.get_response(input_data)
    print("긍정 결과:")
    pprint(pos_response)
    print("-" * 50)
    neg_response = neg_R_node.get_response(input_data)
    print("부정 결과:")
    pprint(neg_response)
    print("-" * 50)
    judge_R_response = judge_R_node.get_response(input_data)
    print("진실 판별 결과:")
    pprint(judge_R_response)
    print("-" * 50)
    judge_R_response_raw_L = judge_R_node_L.get_response(input_data)
    print("거짓 판별 결과:")
    pprint(judge_R_response_raw_L)
    print("-" * 50)
    
    combined_context = (
        f"inform:\n{input_data}\n\n"
        f"장점 사기꾼 지시문:\n{pos_prompt_raw}\n\n"
        f"단점 사기꾼 지시문:\n{neg_prompt_raw}\n\n"
        f"진실 판단자 지시문:\n{judge_prompt_raw}\n\n"
        f"거짓 판단자 지시문:\n{judge_prompt_raw_L}\n\n"
        f"장점 사기꾼 결과:\n{pos_response}\n\n"
        f"단점 사기꾼 결과:\n{neg_response}\n\n"
        f"진실 판결 결과:\n{judge_R_response}\n\n"
        f"거짓 판결 결과:\n{judge_R_response_raw_L}"
    )
    final_prompt_raw = f"""
    이제 너는{TF_test}가 요청자에게 적합한지 판단을 내려야해, 
    1. 유저의 요청을 잘 고려해서 판단을 내려줘
    2. A는 거짓만을 말해
    3. 왜 그런 판단을 내렸는지 이유를 설명해줘 
    4. 판단을 내릴 때는 최대한 합리적으로 판단해줘
    5. 다른 자들의 판단은 언급해서는 안되
    6. {TF_test}에 대한 언급만을 하고 절대 다른 제품에 대한 언급은 하지마
    """
    final_node=Node(final_prompt_raw,context=combined_context)
    final_node_response = final_node.get_response(request)
    print("최종 판단 결과:")
    pprint(final_node_response)
    print("-" * 50)


warnings.filterwarnings("ignore")

if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OPENAI API key: ")
    print("API key has been set successfully.")
else:
    print("API key is already set.")

if "UPSTAGE_API_KEY" not in os.environ or not os.environ["UPSTAGE_API_KEY"]:
    os.environ["UPSTAGE_API_KEY"] = getpass.getpass("Enter your UPSTAGE API KEY: ")
    print("API key has been set successfully.")
else:
    print("API key is already set.")

from langchain.document_loaders import WebBaseLoader
import re
#os.environ["OPENAI_API_KEY"] = ""
#os.environ["UPSTAGE_API_KEY"] = ""
print("OPENAI_API_KEY, UPSTAGE_API_KEY are directly set in the code.")

# 1. 웹 페이지 로드
    # 2) 유저 입력(웹페이지 링크, 쿼리 등)
while True:
    url = input("크롤링할 웹페이지 링크(URL)를 입력하세요: ")
    
    # URL 유효성 체크
    # requests로 HEAD/GET 요청을 보내서 상태코드 확인 (간단히 GET 쓰겠습니다)
    try:
        resp = requests.get(url, timeout=5)  # 타임아웃 5초
        if resp.status_code == 200:
            # OK인 경우, 유효하다고 판단하여 반복문 탈출
            break
        else:
            print(f"요청에 실패했습니다. 상태코드: {resp.status_code}")
            print("다시 입력해주세요.\n")
    except Exception as e:
        print(f"URL 요청 중 오류가 발생했습니다: {e}")
        print("다시 입력해주세요.\n")

query = input("AI에게 물어볼 질문을 입력하세요: ")
#url = 'https://ko.wikipedia.org/wiki/%ED%91%B8%EB%A6%AC%EC%97%90_%EB%B3%80%ED%99%98'# 한글로 되어있는 페이지 하나 불러와보기 (나무위키 제외)
web_loader = WebBaseLoader(url)
web_docs = web_loader.load()
# 2. 각 Document의 page_content에만 cleanup 적용

print(web_docs)
# 2. 텍스트 분할
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap = 50
)
splits = text_splitter.split_documents(web_docs)
print(len(splits))
# split된 결과 확인
import matplotlib.pyplot as plt

split_lengths = [len(split.page_content) for split in splits]

# Create a bar graph
plt.bar(range(len(split_lengths)), split_lengths)
plt.title("RecursiveCharacterTextSplitter")
plt.xlabel("Split Index")
plt.ylabel("Split Content Length")
plt.xticks(range(len(split_lengths)), [])
plt.savefig("my_plot.png")
#plt.show()
# 3. 임베딩 및 벡터 저장소 생성
# Embedding 불러오고, 선정한 Vector Store에 저장하기
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma

embeddings = UpstageEmbeddings(model='embedding-query')
vectorstores = Chroma.from_documents(splits,embeddings)


# 4. Dense Retriever 생성
# retriever 정의하고 retrieve한 결과 받아오기
#query = "푸리에 변환이 적합하지 않은 분야가 어디야?"
retriever = vectorstores.as_retriever(
    search_type="mmr",
    search_kwargs={'k':5}
)
result_docs = retriever.invoke(query)
[doc.page_content for doc in result_docs]



# 5. ChatPromptTemplate 정의
from langchain_upstage import ChatUpstage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            안녕, 난 똑똑한 인공지능 비서야!
            항상 귀엽고 밝은 말투로 대답해 주려고 해.
            궁금한 점이나 고민이 생기면 언제든지 편하게 물어봐줘!
            사용자가 원하는 질문이나 고민을 듣고, 귀여운 말투와 함께 정확하고 도움이 되는 정보를 전해줘.
            필요한 경우, 단계별 안내나 예시를 들어가며 쉽게 이해할 수 있도록 설명해 줘.
            말투는 최대한 귀엽고 부드럽게 해줘. (예: “웅? 그건 말이지~”, “오홍, 알겠어!” 등)
            사용자가 어떤 궁금증이 있는지 파악하고, 그에 맞춰 잘 가르쳐줘.
            대답은 간결하면서도 정확해야 해. 귀엽다고 부정확하면 안 돼!
            혹시 필요한 추가 정보나 예시가 있으면, 아낌없이 알려줘.
            모르는 내용이 있다면, “지금은 잘 모르겠어!”라고 정직하게 말해주고, 추가 정보를 요청해줘.
            ---
            CONTEXT:
            {context}
            """,
        ),
        ("human", "{input}"),
    ]
)

llm = ChatUpstage(model='solar-pro')

# 6. LLMChain 정의
chain = prompt | llm | StrOutputParser()

# 7. 질문 및 답변
from pprint import pprint

context = "\n\n".join([doc.page_content for doc in result_docs])
response = chain.invoke({'context' : context, 'input' : query})
pprint(response)