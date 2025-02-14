---
# LangGraph 예제

이 레포지토리는 `utility.py`에 정의된 클래스를 활용하여 **OpenAPI**와 **Upstage**의 LLM을 이용한 노드를 구현한 예제입니다.

## Node 클래스

`Node` 클래스는 LLM 모델을 선택하고, 프롬프트와 컨텍스트를 설정하여 노드를 생성합니다. 기본적으로 `openai`의 `chatgpt-4o-latest` 모델을 사용하며, 다음과 같이 사용할 수 있습니다:

```python
class Node:
    def __init__(self, prompt, model='openai', context="", gptmodel=None):
        self.controller = apicon
        if model == 'upstage':
            llmmodel = "solar-pro"
        elif model == 'openai':
            if gptmodel:
                llmmodel = gptmodel
            else:
                llmmodel = "chatgpt-4o-latest"
        self.llm = self.controller.get_llm_model(llmmodel, model=model)
        self.prompt = self.controller.get_prompt(prompt)
        self.context = context
```

**사용 예시:**  
```python
node = Node("프롬프트 내용")
```

## 실행 예시: 유튜브 클립 링크 에이전트

`main.py`를 실행하면, `2.txt`와 `2.srt` 파일을 활용하여 아래와 같은 시나리오가 진행

### 입력 예시

```
나는 ios만 10년을 써왔어 그런데 이번엔 갤럭시를 써보려고 하는데 적당히 카메라 성능 잘 나오고 괜찮은 갤럭시 모델 있을까?
```

### 출력 예시

유튜브 클립 링크를 제공하는 에이전트(`extract_yt_str`)의 결과예시

```
사유: 갤럭시 신제품 출시와 관련된 정보는 iOS에서 갤럭시로 전환하려는 사용자에게 중요한 맥락이 될 수 있음
시간:  00:00:17.119000, 00:00:22.680000
링크: https://www.youtube.com/watch?v=3PCR3QzbJSg&t=17s
원문: 
```

## 요약

- **LLM 모델 선택:**  
  - `upstage` 모델 선택 시: `"solar-pro"`
  - `openai` 모델 기본 선택: `chatgpt-4o-latest` (원하는 경우 `gptmodel` 인자를 통해 변경 가능)
- **노드 생성:**  
  `Node(prompt)` 형태로 인스턴스를 생성하여 사용
- **주요 기능:**  
  `main.py` 실행 시, `2.txt`와 `2.srt` 파일을 기반으로 입력에 따른 유튜브 클립 링크 등 필요한 정보를 추출

정밀도 개선 예정...
