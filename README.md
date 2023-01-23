## Basic Info
Fun(Feudalnets)코드의 main.py를 실행해 코드를 돌릴 수 있음

gym-minigrid는 minigrid 환경 코드임

FunMM(Feudalnets Multi Manager)의 코드를 수정해 manager를 추가해 실험할 예정


## Updates
### conv 되도록 수정

Perception 함수 conv 수정: kernel_size, linear의 인수 변경

### wrapper 수정

partial 인수 추가(args.add) -> make_env, Feudalnetwork(=>preprocessor, perception)에 사용

--> make_env: basic_partial_wrapper를 따로 정의하고 partial이면 wrapper로 사용하도록 함

--> preprocessor, perception: partial이면 input_dim을 (56, 56, 3)으로 수정

### mlp 돌아가도록 수정
mlp 부분의 shape 수정 : (shape[-1] * shape[0] * shape[1], ) <- shape[-1]

Feudalnetwork 함수 수정: Perception에 주는 인수 input_dim <- (input_dim[-1])

Perception 함수 mlp Linear 수정: (input_dim[-1] * input_dim[0] * input_dim[1]) <- (input_dim)

## Multi-Manager가 되도록 수정

init_obj(Feudalnnets 클래스에 포함되어 있는)에 higher_goals, higher_states 추가

main.py 92 줄에서 higher_goals, higher_states까지 가져오도록 수정

forward에서 higher_goals, higher_states까지 받아오도록 하고 goals, states와 같은 처리를 진행


## 앞으로 추가할 사항

four-room, empty 외의 환경 하나 더 찾기


## Problems

Doorkey 같이 manager들이 만들어내는 목적들이 가르키는 방향이 다를 수 있는 경우는 어떻게 해야 하는가? 
열쇠를 가지기 전에는 그것을 목적으로 하는 것이 중요하고, 가진 후에는 초록색 목적지로 가는 것이 중요하다,
임무 수행 과정에 따라 어떤 목적에 우선 순위를 어떻게 다르게 두느냐가 문제.


## Env
```
gym==0.19.0
gym_minigrid==1.0.1
```
