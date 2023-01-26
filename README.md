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

## manager-Supervisor-Worker 코드로 수정

1. hidden_dim_supervisor,gamma_s(supervisor discount factor)  인수 추가

2. init_obj에 goal_s, state_s(supervisor) 추가 

    -> template_m과 template_s로 나누어 goal_m, state_m과 goal_s, state_s의 형태를 다르게 세팅

    -> >manager의 goals, states는 _m을 붙여서 처리, supervisor의 goals, states는 _s를 붙여서 처리

3. supervisor detach
4. supervisor의 loss는 state_goal_cosine: d_cos(states_s[t + self.c] - states_s[t], goals_s[t])
5. manager의 loss는 goal_goal,cosine: d_cos(goals_s_present - goals_s_past, goals_m[t])

    -> goals_m의 값을 supervisor에서 반영한 정도를 지표로 사용
   
    ->present, past 모두 각 시점의 goals_s를 concat해서 goals_m과 형태를 맞춰줌

6. 임시 이름은 bracketnet으로 지었으나 아직 확정하지 않아 feudalnet 함수 이름을 그대로 사용
7. feudal_loss에 ret_m, ret_ 같이 supervisor의 discount reward 추가
8. 총 loss: - loss_worker - ((loss_manager + loss_supervisor)/2) + value_w_loss + value_m_loss - args.entropy_coef * entropy

## 앞으로 추가할 사항



## Questions

manager의 goal_goal_cosine에서 past와 present를 각 시점 goals_s를 concat하는 방식을 적용했읋 때, goals_m과 비교하기에 무리가 없는가? 

supervisor의 srnn에 넣는 값을 (manager_goal + state)로 했을 때. 제대로 manager의 정보와 supervisor의 정보가 반영되는 것인가?

feudal_loss에 임시로 supervisor_loss를 집어넣은 것인데, 원래 loss 수식을 만들때 수학적 과정이 있을 것이라 생각하는데 어떻게 하는 것인가?


## Env
```
gym==0.19.0
gym_minigrid==1.0.1
```
