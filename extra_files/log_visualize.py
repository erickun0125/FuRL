import re
import pandas as pd
import matplotlib.pyplot as plt

# 로그 파일 경로 (실제 경로로 변경)
log_file = "logs/furl_rho0.05/door-open-v2-goal-hidden/s0_20250227_041251.log"

# 각 로그 블록을 저장할 리스트
blocks = []

# 헤더 패턴: [T ...K][... min] task_reward: ..., vlm_reward: ...
header_pattern = re.compile(r"\[T (\d+)K\]\[(.*?) min\] task_reward: ([\d\.\-]+), vlm_reward: ([\d\.\-]+)")

# 추가 키-값 패턴: key: value
# (goal은 튜플 형태로 되어 있는데, 여기서는 문자열로 저장)
kv_pattern = re.compile(r"([\w_]+):\s*([\(\)\d\.\-\,\s]+)")

# 파일 읽기
with open(log_file, "r", encoding="utf-8") as f:
    current_block = None
    for line in f:
        # 헤더 라인을 찾으면 새 블록 생성
        header_match = header_pattern.search(line)
        if header_match:
            # 새 블록 딕셔너리 생성
            current_block = {}
            timestep_k = int(header_match.group(1))
            current_block["Timestep"] = timestep_k * 1000   # 예: 10K -> 10000
            current_block["Time (min)"] = float(header_match.group(2))
            current_block["task_reward"] = float(header_match.group(3))
            # 헤더의 vlm_reward와 이후에 나타나는 vlm_reward(예: 추가 정보) 구분을 위해
            current_block["header_vlm_reward"] = float(header_match.group(4))
            blocks.append(current_block)
        else:
            # 만약 줄이 공백이 아니고 현재 블록이 진행중이면, 추가 키-값을 파싱
            if current_block is not None and line.strip():
                # 여러 key-value가 있을 수 있으므로 콤마 기준으로 분리
                parts = line.strip().split(",")
                for part in parts:
                    kv_match = kv_pattern.search(part)
                    if kv_match:
                        key = kv_match.group(1).strip()
                        value_str = kv_match.group(2).strip()
                        # value가 튜플 형태이면 그냥 문자열로 저장하거나, 필요시 파싱 가능
                        try:
                            # 여러 값가 있을 수 있으므로, 단일 float 변환 시도
                            value = float(value_str)
                        except ValueError:
                            value = value_str
                        # 만약 이미 키가 있다면(예: 두 번 나타난 vlm_reward) 새로운 이름으로 저장
                        if key in current_block:
                            key = "second_" + key
                        current_block[key] = value

# DataFrame 생성 (블록이 하나도 없으면 에러 처리)
if not blocks:
    print("로그 파일에서 데이터를 찾지 못했습니다. 정규표현식 및 로그 파일 포맷을 확인하세요.")
else:
    df = pd.DataFrame(blocks)
    # 데이터프레임 확인
    print(df.head())

    # ----- 1. Reward Metrics Plot -----
    # 포함 항목: task_reward, header_vlm_reward, 그리고 만약 존재하면 second_vlm_reward
    plt.figure(figsize=(10, 5))
    plt.plot(df["Timestep"], df["task_reward"], label="Task Reward", marker='o')
    plt.plot(df["Timestep"], df["header_vlm_reward"], label="Header VLM Reward", marker='s')
    if "second_vlm_reward" in df.columns:
        plt.plot(df["Timestep"], df["second_vlm_reward"], label="Additional VLM Reward", marker='^')
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title("Reward Metrics over Time")
    plt.legend()
    plt.grid()
    plt.savefig("reward_metrics.png")
    plt.close()

    # ----- 2. Loss and Q-value Metrics Plot -----
    # 포함 항목: q_loss, a_loss, q, q_max (존재하는 경우만)
    plt.figure(figsize=(10, 5))
    if "q_loss" in df.columns:
        plt.plot(df["Timestep"], df["q_loss"], label="Q Loss", marker='o')
    if "a_loss" in df.columns:
        plt.plot(df["Timestep"], df["a_loss"], label="A Loss", marker='s')
    if "q" in df.columns:
        plt.plot(df["Timestep"], df["q"], label="Q Value", marker='^')
    if "q_max" in df.columns:
        plt.plot(df["Timestep"], df["q_max"], label="Q Max", marker='d')
    plt.xlabel("Timestep")
    plt.ylabel("Loss / Q Value")
    plt.title("Loss and Q Value Metrics")
    plt.legend()
    plt.grid()
    plt.savefig("loss_q_metrics.png")
    plt.close()

    # ----- 3. Return Metrics Plot -----
    # 포함 항목: R, Rmax, Rmin, vlm_R, vlm_Rmax, vlm_Rmin, 그리고 rvlm_R (존재하는 경우)
    plt.figure(figsize=(10, 5))
    if "R" in df.columns:
        plt.plot(df["Timestep"], df["R"], label="R", marker='o')
    if "Rmax" in df.columns:
        plt.plot(df["Timestep"], df["Rmax"], label="Rmax", marker='s')
    if "Rmin" in df.columns:
        plt.plot(df["Timestep"], df["Rmin"], label="Rmin", marker='^')
    if "vlm_R" in df.columns:
        plt.plot(df["Timestep"], df["vlm_R"], label="VLM R", marker='d')
    if "vlm_Rmax" in df.columns:
        plt.plot(df["Timestep"], df["vlm_Rmax"], label="VLM Rmax", marker='p')
    if "vlm_Rmin" in df.columns:
        plt.plot(df["Timestep"], df["vlm_Rmin"], label="VLM Rmin", marker='h')
    if "rvlm_R" in df.columns:
        plt.plot(df["Timestep"], df["rvlm_R"], label="rVLM R", marker='x')
    plt.xlabel("Timestep")
    plt.ylabel("Return")
    plt.title("Return Metrics over Time")
    plt.legend()
    plt.grid()
    plt.savefig("return_metrics.png")
    plt.close()

    # ----- 4. Success Metrics Plot -----
    # 포함 항목: ep_num, success_cnt, success, vlm_success, vlm_step (존재하는 경우)
    plt.figure(figsize=(10, 5))
    if "ep_num" in df.columns:
        plt.plot(df["Timestep"], df["ep_num"], label="Episode Number", marker='o')
    if "success_cnt" in df.columns:
        plt.plot(df["Timestep"], df["success_cnt"], label="Success Count", marker='s')
    if "success" in df.columns:
        plt.plot(df["Timestep"], df["success"], label="Success", marker='^')
    if "vlm_success" in df.columns:
        plt.plot(df["Timestep"], df["vlm_success"], label="VLM Success", marker='d')
    if "vlm_step" in df.columns:
        plt.plot(df["Timestep"], df["vlm_step"], label="VLM Step", marker='p')
    plt.xlabel("Timestep")
    plt.ylabel("Count")
    plt.title("Success Metrics over Time")
    plt.legend()
    plt.grid()
    plt.savefig("success_metrics.png")
    plt.close()

    print("각 범주의 그래프가 png 파일로 저장되었습니다.")
