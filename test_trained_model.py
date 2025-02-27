import os
import time
import numpy as np
import torch
import cv2
import imageio
import clip
import ml_collections
import gymnasium as gym
import torchvision.transforms as T

from models import SACAgent, FuRLAgent, RewardModel
from utils import TASKS, get_logger, make_env, load_liv

# 🔹 테스트 환경 설정
def get_test_config():
    config = ml_collections.ConfigDict()
    config.env_name = "peg-insert-side-v2-goal-observable"
    config.camera_id = 2
    config.seed = 0
    config.eval_episodes = 10  # 테스트할 에피소드 수
    # 체크포인트 경로는 절대 경로로 지정
    config.ckpt_dir = os.path.abspath(f"save_models/{config.env_name}")
    return config

# 🔹 모델 로드 함수
def load_trained_models(config):
    # LIV 모델 로드
    transform = T.Compose([T.ToTensor()])
    liv = load_liv()

    # 환경 초기화
    env = make_env(config.env_name, seed=config.seed, camera_id=config.camera_id)

    # CLIP을 이용한 목표 임베딩 로드
    with torch.no_grad():
        token = clip.tokenize([TASKS[config.env_name]])
        text_embedding = liv(input=token, modality="text")
    text_embedding = text_embedding.detach().cpu().numpy()

    # 모델 초기화 및 체크포인트 불러오기
    vlm_agent = FuRLAgent(obs_dim=env.observation_space.shape[0],
                          act_dim=env.action_space.shape[0],
                          max_action=env.action_space.high[0],
                          seed=config.seed,
                          ckpt_dir=config.ckpt_dir,
                          text_embedding=text_embedding,
                          goal_embedding=None)
    
    sac_agent = SACAgent(obs_dim=env.observation_space.shape[0],
                         act_dim=env.action_space.shape[0],
                         max_action=env.action_space.high[0],
                         seed=config.seed,
                         ckpt_dir=config.ckpt_dir)
    
    reward_model = RewardModel(seed=config.seed,
                               ckpt_dir=config.ckpt_dir,
                               text_embedding=text_embedding,
                               goal_embedding=None)

    # 불러올 체크포인트 스텝 (최근 모델 불러오기)
    vlm_ckpt_dir = os.path.join(config.ckpt_dir, "vlm_agent")
    # 이름이 순수 숫자인 폴더만 후보로 선택 (임시 파일 등 제외)
    ckpt_candidates = [d for d in os.listdir(vlm_ckpt_dir) if d.isdigit()]
    if not ckpt_candidates:
        raise ValueError(f"No valid checkpoint found in {vlm_ckpt_dir}")
    latest_ckpt = max([int(d) for d in ckpt_candidates])

    print(f"🔄 Loading checkpoint from step {latest_ckpt}")

    # config.ckpt_dir만 전달 (내부에서 하위 폴더가 추가됨)
    vlm_agent.load(config.ckpt_dir, latest_ckpt)
    sac_agent.load(config.ckpt_dir, latest_ckpt)
    reward_model.load(config.ckpt_dir, latest_ckpt)

    return env, vlm_agent, sac_agent, reward_model, transform, liv

# 🔹 테스트 실행 함수
def test_model():
    config = get_test_config()
    env, vlm_agent, sac_agent, reward_model, transform, liv = load_trained_models(config)
    
    eval_episodes = config.eval_episodes
    total_rewards = []
    success_count = 0
    video_frames = []

    print(f"🚀 Running {eval_episodes} evaluation episodes...")

    for episode in range(eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step_count = 0  # 타임스텝 카운트

        print(f"\n🎬 Episode {episode + 1} start:")

        while not done:
            step_count += 1

            # VLM Agent에서 행동 샘플링
            action = vlm_agent.sample_action(obs, eval_mode=True)
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 환경에서 받은 보상 (Task Reward)
            episode_reward += env_reward

            # 비전-언어 모델을 활용한 Reward 계산
            with torch.no_grad():
                image = env.mujoco_renderer.render(render_mode="rgb_array", camera_id=config.camera_id).copy()
                image = image[::-1]
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                processed_image = transform(image)
                image_embedding = liv(input=processed_image.to("cuda")[None], modality="vision")
                image_embedding = image_embedding.detach().cpu().numpy()

                vlm_reward = reward_model.get_vlm_reward(reward_model.proj_state, image_embedding).item()

            # 보상 출력
            print(f"🔹 Step {step_count} | Task Reward: {env_reward:.3f}, VLM Reward: {vlm_reward:.3f}")

            # 비디오 프레임 저장
            video_frames.append(image)

            obs = next_obs

        total_rewards.append(episode_reward)
        success_count += int(info["success"])
        print(f"✅ Episode {episode + 1} completed. Total Reward: {episode_reward:.2f}, Success: {info['success']}")

    env.close()

    # 평가 결과 출력
    avg_reward = np.mean(total_rewards)
    success_rate = success_count / eval_episodes * 100
    print(f"\n📊 Average Reward: {avg_reward:.2f}")
    print(f"✅ Success Rate: {success_rate:.2f}%")

    # 비디오 저장
    video_path = os.path.join(config.ckpt_dir, "test_results.mp4")
    imageio.mimsave(video_path, video_frames, fps=30)
    print(f"🎥 Evaluation video saved at: {video_path}")

if __name__ == "__main__":
    test_model()
