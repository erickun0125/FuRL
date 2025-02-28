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
    config.env_name = "door-open-v2-goal-hidden"
    config.camera_id = 2
    config.seed = 0
    # 체크포인트 경로는 절대 경로로 지정
    config.ckpt_dir = os.path.abspath(f"save_models/{config.env_name}")
    return config

# 🔹 환경, LIV, 텍스트 임베딩, 전처리 등 정적 컴포넌트 로드 함수
def load_static_components(config):
    transform = T.Compose([T.ToTensor()])
    liv = load_liv()
    env = make_env(config.env_name, seed=config.seed, camera_id=config.camera_id)
    with torch.no_grad():
        token = clip.tokenize([TASKS[config.env_name]])
        text_embedding = liv(input=token, modality="text")
    text_embedding = text_embedding.detach().cpu().numpy()
    return env, transform, liv, text_embedding

# 🔹 100000부터 latest_ckpt까지 100000 단위 체크포인트 리스트 생성 함수
def get_checkpoint_list(config, start_ckpt=100000):
    vlm_ckpt_dir = os.path.join(config.ckpt_dir, "furl_agent")
    ckpt_candidates = [int(d) for d in os.listdir(vlm_ckpt_dir) if d.isdigit()]
    if not ckpt_candidates:
        raise ValueError(f"No valid checkpoint found in {vlm_ckpt_dir}")
    latest_ckpt = max(ckpt_candidates)
    # 100000 단위로 체크포인트 리스트 생성 (latest_ckpt가 100000의 배수가 아니어도, 그보다 작거나 같은 값들)
    checkpoints = list(range(start_ckpt, latest_ckpt + 1, 100000))
    return checkpoints, latest_ckpt

# 🔹 주어진 체크포인트 번호로 모델 로드 함수
def load_models_at_checkpoint(config, env, text_embedding, ckpt):
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
    
    print(f"🔄 Loading checkpoint {ckpt}")
    vlm_agent.load(config.ckpt_dir, ckpt)
    sac_agent.load(config.ckpt_dir, ckpt)
    reward_model.load(config.ckpt_dir, ckpt)
    
    return vlm_agent, sac_agent, reward_model

# 🔹 체크포인트별로 한 에피소드씩 테스트하고 전체 영상을 저장하는 테스트 실행 함수
def test_model():
    config = get_test_config()
    env, transform, liv, text_embedding = load_static_components(config)
    checkpoints, latest_ckpt = get_checkpoint_list(config)
    
    video_frames = []
    overall_rewards = {}
    overall_successes = {}
    
    print(f"🚀 Testing checkpoints: {checkpoints}")
    
    # 각 체크포인트마다 한 에피소드 실행
    for ckpt in checkpoints:
        episode_reward = 0
        step_count = 0
        
        # 각 체크포인트에 대해 모델 로드
        vlm_agent, sac_agent, reward_model = load_models_at_checkpoint(config, env, text_embedding, ckpt)
        
        print(f"\n🎬 Testing checkpoint {ckpt}:")
        obs, _ = env.reset()
        done = False
        
        while not done:
            step_count += 1
            # VLM Agent에서 행동 샘플링
            action = vlm_agent.sample_action(obs, eval_mode=True)
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += env_reward
            
            # 비전-언어 모델을 활용한 Reward 계산
            with torch.no_grad():
                image = env.mujoco_renderer.render(render_mode="rgb_array", camera_id=config.camera_id).copy()
                image = image[::-1]  # 이미지 상하 반전
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                processed_image = transform(image)
                image_embedding = liv(input=processed_image.to("cuda")[None], modality="vision")
                image_embedding = image_embedding.detach().cpu().numpy()
                
                vlm_reward = reward_model.get_vlm_reward(reward_model.proj_state, image_embedding).item()
            
            print(f"🔹 Checkpoint {ckpt} | Step {step_count} | Task Reward: {env_reward:.3f}, VLM Reward: {vlm_reward:.3f}")
            
            # 비디오 프레임 저장 (모든 체크포인트의 영상이 하나의 비디오에 기록됨)
            video_frames.append(image)
            
            obs = next_obs
        
        overall_rewards[ckpt] = episode_reward
        overall_successes[ckpt] = info.get("success", False)
        print(f"✅ Checkpoint {ckpt} episode completed. Total Reward: {episode_reward:.2f}, Success: {info.get('success', False)}")
    
    env.close()
    
    # 각 체크포인트의 평가 결과 출력
    print("\n📊 Evaluation Summary:")
    for ckpt in checkpoints:
        print(f"Checkpoint {ckpt} -> Total Reward: {overall_rewards[ckpt]:.2f}, Success: {overall_successes[ckpt]}")
    
    # 전체 영상 저장 (모든 에피소드의 프레임을 하나의 영상으로 저장)
    video_path = os.path.join(config.ckpt_dir, "test_results.mp4")
    imageio.mimsave(video_path, video_frames, fps=30)
    print(f"\n🎥 Evaluation video saved at: {video_path}")

if __name__ == "__main__":
    test_model()
