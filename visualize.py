import mujoco
import numpy as np
import re
import gymnasium as gym
from stable_baselines3 import SAC
import sys
sys.path.append('u-chi-learning/darer')
from ASAC import ASAC
from color_utils import rgba_color_list
import os
import customHumEnv
import customHalfCheetahEnv
import customAntEnv
import customSwimmerEnv
import customHopperEnv

env='humanoid'
with open(f"envxmls/racing_{env}s.xml", "r") as f:
    MODEL_XML = f.read()

#path = f"/Users/jacobadamczyk/miniconda3/envs/rlenv10/lib/python3.10/site-packages/gymnasium/envs/mujoco/assets/{env}.xml"
# get conda env path
path = os.path.join(os.environ['CONDA_PREFIX'], 'envs', 'rlenv10', 'lib', 'python3.10', 'site-packages', 'gymnasium', 'envs', 'mujoco', 'assets', f'{env}.xml')
path = os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'python3.10', 'site-packages', 'gymnasium', 'envs', 'mujoco', 'assets', f'{env}.xml')

path = os.path.join(os.environ['CONDA_PREFIX'], 'Lib', 'site-packages', 'gymnasium', 'envs', 'mujoco', 'assets', f'{env}.xml')
path = os.path.abspath(path)

with open(path, "r") as f:
    xml_content = f.read()

# insert all body elements from humanoid into MODEL_XML:
body_match = re.search(r"<body(.*)>(.*)</body>", xml_content, re.DOTALL)
big_xml_string = ""
xml = body_match.group(0)
# remove the floor:
xml = re.sub(r'<geom.*?floor.*?geom>', '', xml)
# get the actuator:
actuator_match = re.search(r"<actuator(.*)>(.*)</actuator>", xml_content, re.DOTALL).group(0).replace('<actuator>', '').replace('</actuator>', '')
big_actuator_xml = ""
tendon_matcher = re.search(r"<tendon(.*)>(.*)</tendon>", xml_content, re.DOTALL)
if tendon_matcher is not None:
    tendon_match = tendon_matcher.group(0).replace('<tendon>', '').replace('</tendon>', '')
else:
    tendon_match = ""
big_tendon_xml = ""

N_AGENTS = 2
N_REPLICAS = 1
# use an mpl color cycle to color the agents:
colors = rgba_color_list(N_AGENTS)
# colors = [
#     "1 0 0 1",
#     "0 1 0 1",
# ][:N_AGENTS]
colors[0] = "0 1 0 1"

colors = [color for _ in range(N_REPLICAS) for color in colors]
# overwrite the first color as red:

for n_agent in range(N_AGENTS*N_REPLICAS):
    # copy humanoid geometry, actuators, and tendons N_AGENTS times:
    new_agent_xml = re.sub(r'name="', f'name="{n_agent}_', xml)
    
    # Remove any existing rgba attribute first
    new_agent_xml = re.sub(r'rgba="[^"]*"', '', new_agent_xml)
    new_agent_xml = re.sub(r'<geom', 
                           f'<geom contype="{n_agent+1}" conaffinity="0" rgba="{colors[n_agent % len(colors)]}"',
                           new_agent_xml
    )
    new_actuator_xml = re.sub(r'joint="', f'joint="{n_agent}_', actuator_match)
    new_actuator_xml = re.sub(r'name="', f'name="{n_agent}_', new_actuator_xml)

    new_tendon_xml = re.sub(r'joint="', f'joint="{n_agent}_', tendon_match)
    new_tendon_xml = re.sub(r'name="', f'name="{n_agent}_', new_tendon_xml)

    big_xml_string += new_agent_xml
    big_actuator_xml += new_actuator_xml
    big_tendon_xml += new_tendon_xml   

xml_content = MODEL_XML.replace("BODIES_", big_xml_string)
xml_content = xml_content.replace("ACTUATORS_", big_actuator_xml)
xml_content = xml_content.replace("TENDONS_", big_tendon_xml)

# save big xml to tmp file:
with open("tmp.xml", "w") as f:
    f.write(xml_content)

# Load the modified model
model = mujoco.MjModel.from_xml_string(xml_content)
data = mujoco.MjData(model)

# capitalize first letter:
# capitalize after _:
mj_env = ''
for fragment in env.split('_'):
    mj_env += fragment.capitalize()
asac_model = ASAC.load(args=(mj_env+'-v5',), path=f'ASAC_{mj_env}-v5', device='cpu')
sac_model = SAC.load(env+"-v5-sac-expert")
env_name = f'Racing{mj_env}s-v5'
tmp_path = os.path.join(os.getcwd(), 'tmp.xml')

env = gym.make(env_name, n_agents=N_AGENTS*N_REPLICAS, xml_file=tmp_path, render_mode='human')
# agents = [model for _ in range(N_AGENTS)]
agents = [sac_model, asac_model.actor][::1][:N_AGENTS]#[asac_model.actor, asac_model.actor][:N_AGENTS]
agents = [agent for _ in range(N_REPLICAS) for agent in agents]
assert N_AGENTS*N_REPLICAS == len(agents)
obs, info = env.reset()
full_len = obs.shape[0]
ep_return = 0
one_obs_len = full_len//(N_REPLICAS*N_AGENTS)
for _ in range(10000):
    # combine actions from each agent:
    actions = []
    agent_obs = np.split(obs, N_REPLICAS*N_AGENTS, axis=0) 

    for n_agent, agent in enumerate(agents):
        o = agent_obs[n_agent]
        # o = obs[n_agent*one_obs_len:(n_agent+1)*one_obs_len]
        action, _ = agent.predict(o, deterministic=True)
        actions.append(action)
    # action = env.action_space.sample()

    action = np.concatenate(actions)
    obs, reward, term, trunc, info = env.step(action)
    ep_return += reward
    # env.render()
    # print(ep_return)
