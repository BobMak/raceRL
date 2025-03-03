import mujoco
import numpy as np
import re
import gymnasium as gym
from stable_baselines3 import SAC
from color_utils import rgba_color_list
import os
import customHumEnv
import customHalfCheetahEnv

env='humanoid'
with open(f"envxmls/racing_{env}s.xml", "r") as f:
    MODEL_XML = f.read()

#path = f"/Users/jacobadamczyk/miniconda3/envs/rlenv10/lib/python3.10/site-packages/gymnasium/envs/mujoco/assets/{env}.xml"
# get conda env path
# path = os.path.join(os.environ['CONDA_PREFIX'], 'envs', 'rlenv10', 'lib', 'python3.10', 'site-packages', 'gymnasium', 'envs', 'mujoco', 'assets', f'{env}.xml')
# path = os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'python3.10', 'site-packages', 'gymnasium', 'envs', 'mujoco', 'assets', f'{env}.xml')

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

N_AGENTS = 20
# use an mpl color cycle to color the agents:
colors = rgba_color_list(N_AGENTS)

for n_agent in range(N_AGENTS):
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


model = SAC.load(env)
# capitalize first letter:
env_name = f'Racing{env.capitalize()}s-v5'
tmp_path = os.path.join(os.getcwd(), 'tmp.xml')

env = gym.make(env_name, n_agents=N_AGENTS, xml_file=tmp_path, render_mode='human')

# exit()
agents = [model for _ in range(N_AGENTS)]
obs, info = env.reset()
obs_len = obs.shape[0]
for _ in range(10000):
    # combine actions from each agent:
    actions = []
    for n_agent, agent in enumerate(agents):
        action, _ = agent.predict(obs[n_agent*obs_len//N_AGENTS:(n_agent+1)*obs_len//N_AGENTS])
        actions.append(action)
    # action = env.action_space.sample()

    action = np.concatenate(actions)
    obs, reward, term, trunc, info = env.step(action)
    env.render()
