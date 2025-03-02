import mujoco
# import mujoco_viewer
import numpy as np
import re

with open("envxmls/racinghums.xml", "r") as f:
    MODEL_XML = f.read()

# Load the original XML
env='humanoid'
#path = f"/Users/jacobadamczyk/miniconda3/envs/rlenv10/lib/python3.10/site-packages/gymnasium/envs/mujoco/assets/{env}.xml"
# get conda env path
import os
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
tendon_match = re.search(r"<tendon(.*)>(.*)</tendon>", xml_content, re.DOTALL).group(0).replace('<tendon>', '').replace('</tendon>', '')
big_tendon_xml = ""

N_AGENTS = 2
colors = [
    "1 0 0 1",  # Red
    "0 1 0 1",  # Green
    "0 0 1 1",  # Blue
    "1 1 0 1",  # Yellow
    "1 0 1 1",  # Magenta
][:N_AGENTS]


for n_agent in range(N_AGENTS):
    # copy humanoid geometry, actuators, and tendons N_AGENTS times:
    new_agent_xml = re.sub(r'name="', f'name="{n_agent}_', xml)
    # properly replace unique geom tags using same sub logic:
    
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
    

xml_content = MODEL_XML.replace("SPHEROS_", big_xml_string)
xml_content = xml_content.replace("ACTUATORS_", big_actuator_xml)
xml_content = xml_content.replace("TENDONS_", big_tendon_xml)

# save big xml to tmp file:
with open("tmp.xml", "w") as f:
    f.write(xml_content)

# Load the modified model
model = mujoco.MjModel.from_xml_string(xml_content)
data = mujoco.MjData(model)

# Simulation loop
import gymnasium as gym
from stable_baselines3 import SAC
# model = SAC.load("humanoid-v5-sac-simple", custom_objects=
model = SAC.load("humanoid")
model2 = SAC.load("humanoid")

env_name='RaceingHumanoids-v5'
#xml_path='/Users/jacobadamczyk/Documents/Github/raceRL/tmp.xml'
import customHumEnv
tmp_path = os.path.join(os.getcwd(), 'tmp.xml')
env = gym.make(env_name, n_agents=2, xml_file=tmp_path, render_mode='human')
# exit()

obs, info = env.reset()

for _ in range(10000):
    # normal rl rendering / steps:
    action = model.predict(obs[:len(obs)//N_AGENTS], deterministic=True)[0]
    action2 = model2.predict(obs[len(obs)//N_AGENTS:], deterministic=True)[0]
    action = np.concatenate([action, action2])
    obs, reward, term, trunc, info = env.step(action)
    env.render()
    
# viewer.close()

print("Data collection complete. Saved to two_humanoids_random_data.npy")
