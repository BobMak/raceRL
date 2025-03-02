import mujoco
import mujoco_viewer
import numpy as np
import re

MODEL_XML = """<?xml version="1.0" ?>
<mujoco>
    <compiler angle="degree" inertiafromgeom="true"/>

    <option timestep="0.005" />
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom conaffinity="1" condim="1" contype="1" margin="0.001" material="geom" rgba="0.8 0.6 .4 1"/>
        <motor ctrllimited="true" ctrlrange="-.4 .4"/>
    </default>
    <option integrator="RK4" iterations="50" solver="PGS" timestep="0.003">
        <!-- <flags solverstat="enable" energy="enable"/>-->
    </option>
    <size nkey="5" nuser_geom="1"/>
    <asset>
        <texture builtin="gradient" height="100" rgb1=".4 .5 .6" rgb2="0 0 0" type="skybox" width="100"/>
        <!-- <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>-->
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>
    <worldbody>
        <geom condim="3" friction="1 .1 .1" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 0.125" type="plane" contype="1" conaffinity="255"/>
        <camera euler="0 0 0" fovy="40" name="rgb" pos="0 0 2.5"></camera>
        SPHEROS_
    </worldbody>
    <actuator>
        ACTUATORS_
    </actuator>
    <tendon>
        TENDONS_
    </tendon>
</mujoco>
"""

# Load the original XML
env='humanoid'
path = f"/Users/jacobadamczyk/miniconda3/envs/rlenv10/lib/python3.10/site-packages/gymnasium/envs/mujoco/assets/{env}.xml"
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
    # Replace each body tag name with appended agent number:
    # for ex.         <body name="torso" pos="0 0 1.4">
    # becomes         <body name="torso_0" pos="0 0 1.4">:
    # print(n_agent)
    
    new_agent_xml = re.sub(r'name="', f'name="{n_agent}_', xml)
    # properly replace unique geom tags using same sub logic:
    
    new_agent_xml = re.sub(r'<geom', 
                           f'<geom contype="{n_agent+1}" conaffinity="0" rgba="{colors[n_agent % len(colors)]}"',
                           new_agent_xml
    )
    # Copy the actuators, renaming  the joint properly
    # loop through joints and get proper name, relabelling 
    # e.g.     <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="150"/>
    # becomes  <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4_0" gear="150"/>
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


env_name='Humanoid-v5'
xml_path='/Users/jacobadamczyk/Documents/Github/raceRL/tmp.xml'
env = gym.make(env_name, xml_file=xml_path, render_mode='human')
# exit()
# action_dim = env.action_space.shape[0]
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
