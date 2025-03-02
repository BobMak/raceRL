# for a given environment, this script makes a new xml
# that allows for multiple bodies to be controlled and visualized
# in the same mujoco scene

import os
import xml.etree.ElementTree as ET
path = '/Users/jacobadamczyk/miniconda3/envs/rlenv10/lib/python3.10/site-packages/gymnasium/envs/mujoco/assets'
ENV_NAME = 'half_cheetah'
# Load the original xml file
tree = ET.parse(f'{path}/{ENV_NAME}.xml')

# copy the entire body element
body = tree.find('worldbody/body')
# change the position a bit:
for i in range(1, 5):
    new_body = ET.Element('body', name=f'body{i}', pos=f'{i} 0 0')
    new_geom = ET.Element('geom', name=f'geom{i}', pos='0 0 0', size='0.5', type='sphere')
    new_body.append(new_geom)
    body.append(new_body)

# now, add the new body to a new xml file and save it:
new_tree = ET.ElementTree(tree.getroot())
new_tree.write(f'{path}/{ENV_NAME}_multi.xml')
