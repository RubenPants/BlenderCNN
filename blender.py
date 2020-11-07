import json
import os
from random import choice, uniform

import bpy

# Hyper-parameters
N_SAMPLES = 10000
PATH = os.path.expanduser('~/data/flying_dots/')
METADATA = {}


def get_random_color():
    """Return a random but vibrant color."""
    c = choice(range(6))
    var = uniform(0, 1)
    if c == 0: return 1, var, 0, 1
    if c == 1: return var, 1, 0, 1
    if c == 2: return 0, 1, var, 1
    if c == 3: return 0, var, 1, 1
    if c == 4: return var, 0, 1, 1
    return 1, 0, var, 1


def generate(idx=0):
    # Get focus of Ball object
    so = bpy.context.scene.objects["Ball"]
    
    # Move to random location
    so.location[0] = uniform(-1, 1)  # x-axis
    so.location[1] = uniform(-1, 1)  # y-axis
    
    # Change color of Glow material
    material = bpy.data.materials["Glow"].node_tree.nodes["Emission"].inputs[0]
    material.default_value = get_random_color()
    
    # Debug: don't persist if empty filename
    if not idx: return
    filename = f'sample_{idx}'
    
    # Append hex-values of Ball's Glow
    METADATA[filename] = list(material.default_value)[:3] + list(so.location[:2])
    
    # Render and save
    bpy.context.scene.render.filepath = os.path.join(PATH, filename + '.png')
    bpy.ops.render.render(write_still=True)
    print(f"Rendered and saved '{filename}'")
    
    # Save intermediately
    if idx % 10 == 0:
        with open(os.path.join(PATH, 'temp', f'metadata_{idx}.json'), 'w') as f:
            json.dump(METADATA, f, indent=2)


# Generate samples
for i in range(1, N_SAMPLES + 1):
    generate(i)

# Store hex-data of generated samples
with open(os.path.join(PATH, 'metadata.json'), 'w') as f:
    json.dump(METADATA, f, indent=2)

print(f"\n==> FINISHED GENERATING {N_SAMPLES} SAMPLES <==")
