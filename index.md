# 1. Project description

The goal of the project is to generate images based on vectoral data. In every image, a floating glowing ball is shown in a dark room. There are two aspects that vary in every image, this being the color and the position of the floating ball.

Every scene is represented by a vector of five floating numbers: 

- Three values representing the RGB-color that the glowing ball emits
- Two values representing the position (x and y axis) of the ball

Based on this vector (also called the image's feature), a Deep Learning model tries to re-generate an image of the original scene.

# 2. Blender

Since the number of images with a *floating glowing ball in a dark room* is sparse, another way to collect this data needs to be found. [Blender](https://www.blender.org/), a free and open source 3D creation suite, comes to the rescue! Blender is an easy to use program with a vibrant community that covers all artistic needs ranging from creating art to creating data for Machine Learning.

## 2.1. Scene

For our use-case, we'll only need a simple scene that has a room (this being a floor and two walls), as well as a floating ball. To make the scene more visually pleasing, we'll make the ball glow and the floor highly reflective (since in Blender, you can do whatever tickles your fancy).

![Blender scene](https://github.com/RubenPants/BlenderCNN/blob/main/images/blender_combined.png?raw=true)

## 2.2. Code

Luckily, you don't need to change and render each scene manually in order to collect your data. More even, you can write your own Python scripts in Blender to automate this process! The following lines of code show you the most important methods to call in order to automate the data generation. To experiment with this code yourself, open the *Scripting* tab in the Blender-scene included in the repo. **Note:** The code shown below assumes that you've already created a Blender scene. 

In order to let the Blender-magic spark, you'll need to import *blenderpy* (`bpy`) package first:

```python
import bpy
```

To focus on the right object (i.e. the object you want to augment) in your blender scene, run:

```python
so = bpy.context.scene.objects["<name-of-your-object>"]
```

You can make changes to your object as follows:

```python
# Move to random location
so.location[0] = uniform(-1, 1)  # x-axis
so.location[1] = uniform(-1, 1)  # y-axis

# Change color of the 'Glow' material
material = bpy.data.materials["Glow"].node_tree.nodes["Emission"].inputs[0]
material.default_value = get_random_color()  # RGBA color, e.g. (0.1, 0.2, 0.3, 1.0)
```

At last, once you're satisfied with how you're scene currently is, you can render it ('save the image' in layman terms):

```python
bpy.context.scene.render.filepath = os.path.join('<your-path>', '<filename>.png')
bpy.ops.render.render(write_still=True)
```

**Note:** The speed at which each data sample is rendered depends on the Graphic card you have (or the lack of, God have mercy on your soul if so).

# 3. Deep learning model

TODO

![Model architecture](https://github.com/RubenPants/BlenderCNN/blob/main/images/architecture.png?raw=true)

# 4. Training

10.000 samples total

![TensorBoard overview](https://github.com/RubenPants/BlenderCNN/blob/main/images/tensorboard.png?raw=true)

# 5. Results

TODO

![Results side-by-side comparison](https://github.com/RubenPants/BlenderCNN/blob/main/images/sample_combined.png?raw=true)

# 6. Sources

- https://docs.blender.org/api/current/index.html 
