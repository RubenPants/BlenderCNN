# 0. Table of Contents

1. [Project description](#1. Project description)
2. [Blender](#2. Blender)
   1. [Scene](#2.1. Scene)
   2. [Code](#2.2. Code)
3. [Deep learning model](#3. Deep learning model)
4. [Training](#4. Training)
5. [Results](#5. Results)
6. [Sources](#6. Sources)

# 1. Project description

The goal of the project is to generate images based on vectoral data. In every image, a floating glowing ball is shown in a dark room. There are two aspects that vary in every image, this being the color and the position of the floating ball.

Every scene is represented by a vector of five floating numbers: 

- Three values representing the RGB-color that the glowing ball emits
- Two values representing the position (x and y axis) of the ball

Based on this vector (also called the image's feature), a Deep Learning model tries to re-generate an image of the original scene.

# 2. Blender

Since the number of images with a *floating glowing ball in a dark room* is sparse, another way to collect this data needs to be found.

## 2.1. Scene

![Blender scene](https://github.com/RubenPants/BlenderCNN/blob/main/images/blender_combined.png?raw=true)

## 2.2. Code

# 3. Deep learning model

TODO

![Model architecture](https://github.com/RubenPants/BlenderCNN/blob/main/images/architecture.png?raw=true)

# 4. Training

TODO

![TensorBoard overview](https://github.com/RubenPants/BlenderCNN/blob/main/images/tensorboard.png?raw=true)

# 5. Results

TODO

![Results side-by-side comparison](https://github.com/RubenPants/BlenderCNN/blob/main/images/sample_combined.png?raw=true)

# 6. Sources

- https://docs.blender.org/api/current/index.html 
