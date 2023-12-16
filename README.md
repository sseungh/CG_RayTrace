# OpenGL SIMPLE RayTrace

## Conda env
- conda create -n cg python=3.8
- conda activate cg
- conda install numpy pyopengl freeglut
- conda install Pillow
- conda install opencv

# Requirement
- intel graphic card (unless occur opengl error=1281)

# Test procedure
1. Add all the dependencies. There might be more than written above.
2. Run python file_name.py
3. press l key to set light
4. left click and drag to set viewpoint
5. press r key to render

# Comparison
- Ray shooting like bullets 0.1 fly sphere
![img](https://raw.githubusercontent.com/sseungh/OpenGL_Simple_RayTrace/main/Images/1702739799305.png)
![img](https://raw.githubusercontent.com/sseungh/OpenGL_Simple_RayTrace/main/Images/1702739797018.png)
![img](https://raw.githubusercontent.com/sseungh/OpenGL_Simple_RayTrace/main/Images/1702739798215.png)
- Ray shooting to glass
![img](https://raw.githubusercontent.com/sseungh/OpenGL_Simple_RayTrace/main/Images/1702739800431.png)
- Calculate ray to sphere
![img](https://raw.githubusercontent.com/sseungh/OpenGL_Simple_RayTrace/main/Images/1702739793742.png)
![img](https://raw.githubusercontent.com/sseungh/OpenGL_Simple_RayTrace/main/Images/1702739792674.png)
![img](https://raw.githubusercontent.com/sseungh/OpenGL_Simple_RayTrace/main/Images/1702739793742.png)
- Calculate ray to box
![img](https://raw.githubusercontent.com/sseungh/OpenGL_Simple_RayTrace/main/Images/1702739791500.png)
![img](https://raw.githubusercontent.com/sseungh/OpenGL_Simple_RayTrace/main/Images/1702739788905.png)
![img](https://raw.githubusercontent.com/sseungh/OpenGL_Simple_RayTrace/main/Images/1702739790314.png)
