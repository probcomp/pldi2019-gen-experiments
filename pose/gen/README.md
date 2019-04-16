# Body pose inference in Gen

## Build

Build PyCall with the `PYTHON` environment variable set to the absolute path of the Python executable in a virtualenv with TensorFlow==1.12 and rpyc installed, e.g.:
```
PYTHON=/home/my_tf_env/bin/python JULIA_PROJECT=. -e 'using Pkg; Pkg.build("PyCall")'
```

## Run

1. Start a blender server to serve the depth images on port 62000 (this port number is defined in model.jl)
```
xvfb-run --auto-servernum -server-num=1 blender -b ../HumanKTH.decimated.blend -P ../blender_depth_server.py -- 62000
```

2. Start a blender server to serve wireframe images on port 62001 (this port number is defined in model.jl)
```
xvfb-run --auto-servernum -server-num=1 blender -b ../HumanKTH.decimated.blend -P ../blender_depth_server.py -- 62001
```

3. To train the network:
```
JULIA_PROJECT=. julia train.jl
```

4. To run importance sampling on a test image:
```
JULIA_PROJECT=. julia importance.jl
```
