## To Build

Build PyCall with the `PYTHON` environment variable set to the absolute path of the Python executable in a virtualenv with TensorFlow==1.12 and rpyc installed, e.g.:
```
virtualenv -p python3 $HOME/my_env
source $HOME/my_env/bin/activate
pip install rpyc tensorflow==1.12
deactivate
PYTHON=$HOME/my_env/bin/python JULIA_PROJECT=. -e 'using Pkg; Pkg.build("PyCall")'
```

## To run

1. Start a blender server to serve the depth images on port 62000 (this port number is defined in model.jl)
```
xvfb-run --auto-servernum -server-num=1 blender -b ../HumanKTH.decimated.blend -P ../blender_depth_server.py -- 62000
```

2. Start a blender server to serve wireframe images on port 62001 (this port number is defined in model.jl)
```
xvfb-run --auto-servernum -server-num=1 blender -b ../HumanKTH.decimated.blend -P ../blender_depth_server.py -- 62001
```
