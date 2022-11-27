# Position Based Fluids

An implementation of [Position Based Fluids](https://mmacklin.com/pbf_sig_preprint.pdf) by Macklin and Mueller.

This was my final project in CSCE 450: Computer Animation at Texas A&M with Dr. Shinjiro Sueda.

The base rendering code is based off of Dr. Sueda's code provided in class.

## Build
- Only tested on Arch Linux
  - Does not work on Windows via WSL2 & WSLg due to [CUDA-OpenGL interop not being implemented](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
- Dependencies: CUDA, GLEW, GLFW, GLM
  - `pacman -S cmake cuda cuda-toolkit glew glfw glm`

## Usage
- `SPACE` to toggle simulation
- Orbit controls with mouse
  - `CTRL` translate
  - `SHIFT` zoom
- Update simulation parameters in `CUDA/Parameters.cuh` and recompile. (runtime config wip)

## TODO
These are things I wanted to implement for this project, but ran out of time before the deadline.
- Collision with surfaces defined via SDF
  - Mesh to SDF conversion
- Implement Yu and Turk's paper on particle fluid surface reconstruction
- Screen-space fluid rendering

## References
- Macklin, M., and Mueller, M. -- [Position Based Fluids](https://mmacklin.com/pbf_sig_preprint.pdf), SIGGRAPH 2013
- Hoetzlin R. C. -- [Fast Fixed-Radius Nearest Neighbors: Interactive Million-Particle Fluids](https://web.archive.org/web/20160304053353/http://on-demand.gputechconf.com/gtc/2014/presentations/S4117-fast-fixed-radius-nearest-neighbor-gpu.pdf), GTC 2014
- Yu, J., and Turk, G. -- [Reconstructing Surfaces of Particle-Based Fluids
Using Anisotropic Kernels](https://faculty.cc.gatech.edu/~turk/my_papers/sph_surfaces.pdf), SIGGRAPH 2010
- Green, S. -- [Screen Space Fluid Rendering for Games](https://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf), GDC 2010
- [CUDA - OpenGL interoperability](https://github.com/NVIDIA/cuda-samples/tree/3f1c50965017932fc81e6d94a3fc9e04c105b312/Samples/5_Domain_Specific/simpleGL)
