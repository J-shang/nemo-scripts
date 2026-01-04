# image: nvcr.io/nvidia/pytorch:25.12-py3

export BRANCH="v2.6.0"

apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging --break-system-packages
pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[all] --break-system-packages
