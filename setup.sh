# Check if the OS is MacOS
if [[ "$(uname)" == "Darwin" ]]; then
    ENV_NAME="lae-mac"
else
    ENV_NAME="lae"
fi

# Check if the environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment $ENV_NAME already exists. Activating it..."
    conda activate $ENV_NAME
else
    echo "Creating environment $ENV_NAME..."
    conda create --name $ENV_NAME python=3.8 -y
    conda activate $ENV_NAME
fi

# conda create --name lae python=3.8 -y
# conda activate lae-mac
cd mmdetection_lae
if [[ "$(uname)" == "Darwin" ]]; then
    pip3 install torch torchvision torchaudio
else
    pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
fi

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

pip install -v -e .
pip install -r requirements/multimodal.txt
pip install emoji ddd-dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install git+https://github.com/mlfoundations/open_clip.git
