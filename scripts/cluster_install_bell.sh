echo "Installing Environment for dp_NDIS [Main Environment]"

module load anaconda

# Initialize conda in this non-interactive shell
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "conda not found after loading anaconda3 module" >&2
    exit 1
fi

# Set the conda environment path using the python-venv directory as prefix
mkdir -p /tmp/python-venv

CONDA_ENV_PATH="/tmp/python-venv/dp_NDIS_venv"

if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Conda env 'dp_NDIS_venv' already exists in $CONDA_ENV_PATH."
else
    echo "Creating conda env 'dp_NDIS_venv' in $CONDA_ENV_PATH..."
    conda create --prefix "$CONDA_ENV_PATH" python=3.11 -y || { echo "conda create failed" >&2; exit 1; }
fi

# Activate the conda environment
conda activate "$CONDA_ENV_PATH"

# Install conda packages first (C libraries)
# Note: chi2comb needs both C library (conda) and Python bindings (pip)
echo "Installing conda packages (C libraries)..."
conda install -c conda-forge chi2comb -y || { echo "conda install chi2comb failed" >&2; exit 1; }

# Upgrade pip
pip install --upgrade pip

# Install packages from requirements.txt
echo "Installing pip packages..."
pip install -r requirements.txt || { echo "pip install failed" >&2; exit 1; }

# Register the kernel
python -m ipykernel install --user --name=dp-NDIS-env --display-name "dp-NDIS-env" || { echo "kernel registration failed" >&2; exit 1; }

# Create the custom kernel spec directory
KERNEL_DIR=~/.local/share/jupyter/kernels/dp-NDIS-env
mkdir -p "$KERNEL_DIR"

# Write the kernel.json
cat > "$KERNEL_DIR/kernel.json" <<EOL
{
  "argv": [
    "$CONDA_ENV_PATH/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (dp-NDIS-env)",
  "language": "python"
}
EOL

# Deactivate conda environment
conda deactivate