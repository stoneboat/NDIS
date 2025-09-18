echo "Installing Environment for dp_NDIS [Main Environment]"

# Set the virtual environment path using the python-venv directory as prefix
mkdir -p /tmp/python-venv

# Create the virtual environment for the main project (dp_private_learning)
MAIN_VENV_PATH="/tmp/python-venv/dp_NDIS_venv"

if [ -d "$MAIN_VENV_PATH" ]; then
    echo "Virtual environment 'dp_NDIS_venv' already exists in $MAIN_VENV_PATH."
else
    echo "Creating virtual environment 'dp_NDIS_venv' in $MAIN_VENV_PATH..."
    python3.11 -m venv "$MAIN_VENV_PATH"
fi

source "$MAIN_VENV_PATH/bin/activate"

pip install --upgrade pip

# Install packages from requirements.txt
pip install -r requirements.txt

# Register the kernel
python -m ipykernel install --user --name=dp-NDIS-env --display-name "dp-NDIS-env"

# Create the custom kernel spec directory
KERNEL_DIR=~/.local/share/jupyter/kernels/dp-NDIS-env
mkdir -p "$KERNEL_DIR"

# Get absolute path for the virtual environment
ABSOLUTE_VENV_PATH=$(realpath "$MAIN_VENV_PATH")

# Write the kernel.json
cat > "$KERNEL_DIR/kernel.json" <<EOL
{
  "argv": [
    "$ABSOLUTE_VENV_PATH/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (dp-NDIS-env)",
  "language": "python"
}
EOL

deactivate