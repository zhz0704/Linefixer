import subprocess
import sys
import re

def get_cuda_version():
    """Get CUDA version from nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        # Extract CUDA version from output
        match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
        if match:
            return match.group(1)
    except:
        pass
    return None

def map_cuda_to_pytorch(cuda_version):
    """Map CUDA version to PyTorch wheel index"""
    if cuda_version is None:
        return "cpu"
    
    major, minor = map(int, cuda_version.split('.'))
    
    return f"cu{major}{minor}"

def install_pytorch(cuda_index):
    """Install PyTorch with correct CUDA version"""
    url = f"https://download.pytorch.org/whl/{cuda_index}"
    cmd = [sys.executable, "-m", "pip", "install", 
           "torch", "torchvision", "torchaudio", 
           "--index-url", url]
    
    print(f"Installing PyTorch for {cuda_index}...")
    subprocess.run(cmd)

def install_requirements():
    """Install remaining requirements"""
    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    print("Installing remaining requirements...")
    subprocess.run(cmd)

if __name__ == "__main__":
    cuda_version = get_cuda_version()
    print(f"Detected CUDA version: {cuda_version}")
    
    cuda_index = map_cuda_to_pytorch(cuda_version)
    print(f"Using PyTorch index: {cuda_index}")
    
    install_pytorch(cuda_index)
    install_requirements()
    
    print("Installation complete!")