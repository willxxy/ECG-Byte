import torch
import sys

def check_pytorch_installation():
    print("PyTorch Installation Check")
    print("==========================")

    print(f"PyTorch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print("\nCUDA Information:")
        print("-----------------")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device ID: {current_device}")
        
        device = torch.device(f'cuda:{current_device}')
        print(f"Current CUDA device: {device}")
        
        device_name = torch.cuda.get_device_name(current_device)
        print(f"CUDA device name: {device_name}")

        print("\nTensor Operations:")
        print("------------------")
        try:
            x = torch.tensor(1).to(device)
            print(f"Moved tensor: {x}")

            y = torch.tensor(1, device=device)
            print(f"Created tensor: {y}")

            if x.device.type == 'cuda' and y.device.type == 'cuda':
                print("\nSuccess: PyTorch is correctly installed with CUDA support!")
            else:
                print("\nWarning: Tensors were not properly moved to CUDA devices.")
        except Exception as e:
            print(f"\nError during tensor operations: {e}")
    else:
        print("\nNote: CUDA is not available. PyTorch will run on CPU only.")

    print("\nSystem Information:")
    print("-------------------")
    print(f"Python version: {sys.version}")
    print(f"Operating System: {sys.platform}")

if __name__ == "__main__":
    check_pytorch_installation()