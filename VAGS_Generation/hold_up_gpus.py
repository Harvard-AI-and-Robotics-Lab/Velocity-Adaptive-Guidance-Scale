import torch
import time
import sys

# Define the size of memory chunks to allocate in MB
CHUNK_SIZE_MB = 256

def hold_gpu_memory():
    """
    Identifies all available GPUs and allocates memory on them until they are full.
    Then, it holds the memory until the script is manually stopped (Ctrl+C).
    """
    if not torch.cuda.is_available():
        print("🔴 No CUDA-enabled GPUs found. Exiting.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"✅ Found {num_gpus} CUDA-enabled GPU(s).")

    gpus_held = []

    for i in range(num_gpus):
        try:
            print(f"\n--- Targeting GPU {i}: {torch.cuda.get_device_name(i)} ---")
            torch.cuda.set_device(i)

            # Get free memory info
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            print(f"Initial free memory: {free_mem / 1024**3:.2f} GB / {total_mem / 1024**3:.2f} GB")

            allocated_tensors_gpu = []
            allocated_mb = 0
            
            # Calculate chunk size in bytes for a float32 tensor
            chunk_bytes = CHUNK_SIZE_MB * 1024 * 1024
            num_elements = chunk_bytes // 4  # float32 is 4 bytes

            while True:
                try:
                    # Allocate a chunk of memory
                    tensor = torch.ones(num_elements, dtype=torch.float32, device=f'cuda:{i}')
                    allocated_tensors_gpu.append(tensor)
                    allocated_mb += CHUNK_SIZE_MB
                    
                    # Progress update without flooding the console
                    sys.stdout.write(f"\rAllocated {allocated_mb / 1024:.2f} GB on GPU {i}...")
                    sys.stdout.flush()

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n✅ GPU {i} is now full.")
                        break
                    else:
                        raise e # Re-raise other runtime errors
            
            gpus_held.append(allocated_tensors_gpu)

        except Exception as e:
            print(f"\n❌ An error occurred on GPU {i}: {e}")

    print(f"\n🚀 All {len(gpus_held)} GPU(s) are now held.")
    print("The script is running. Press Ctrl+C to release memory and exit.")

    try:
        while True:
            time.sleep(3600) # Sleep for a long time to keep script alive
    except KeyboardInterrupt:
        print("\n\nSIGINT received. Releasing memory and shutting down...")
        # The script will now exit, and Python's garbage collector
        # will release the memory held by the tensors.
        print("✅ Memory released. Goodbye!")

if __name__ == "__main__":
    hold_gpu_memory()