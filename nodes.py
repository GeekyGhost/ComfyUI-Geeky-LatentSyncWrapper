import os
import tempfile
import uuid
import sys
import shutil
import time

# Global model cache to avoid reloading models - Using unique name to avoid conflicts
_GEEKY_MODEL_CACHE = {}

# Function to check for potential conflicts with other LatentSync implementations
def check_for_conflicts():
    """Check if other LatentSync implementations might conflict"""
    try:
        import folder_paths
        custom_nodes_dir = folder_paths.get_folder_paths("custom_nodes")[0]
        original_path = os.path.join(custom_nodes_dir, "ComfyUI-LatentSyncWrapper")
        
        if os.path.exists(original_path):
            print("[Geeky LatentSync] Detected ComfyUI-LatentSyncWrapper - using isolated paths to avoid conflicts")
            return True
    except:
        pass
    return False

# Function to find ComfyUI directories
def get_comfyui_temp_dir():
    """Dynamically find the ComfyUI temp directory"""
    # First check using folder_paths if available
    try:
        import folder_paths
        comfy_dir = os.path.dirname(os.path.dirname(os.path.abspath(folder_paths.__file__)))
        temp_dir = os.path.join(comfy_dir, "temp")
        return temp_dir
    except:
        pass
    
    # Try to locate based on current script location
    try:
        # This script is likely in a ComfyUI custom nodes directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up until we find the ComfyUI directory
        potential_dir = current_dir
        for _ in range(5):  # Limit to 5 levels up
            if os.path.exists(os.path.join(potential_dir, "comfy.py")):
                return os.path.join(potential_dir, "temp")
            potential_dir = os.path.dirname(potential_dir)
    except:
        pass
    
    # Return None if we can't find it
    return None

# Function to clean up any ComfyUI temp directories
def cleanup_comfyui_temp_directories():
    """Find and clean up any ComfyUI temp directories"""
    comfyui_temp = get_comfyui_temp_dir()
    if not comfyui_temp:
        print("Could not locate ComfyUI temp directory")
        return
    
    comfyui_base = os.path.dirname(comfyui_temp)
    
    # Check for the main temp directory
    if os.path.exists(comfyui_temp):
        try:
            shutil.rmtree(comfyui_temp)
            print(f"Removed ComfyUI temp directory: {comfyui_temp}")
        except Exception as e:
            print(f"Could not remove {comfyui_temp}: {str(e)}")
            # If we can't remove it, try to rename it
            try:
                backup_name = f"{comfyui_temp}_backup_{uuid.uuid4().hex[:8]}"
                os.rename(comfyui_temp, backup_name)
                print(f"Renamed {comfyui_temp} to {backup_name}")
            except:
                pass
    
    # Find and clean up any backup temp directories
    try:
        all_directories = [d for d in os.listdir(comfyui_base) if os.path.isdir(os.path.join(comfyui_base, d))]
        for dirname in all_directories:
            if dirname.startswith("temp_backup_"):
                backup_path = os.path.join(comfyui_base, dirname)
                try:
                    shutil.rmtree(backup_path)
                    print(f"Removed backup temp directory: {backup_path}")
                except Exception as e:
                    print(f"Could not remove backup dir {backup_path}: {str(e)}")
    except Exception as e:
        print(f"Error cleaning up temp directories: {str(e)}")

def get_unique_temp_path(suffix=""):
    """Generate unique temp paths for geeky wrapper"""
    return os.path.join(tempfile.gettempdir(), f"geeky_latentsync_{uuid.uuid4().hex[:8]}{suffix}")

# Create a module-level function to set up system-wide temp directory
def init_temp_directories():
    """Initialize global temporary directory settings"""
    # First clean up any existing temp directories
    cleanup_comfyui_temp_directories()
    
    # Generate a unique base directory for this module
    system_temp = tempfile.gettempdir()
    unique_id = str(uuid.uuid4())[:8]
    temp_base_path = os.path.join(system_temp, f"geeky_latentsync_{unique_id}")
    os.makedirs(temp_base_path, exist_ok=True)
    
    # Create a persistent model cache directory
    model_cache_dir = os.path.join(system_temp, "geeky_latentsync_model_cache")
    os.makedirs(model_cache_dir, exist_ok=True)
    # Link it into our temp directory for convenience
    try:
        os.symlink(model_cache_dir, os.path.join(temp_base_path, "model_cache"), target_is_directory=True)
    except (OSError, NotImplementedError):
        # If symlinks aren't supported, just use the cache dir directly
        shutil.copytree(model_cache_dir, os.path.join(temp_base_path, "model_cache"), dirs_exist_ok=True)
    
    # Override environment variables that control temp directories
    os.environ['TMPDIR'] = temp_base_path
    os.environ['TEMP'] = temp_base_path
    os.environ['TMP'] = temp_base_path
    
    # Force Python's tempfile module to use our directory
    tempfile.tempdir = temp_base_path
    
    # Final check for ComfyUI temp directory
    comfyui_temp = get_comfyui_temp_dir()
    if comfyui_temp and os.path.exists(comfyui_temp):
        try:
            shutil.rmtree(comfyui_temp)
            print(f"Removed ComfyUI temp directory: {comfyui_temp}")
        except Exception as e:
            print(f"Could not remove {comfyui_temp}, trying to rename: {str(e)}")
            try:
                backup_name = f"{comfyui_temp}_backup_{unique_id}"
                os.rename(comfyui_temp, backup_name)
                print(f"Renamed {comfyui_temp} to {backup_name}")
                # Try to remove the renamed directory as well
                try:
                    shutil.rmtree(backup_name)
                    print(f"Removed renamed temp directory: {backup_name}")
                except:
                    pass
            except:
                print(f"Failed to rename {comfyui_temp}")
    
    print(f"Set up system temp directory: {temp_base_path}")
    return temp_base_path

# Function to clean up everything when the module exits
def module_cleanup():
    """Clean up all resources when the module is unloaded"""
    global MODULE_TEMP_DIR, _GEEKY_MODEL_CACHE
    
    # Clear model cache references to free memory
    _GEEKY_MODEL_CACHE.clear()
    
    # Clean up temp directory except model cache
    if MODULE_TEMP_DIR and os.path.exists(MODULE_TEMP_DIR):
        try:
            for item in os.listdir(MODULE_TEMP_DIR):
                if item != "model_cache":
                    path = os.path.join(MODULE_TEMP_DIR, item)
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        try:
                            os.remove(path)
                        except:
                            pass
            print(f"Cleaned up module temp directory (preserving model cache)")
        except:
            pass
    
    # Do a final sweep for any ComfyUI temp directories
    cleanup_comfyui_temp_directories()

# Call this before anything else
MODULE_TEMP_DIR = init_temp_directories()

# Register the cleanup handler to run when Python exits
import atexit
atexit.register(module_cleanup)

# Now import regular dependencies
import math
import torch
import random
import torchaudio
import folder_paths
import numpy as np
import platform
import subprocess
import importlib.util
import importlib.machinery
import argparse
from omegaconf import OmegaConf
from PIL import Image
from decimal import Decimal, ROUND_UP
import requests

# Check for conflicts with other implementations
conflict_detected = check_for_conflicts()

# Modify folder_paths module to use our temp directory
if hasattr(folder_paths, "get_temp_directory"):
    original_get_temp = folder_paths.get_temp_directory
    folder_paths.get_temp_directory = lambda: MODULE_TEMP_DIR
else:
    # Add the function if it doesn't exist
    setattr(folder_paths, 'get_temp_directory', lambda: MODULE_TEMP_DIR)

def get_cached_model(model_path, model_type, device):
    """Load model from geeky-specific cache or disk and cache it"""
    global _GEEKY_MODEL_CACHE
    cache_key = f"geeky_{model_type}_{model_path}"
    
    if cache_key in _GEEKY_MODEL_CACHE:
        # Check if the cached model is on the right device
        cached_model = _GEEKY_MODEL_CACHE[cache_key]
        model_device = next(cached_model.parameters()).device
        if str(model_device) == str(device):
            print(f"Using cached {model_type} model from Geeky cache")
            return cached_model
        else:
            print(f"Moving cached {model_type} model to {device}")
            cached_model = cached_model.to(device)
            return cached_model
    
    print(f"Loading {model_type} model from disk into Geeky cache")
    # Load the model
    model = torch.load(model_path, map_location=device)
    
    # Cache the model
    _GEEKY_MODEL_CACHE[cache_key] = model
    return model

def import_inference_script(script_path):
    """Import a Python file as a module using its file path."""
    if not os.path.exists(script_path):
        raise ImportError(f"Script not found: {script_path}")

    module_name = "geeky_latentsync_inference"  # Unique module name
    
    # Check if the module is already loaded
    if module_name in sys.modules:
        print("Using previously imported Geeky inference module")
        return sys.modules[module_name]
    
    print(f"Importing Geeky inference script from {script_path}")
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None:
        raise ImportError(f"Failed to create module spec for {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        raise ImportError(f"Failed to execute module: {str(e)}")

    return module

def check_ffmpeg():
    try:
        if platform.system() == "Windows":
            # Check if ffmpeg exists in PATH
            ffmpeg_path = shutil.which("ffmpeg.exe")
            if ffmpeg_path is None:
                # Look for ffmpeg in common locations
                possible_paths = [
                    os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "ffmpeg", "bin"),
                    os.path.join(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)"), "ffmpeg", "bin"),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg", "bin"),
                ]
                for path in possible_paths:
                    if os.path.exists(os.path.join(path, "ffmpeg.exe")):
                        # Add to PATH
                        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                        return True
                print("FFmpeg not found. Please install FFmpeg and add it to PATH")
                return False
            return True
        else:
            subprocess.run(["ffmpeg"], capture_output=True)
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found. Please install FFmpeg")
        return False

def check_and_install_dependencies():
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg is required but not found")
    required_packages = [
        'omegaconf',
        'transformers',
        'accelerate',
        'huggingface_hub',
        'einops',
        'diffusers',
        'ffmpeg-python' 
    ]
    
    # Check if we've already run this function successfully
    cache_dir = os.path.join(MODULE_TEMP_DIR, "model_cache")
    # Create the cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_marker = os.path.join(cache_dir, ".geeky_deps_installed")
    if os.path.exists(cache_marker):
        print("Geeky dependencies already verified, skipping check.")
        return
        
    def is_package_installed(package_name):
        return importlib.util.find_spec(package_name) is not None
        
    def install_package(package):
        python_exe = sys.executable
        try:
            subprocess.check_call([python_exe, '-m', 'pip', 'install', package],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            print(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {str(e)}")
            raise RuntimeError(f"Failed to install required package: {package}")
            
    missing_packages = []
    for package in required_packages:
        if not is_package_installed(package):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                install_package(package)
            except Exception as e:
                print(f"Warning: Failed to install {package}: {str(e)}")
                raise
    else:
        print("All required packages are already installed.")
    
    # Create marker file
    try:
        with open(cache_marker, 'w') as f:
            f.write(f"Geeky dependencies checked on {time.ctime()}")
    except Exception as e:
        print(f"Warning: Could not create cache marker file: {str(e)}")

def normalize_path(path):
    """Normalize path to handle spaces and special characters"""
    return os.path.normpath(path).replace('\\', '/')

def get_ext_dir(subpath=None, mkdir=False):
    """Get extension directory path, optionally with a subpath"""
    # Get the directory containing this script
    dir = os.path.dirname(os.path.abspath(__file__))
    
    # Special case for temp directories
    if subpath and ("temp" in subpath.lower() or "tmp" in subpath.lower()):
        # Use our global temp directory instead
        global MODULE_TEMP_DIR
        sub_temp = os.path.join(MODULE_TEMP_DIR, subpath)
        if mkdir and not os.path.exists(sub_temp):
            os.makedirs(sub_temp, exist_ok=True)
        return sub_temp
    
    if subpath is not None:
        dir = os.path.join(dir, subpath)

    if mkdir and not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    
    return dir

def download_model(url, save_path):
    """Download a model from a URL and save it to the specified path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Check if file already exists
    if os.path.exists(save_path):
        print(f"Model file already exists at {save_path}, skipping download.")
        return
        
    print(f"Downloading {url} to {save_path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rDownload progress: {percent:.1f}%", end="")
    print("\nDownload complete")

def pre_download_models():
    """Pre-download all required models."""
    models = {
        "s3fd-e19a316812.pth": "https://www.adrianbulat.com/downloads/python-fan/s3fd-e19a316812.pth",
        # Add other models here
    }

    cache_dir = os.path.join(MODULE_TEMP_DIR, "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check if we've already run this function successfully by creating a marker file
    cache_marker = os.path.join(cache_dir, ".geeky_cache_complete")
    if os.path.exists(cache_marker):
        print("Pre-downloaded Geeky models already exist, skipping download.")
        return
    
    for model_name, url in models.items():
        save_path = os.path.join(cache_dir, model_name)
        if not os.path.exists(save_path):
            print(f"Downloading {model_name}...")
            download_model(url, save_path)
        else:
            print(f"{model_name} already exists in Geeky cache.")
    
    # Create marker file to indicate successful completion
    with open(cache_marker, 'w') as f:
        f.write(f"Geeky cache completed on {time.ctime()}")

def setup_models():
    """Setup and pre-download all required models."""
    # Use our global temp directory
    global MODULE_TEMP_DIR
    
    # Pre-download additional models
    pre_download_models()

    # Existing setup logic for LatentSync models - using unique directory
    cur_dir = get_ext_dir()
    ckpt_dir = os.path.join(cur_dir, "geeky_checkpoints")  # Changed from "checkpoints" to "geeky_checkpoints"
    whisper_dir = os.path.join(ckpt_dir, "whisper")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(whisper_dir, exist_ok=True)

    # Create a temp_downloads directory in our system temp
    temp_downloads = os.path.join(MODULE_TEMP_DIR, "downloads")
    os.makedirs(temp_downloads, exist_ok=True)
    
    unet_path = os.path.join(ckpt_dir, "latentsync_unet.pt")
    whisper_path = os.path.join(whisper_dir, "tiny.pt")

    # Only download if the files don't already exist
    if os.path.exists(unet_path) and os.path.exists(whisper_path):
        print("Geeky model checkpoints already exist, skipping download.")
        return
        
    print("Downloading required Geeky model checkpoints... This may take a while.")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id="ByteDance/LatentSync-1.5",
                         allow_patterns=["latentsync_unet.pt", "whisper/tiny.pt"],
                         local_dir=ckpt_dir, 
                         local_dir_use_symlinks=False,
                         cache_dir=temp_downloads)
        print("Geeky model checkpoints downloaded successfully!")
    except Exception as e:
        print(f"Error downloading Geeky models: {str(e)}")
        print("\nPlease download models manually for Geeky LatentSync:")
        print("1. Visit: https://huggingface.co/chunyu-li/LatentSync")
        print("2. Download: latentsync_unet.pt and whisper/tiny.pt")
        print(f"3. Place them in: {ckpt_dir}")
        print(f"   with whisper/tiny.pt in: {whisper_dir}")
        raise RuntimeError("Geeky model download failed. See instructions above.")

class GeekyLatentSyncNode:
    def __init__(self):
        # Make sure our temp directory is the current one
        global MODULE_TEMP_DIR
        if not os.path.exists(MODULE_TEMP_DIR):
            os.makedirs(MODULE_TEMP_DIR, exist_ok=True)
        
        # Ensure ComfyUI temp doesn't exist
        comfyui_temp = get_comfyui_temp_dir()
        if comfyui_temp and os.path.exists(comfyui_temp):
            backup_name = f"{comfyui_temp}_backup_{uuid.uuid4().hex[:8]}"
            try:
                os.rename(comfyui_temp, backup_name)
            except:
                pass
        
        check_and_install_dependencies()
        setup_models()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "images": ("IMAGE",),
                    "audio": ("AUDIO", ),
                    "seed": ("INT", {"default": 1247}),
                    "lips_expression": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
                    "inference_steps": ("INT", {"default": 20, "min": 1, "max": 999, "step": 1}),
                    "vram_usage": (["high", "medium", "low"], {"default": "medium"}),
                 },}

    CATEGORY = "GeekyLatentSync"

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio") 
    FUNCTION = "inference"

    def process_batch(self, batch, use_mixed_precision=False):
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            processed_batch = batch.float() / 255.0
            if len(processed_batch.shape) == 3:
                processed_batch = processed_batch.unsqueeze(0)
            if processed_batch.shape[0] == 3:
                processed_batch = processed_batch.permute(1, 2, 0)
            if processed_batch.shape[-1] == 4:
                processed_batch = processed_batch[..., :3]
            return processed_batch

    def inference(self, images, audio, seed, lips_expression=1.5, inference_steps=20, vram_usage="medium"):
        # Add timing information
        import time
        start_time = time.time()
        
        # Use our module temp directory
        global MODULE_TEMP_DIR
        
        # Define timing checkpoint function
        def log_timing(step):
            elapsed = time.time() - start_time
            print(f"[Geeky {elapsed:.2f}s] {step}")
        
        log_timing("Starting Geeky inference")
        
        # Get GPU capabilities and memory
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_mixed_precision = False
        
        # Set VRAM usage based on user preference
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            gpu_mem_gb = gpu_mem / (1024 ** 3)
            
            # Dynamic batch size and settings based on VRAM usage preference
            if vram_usage == "high":
                BATCH_SIZE = min(32, 120 // inference_steps)
                use_mixed_precision = True
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True if hasattr(torch.backends.cuda, "matmul") else False
                torch.backends.cudnn.allow_tf32 = True
                torch.cuda.set_per_process_memory_fraction(0.95)
                print(f"Using Geeky high VRAM settings with {BATCH_SIZE} batch size")
            elif vram_usage == "medium":
                BATCH_SIZE = min(16, 80 // inference_steps)
                use_mixed_precision = True
                torch.backends.cudnn.benchmark = True
                torch.cuda.set_per_process_memory_fraction(0.85)
                print(f"Using Geeky medium VRAM settings with {BATCH_SIZE} batch size")
            else:  # low
                BATCH_SIZE = min(8, 40 // inference_steps)
                use_mixed_precision = False
                torch.cuda.set_per_process_memory_fraction(0.75)
                print(f"Using Geeky low VRAM settings with {BATCH_SIZE} batch size")
                
            # Clear GPU cache before processing
            torch.cuda.empty_cache()
        else:
            # CPU fallback settings
            BATCH_SIZE = 4
            print("No GPU detected, using CPU with minimal batch size")
        
        # Create a run-specific subdirectory in our temp directory
        run_id = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        temp_dir = os.path.join(MODULE_TEMP_DIR, f"geeky_run_{run_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Ensure ComfyUI temp doesn't exist again
        comfyui_temp = get_comfyui_temp_dir()
        if comfyui_temp and os.path.exists(comfyui_temp):
            backup_name = f"{comfyui_temp}_backup_{uuid.uuid4().hex[:8]}"
            try:
                os.rename(comfyui_temp, backup_name)
            except:
                pass
        
        temp_video_path = None
        output_video_path = None
        audio_path = None

        try:
            # Create temporary file paths in our system temp directory
            temp_video_path = os.path.join(temp_dir, f"geeky_temp_{run_id}.mp4")
            output_video_path = os.path.join(temp_dir, f"geeky_latentsync_{run_id}_out.mp4")
            audio_path = os.path.join(temp_dir, f"geeky_latentsync_{run_id}_audio.wav")
            
            # Get the extension directory
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            
            log_timing("Processing input frames")
            # Process input frames
            if isinstance(images, list):
                frames = torch.stack(images).to(device)
            else:
                frames = images.to(device)
            frames = (frames * 255).byte()

            # Process audio data to get expected frame count for a single image
            waveform = audio["waveform"].to(device)
            sample_rate = audio["sample_rate"]
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)
                
            # Check if we have a single image (either as a batch of 1 or a single 3D tensor)
            is_single_image = False
            if len(frames.shape) == 3:  # Single 3D tensor (H,W,C)
                frames = frames.unsqueeze(0)
                is_single_image = True
            elif frames.shape[0] == 1:  # Batch of 1
                is_single_image = True
                
            # If it's a single image, duplicate it to match audio duration
            if is_single_image:
                # Calculate audio duration in seconds
                audio_duration = waveform.shape[1] / sample_rate
                
                # Calculate how many frames we need at 25fps (standard for this model)
                required_frames = math.ceil(audio_duration * 25)
                
                # Duplicate the single frame to match required frame count
                # (minimum 4 frames to avoid tensor stack issues)
                required_frames = max(required_frames, 4)
                single_frame = frames[0]
                duplicated_frames = single_frame.unsqueeze(0).repeat(required_frames, 1, 1, 1)
                frames = duplicated_frames
                print(f"Geeky: Duplicated single image to create {required_frames} frames matching audio duration")

            log_timing("Processing audio")
            # Resample audio if needed
            if sample_rate != 16000:
                new_sample_rate = 16000
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=new_sample_rate
                ).to(device)
                waveform_16k = resampler(waveform)
                waveform, sample_rate = waveform_16k, new_sample_rate

            # Package resampled audio
            resampled_audio = {
                "waveform": waveform.unsqueeze(0),
                "sample_rate": sample_rate
            }
            
            log_timing("Saving temporary files")
            # Move waveform to CPU for saving
            waveform_cpu = waveform.cpu()
            torchaudio.save(audio_path, waveform_cpu, sample_rate)

            # Move frames to CPU for saving to video
            frames_cpu = frames.cpu()
            try:
                import torchvision.io as io
                io.write_video(temp_video_path, frames_cpu, fps=25, video_codec='h264')
            except TypeError:
                import av
                container = av.open(temp_video_path, mode='w')
                stream = container.add_stream('h264', rate=25)
                stream.width = frames_cpu.shape[2]
                stream.height = frames_cpu.shape[1]

                for frame in frames_cpu:
                    frame = av.VideoFrame.from_ndarray(frame.numpy(), format='rgb24')
                    packet = stream.encode(frame)
                    container.mux(packet)

                packet = stream.encode(None)
                container.mux(packet)
                container.close()
            
            # Free up memory after saving
            del frames_cpu, waveform_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            log_timing("Setting up model paths")
            # Define paths to required files and configs - using geeky_checkpoints
            inference_script_path = os.path.join(cur_dir, "scripts", "inference.py")
            config_path = os.path.join(cur_dir, "configs", "unet", "stage2.yaml")
            scheduler_config_path = os.path.join(cur_dir, "configs")
            ckpt_path = os.path.join(cur_dir, "geeky_checkpoints", "latentsync_unet.pt")  # Updated path
            whisper_ckpt_path = os.path.join(cur_dir, "geeky_checkpoints", "whisper", "tiny.pt")  # Updated path

            # Create config and args
            config = OmegaConf.load(config_path)

            # Set the correct mask image path
            mask_image_path = os.path.join(cur_dir, "latentsync", "utils", "mask.png")
            # Make sure the mask image exists
            if not os.path.exists(mask_image_path):
                # Try to find it in the utils directory directly
                alt_mask_path = os.path.join(cur_dir, "utils", "mask.png")
                if os.path.exists(alt_mask_path):
                    mask_image_path = alt_mask_path
                else:
                    print(f"Warning: Could not find mask image at expected locations")

            # Set mask path in config
            if hasattr(config, "data") and hasattr(config.data, "mask_image_path"):
                config.data.mask_image_path = mask_image_path

            args = argparse.Namespace(
                unet_config_path=config_path,
                inference_ckpt_path=ckpt_path,
                video_path=temp_video_path,
                audio_path=audio_path,
                video_out_path=output_video_path,
                seed=seed,
                inference_steps=inference_steps,
                guidance_scale=lips_expression,  # Using lips_expression for the guidance_scale
                scheduler_config_path=scheduler_config_path,
                whisper_ckpt_path=whisper_ckpt_path,
                device=device,
                batch_size=BATCH_SIZE,
                use_mixed_precision=use_mixed_precision,
                temp_dir=temp_dir,
                mask_image_path=mask_image_path
            )

            # CRITICAL FIX: Create symlink or copy to handle hardcoded paths in inference script
            old_checkpoints_dir = os.path.join(cur_dir, "checkpoints")
            if not os.path.exists(old_checkpoints_dir):
                try:
                    # Try to create a symlink first (faster)
                    os.symlink(os.path.join(cur_dir, "geeky_checkpoints"), old_checkpoints_dir)
                    print(f"Created symlink from {old_checkpoints_dir} to geeky_checkpoints")
                except (OSError, NotImplementedError):
                    # If symlinks aren't supported, copy the directory
                    shutil.copytree(os.path.join(cur_dir, "geeky_checkpoints"), old_checkpoints_dir, dirs_exist_ok=True)
                    # Create a marker file so we know this is our temporary copy
                    with open(os.path.join(old_checkpoints_dir, ".geeky_temp_copy"), 'w') as f:
                        f.write("Temporary copy created by Geeky LatentSync")
                    print(f"Copied geeky_checkpoints to {old_checkpoints_dir}")
            elif os.path.isdir(old_checkpoints_dir) and not os.path.islink(old_checkpoints_dir):
                # If checkpoints directory exists and is not our symlink, warn user
                print("Warning: Found existing 'checkpoints' directory. This might conflict with other LatentSync implementations.")
                print("Geeky LatentSync will use the existing directory but recommend using separate installations.")

            # Set PYTHONPATH to include our directories 
            package_root = os.path.dirname(cur_dir)
            if package_root not in sys.path:
                sys.path.insert(0, package_root)
            if cur_dir not in sys.path:
                sys.path.insert(0, cur_dir)

            # Clean GPU cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Check and prevent ComfyUI temp creation again
            comfyui_temp = get_comfyui_temp_dir()
            if comfyui_temp and os.path.exists(comfyui_temp):
                try:
                    os.rename(comfyui_temp, f"{comfyui_temp}_backup_{uuid.uuid4().hex[:8]}")
                except:
                    pass

            log_timing("Importing inference module")
            # Import the inference module
            inference_module = import_inference_script(inference_script_path)
            
            # Monkey patch any temp directory functions in the inference module
            if hasattr(inference_module, 'get_temp_dir'):
                inference_module.get_temp_dir = lambda *args, **kwargs: temp_dir
                
            # Create subdirectories that the inference module might expect
            inference_temp = os.path.join(temp_dir, "temp")
            os.makedirs(inference_temp, exist_ok=True)
            
            log_timing("Running Geeky inference")
            # Run inference
            inference_module.main(config, args)

            log_timing("Processing output")
            # Clean GPU cache after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Verify output file exists
            if not os.path.exists(output_video_path):
                raise FileNotFoundError(f"Output video not found at: {output_video_path}")
            
            # Read the processed video - ensure it's loaded as CPU tensor
            import torchvision.io as io
            processed_frames = io.read_video(output_video_path, pts_unit='sec')[0]
            processed_frames = processed_frames.float() / 255.0

            # Ensure audio is on CPU before returning
            if torch.cuda.is_available():
                if hasattr(resampled_audio["waveform"], 'device') and resampled_audio["waveform"].device.type == 'cuda':
                    resampled_audio["waveform"] = resampled_audio["waveform"].cpu()
                if hasattr(processed_frames, 'device') and processed_frames.device.type == 'cuda':
                    processed_frames = processed_frames.cpu()

            total_time = time.time() - start_time
            print(f"Geeky total processing time: {total_time:.2f}s")
            
            return (processed_frames, resampled_audio)

        except Exception as e:
            print(f"Error during Geeky inference: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Clean up the temporary symlink/copy created for inference script compatibility
            try:
                old_checkpoints_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
                if os.path.exists(old_checkpoints_dir):
                    if os.path.islink(old_checkpoints_dir):
                        os.unlink(old_checkpoints_dir)
                        print("Cleaned up temporary symlink")
                    elif os.path.isdir(old_checkpoints_dir):
                        # Only remove if it's our temporary copy (check if it contains geeky files)
                        geeky_marker = os.path.join(old_checkpoints_dir, ".geeky_temp_copy")
                        if os.path.exists(geeky_marker):
                            shutil.rmtree(old_checkpoints_dir)
                            print("Cleaned up temporary copy")
            except:
                pass  # Ignore cleanup errors
                
            # Only remove temporary files if successful (keep for debugging if failed)
            try:
                # Clean up temporary files individually
                for path in [temp_video_path, output_video_path, audio_path]:
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass
            except:
                pass  # Ignore cleanup errors

            # Final GPU cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class GeekyVideoLengthAdjuster:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "mode": (["normal", "pingpong", "loop_to_audio"], {"default": "normal"}),
                "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0}),
                "silent_padding_sec": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 3.0, "step": 0.1}),
            }
        }

    CATEGORY = "GeekyLatentSync"
    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "adjust"

    def adjust(self, images, audio, mode, fps=25.0, silent_padding_sec=0.5):
        waveform = audio["waveform"].squeeze(0)
        sample_rate = int(audio["sample_rate"])
        original_frames = [images[i] for i in range(images.shape[0])] if isinstance(images, torch.Tensor) else images.copy()

        if mode == "normal":
            # Add silent padding to the audio and then trim video to match
            audio_duration = waveform.shape[1] / sample_rate
            
            # Add silent padding to the audio
            silence_samples = math.ceil(silent_padding_sec * sample_rate)
            silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
            padded_audio = torch.cat([waveform, silence], dim=1)
            
            # Calculate required frames based on the padded audio
            padded_audio_duration = (waveform.shape[1] + silence_samples) / sample_rate
            required_frames = int(padded_audio_duration * fps)
            
            if len(original_frames) > required_frames:
                # Trim video frames to match padded audio duration
                adjusted_frames = original_frames[:required_frames]
            else:
                # If video is shorter than padded audio, keep all video frames
                # and trim the audio accordingly
                adjusted_frames = original_frames
                required_samples = int(len(original_frames) / fps * sample_rate)
                padded_audio = padded_audio[:, :required_samples]
            
            return (
                torch.stack(adjusted_frames),
                {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
            )

        elif mode == "pingpong":
            video_duration = len(original_frames) / fps
            audio_duration = waveform.shape[1] / sample_rate
            if audio_duration <= video_duration:
                required_samples = int(video_duration * sample_rate)
                silence = torch.zeros((waveform.shape[0], required_samples - waveform.shape[1]), dtype=waveform.dtype)
                adjusted_audio = torch.cat([waveform, silence], dim=1)

                return (
                    torch.stack(original_frames),
                    {"waveform": adjusted_audio.unsqueeze(0), "sample_rate": sample_rate}
                )

            else:
                silence_samples = math.ceil(silent_padding_sec * sample_rate)
                silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
                padded_audio = torch.cat([waveform, silence], dim=1)
                total_duration = (waveform.shape[1] + silence_samples) / sample_rate
                target_frames = math.ceil(total_duration * fps)
                reversed_frames = original_frames[::-1][1:-1]  # Remove endpoints
                frames = original_frames + reversed_frames
                while len(frames) < target_frames:
                    frames += frames[:target_frames - len(frames)]
                return (
                    torch.stack(frames[:target_frames]),
                    {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
                )

        elif mode == "loop_to_audio":
            # Add silent padding then simple loop
            silence_samples = math.ceil(silent_padding_sec * sample_rate)
            silence = torch.zeros((waveform.shape[0], silence_samples), dtype=waveform.dtype)
            padded_audio = torch.cat([waveform, silence], dim=1)
            total_duration = (waveform.shape[1] + silence_samples) / sample_rate
            target_frames = math.ceil(total_duration * fps)

            frames = original_frames.copy()
            while len(frames) < target_frames:
                frames += original_frames[:target_frames - len(frames)]
            
            return (
                torch.stack(frames[:target_frames]),
                {"waveform": padded_audio.unsqueeze(0), "sample_rate": sample_rate}
            )

# Node Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeekyLatentSyncNode": GeekyLatentSyncNode,
    "GeekyVideoLengthAdjuster": GeekyVideoLengthAdjuster,
}

# Display Names for ComfyUI - Clear distinction from original
NODE_DISPLAY_NAME_MAPPINGS = {
    "GeekyLatentSyncNode": "Geeky LatentSync 1.5 (Optimized)",
    "GeekyVideoLengthAdjuster": "Geeky Video Length Adjuster (Fast)",
}
