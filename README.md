# (Out Dated) ComfyUI-Geeky-LatentSyncWrapper 1.5 (Mediapipe isn't compatible with some changes, will need to downgrade python version of portable comfyUI and remove xformers). 

Unofficial **optimized and enhanced** fork of [LatentSync 1.5](https://github.com/bytedance/LatentSync) implementation for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on Windows and WSL 2.0.

This node provides advanced lip-sync capabilities in ComfyUI using ByteDance's LatentSync 1.5 model with **significantly improved performance, memory efficiency, and stability**. This fork focuses on speed, reliability, and conflict-free coexistence with other LatentSync implementations.

<img width="1893" height="1357" alt="workflow (4)" src="https://github.com/user-attachments/assets/51d3b98a-e549-4c71-a2f2-2af5cc52f291" />

## Why This Fork? Performance & Stability

**🚀 Much Faster Performance**: This implementation is significantly faster than other versions and eliminates OOM (Out of Memory) errors that plague other implementations.

**🧠 Better Memory Management**: Intelligent VRAM usage with user-selectable settings (high/medium/low) and automatic cleanup prevents memory issues.

**🔒 Conflict-Free**: Can be installed alongside other LatentSync implementations without interference - uses isolated paths and unique node names.

**⚡ LatentSync 1.5 vs 1.6**: We use LatentSync 1.5 instead of 1.6 because:
- **More Stable**: 1.5 has proven stability and reliability in production use
- **Better Performance**: 1.5 runs faster and uses less VRAM than 1.6
- **No Manual Downloads**: 1.5 models download automatically, unlike 1.6's private repository requirements
- **Fewer Dependencies**: Simpler, more reliable dependency chain

## What's New in This Optimized Fork?

### Performance Enhancements
1. **Advanced Memory Management**: Intelligent VRAM allocation with user-selectable modes
2. **Faster Processing**: Optimized batch processing and GPU utilization
3. **No OOM Errors**: Comprehensive memory cleanup and management
4. **Mixed Precision Support**: Automatic FP16 optimization when beneficial

### User Experience Improvements  
5. **Single Image Support**: Process individual images with LatentSync
6. **Batch Image Processing**: Process multiple images efficiently
7. **Smart Temp Management**: Isolated temporary directories prevent conflicts
8. **Better Error Handling**: Robust error recovery and informative messages

### Compatibility Features
9. **Conflict-Free Installation**: Can coexist with ShmuelRonen's implementation
10. **Unique Node Names**: "Geeky" prefixed nodes prevent naming conflicts
11. **Isolated Model Storage**: Uses `geeky_checkpoints/` directory
12. **Automatic Path Management**: Handles compatibility transparently

## Original LatentSync 1.5 Features

1. **Temporal Layer Improvements**: Corrected implementation provides significantly improved temporal consistency compared to version 1.0
2. **Better Chinese Language Support**: Performance on Chinese videos is substantially improved through additional training data
3. **Reduced VRAM Requirements**: Optimized to run on 20GB VRAM (RTX 3090 compatible) through various optimizations:
   - Gradient checkpointing in U-Net, VAE, SyncNet and VideoMAE
   - Native PyTorch FlashAttention-2 implementation (no xFormers dependency)
   - More efficient CUDA cache management
   - Focused training of temporal and audio cross-attention layers only
4. **Code Optimizations**:
   - Removed dependencies on xFormers and Triton
   - Upgraded to diffusers 0.32.2

## Compatibility with Other LatentSync Nodes

This repository can be installed alongside ShmuelRonen's ComfyUI-LatentSyncWrapper **without conflicts**:

- ✅ **Different node names**: Geeky nodes use "Geeky" prefix ("Geeky LatentSync 1.5 (Optimized)")
- ✅ **Separate checkpoints**: Uses `geeky_checkpoints/` directory  
- ✅ **Independent models**: Downloads to isolated paths
- ✅ **No shared resources**: Completely separate from other LatentSync implementations
- ✅ **Isolated temp directories**: Prevents interference with other nodes

Both repositories can coexist and users can choose which nodes to use based on their performance needs.

## Prerequisites

Before installing this node, you must install the following in order:

1. [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed and working

2. FFmpeg installed on your system:
   - Windows: Download from [here](https://github.com/BtbN/FFmpeg-Builds/releases) and add to system PATH

## Installation

Only proceed with installation after confirming all prerequisites are installed and working.

1. Clone this repository into your ComfyUI custom_nodes directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/GeekyGhost/ComfyUI-Geeky-LatentSyncWrapper.git
cd ComfyUI-Geeky-LatentSyncWrapper
pip install -r requirements.txt
```

## Required Dependencies
```
diffusers>=0.32.2
transformers
huggingface-hub
omegaconf
einops
opencv-python
mediapipe
face-alignment
decord
ffmpeg-python
safetensors
soundfile
```

## Note on Model Downloads

On first use, the node will **automatically download** required model files from HuggingFace:
- LatentSync 1.5 UNet model (~5GB)
- Whisper model for audio processing (~1.6GB)
- All models download automatically - no manual intervention required
- Models are stored in isolated `geeky_checkpoints/` directory

### Checkpoint Directory Structure

After successful installation and model download, your checkpoint directory structure will look like this:

```
./geeky_checkpoints/
|-- .cache/
|-- auxiliary/
|-- whisper/
|   `-- tiny.pt
|-- config.json
|-- latentsync_unet.pt  (~5GB)
|-- stable_syncnet.pt   (~1.6GB)
```

Make sure all these files are present for proper functionality. The main model files are:
- `latentsync_unet.pt`: The primary LatentSync 1.5 model
- `stable_syncnet.pt`: The SyncNet model for lip-sync supervision
- `whisper/tiny.pt`: The Whisper model for audio processing

## Usage

### For Videos:
1. Select an input video file with a video loader
2. Load an audio file using ComfyUI audio loader
3. (Optional) Set a seed value for reproducible results
4. (Optional) Adjust the lips_expression parameter to control lip movement intensity
5. (Optional) Modify the inference_steps parameter to balance quality and speed
6. (Optional) Choose VRAM usage setting based on your GPU
7. Connect to the **Geeky LatentSync 1.5 (Optimized)** node
8. Run the workflow

### For Single Images:
1. Load a single image using ComfyUI's image loader
2. Load an audio file using ComfyUI audio loader
3. Connect to the **Geeky LatentSync 1.5 (Optimized)** node
4. Adjust parameters as needed
5. Run the workflow

### For Batch Images:
1. Load multiple images using ComfyUI's batch image loader or image list to batch node
2. Load an audio file using ComfyUI audio loader
3. Connect to the **Geeky LatentSync 1.5 (Optimized)** node
4. Adjust parameters as needed
5. Run the workflow

The processed video or images will be saved in ComfyUI's output directory.

### Node Parameters:
- `images`: Input image(s) - supports single images, video frames, or batch processing
- `audio`: Audio input from ComfyUI audio loader
- `seed`: Random seed for reproducible results (default: 1247)
- `lips_expression`: Controls the expressiveness of lip movements (default: 1.5)
  - Higher values (2.0-3.0): More pronounced lip movements, better for expressive speech
  - Lower values (1.0-1.5): Subtler lip movements, better for calm speech
  - This parameter affects the model's guidance scale, balancing between natural movement and lip sync accuracy
- `inference_steps`: Number of denoising steps during inference (default: 20)
  - Higher values (30-50): Better quality results but slower processing
  - Lower values (10-15): Faster processing but potentially lower quality
  - The default of 20 usually provides a good balance between quality and speed
- `vram_usage`: **NEW** - Choose memory usage profile (default: medium)
  - **High**: Maximum performance, uses 95% VRAM, enables all optimizations
  - **Medium**: Balanced performance, uses 85% VRAM, good for most users
  - **Low**: Conservative usage, uses 75% VRAM, for systems with limited memory

### Available Nodes:
- **Geeky LatentSync 1.5 (Optimized)**: Main lip-sync processing node
- **Geeky Video Length Adjuster (Fast)**: Utility node for video/audio length matching

### Tips for Better Results:
- **Performance**: Start with "medium" VRAM usage and increase to "high" if you have sufficient GPU memory
- **Quality**: For speeches or presentations, try increasing lips_expression to 2.0-2.5
- **Efficiency**: For quick previews, use "low" VRAM setting with 10-15 inference steps
- **Stability**: This implementation handles single images automatically by duplicating frames to match audio length
- **Memory**: The optimized memory management prevents OOM errors even with long audio clips

## Performance Comparison

| Feature | This Fork (Geeky) | Original Implementation |
|---------|-------------------|------------------------|
| OOM Errors | ❌ None | ✅ Frequent |
| Processing Speed | 🚀 Much Faster | 🐌 Slower |
| Memory Usage | 🧠 Optimized | 💾 High |
| VRAM Settings | ✅ 3 Modes | ❌ Fixed |
| Conflict-Free | ✅ Yes | ❌ No |
| Auto Downloads | ✅ Yes | ⚠️ Manual (1.6) |

## Known Limitations

- Works best with clear, frontal face images/videos
- Currently does not support anime/cartoon faces
- Video should be at 25 FPS (will be automatically converted)
- Face should be visible throughout the image/video
- Single images are automatically extended to match audio duration

## Troubleshooting

### Common Issues:
1. **"Geeky model checkpoints already exist"**: This is normal - models are cached for faster startup
2. **Memory errors**: Try lowering VRAM usage setting from high → medium → low
3. **Slow performance**: Ensure you're using a CUDA-compatible GPU and try "high" VRAM setting
4. **Node not appearing**: Restart ComfyUI after installation and refresh your browser

## Credits

This optimized fork is based on:
- [LatentSync 1.5](https://github.com/bytedance/LatentSync) by ByteDance Research
- [ComfyUI-LatentSyncWrapper](https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper) by ShmuelRonen
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

Special thanks to the original developers for their groundbreaking work. This fork focuses on performance optimization, memory efficiency, and user experience improvements.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
