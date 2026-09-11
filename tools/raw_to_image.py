#!/usr/bin/env python3
"""
SGFND RAW to PNG/JPEG Converter
Reads generated_cuda_image.raw and converts to standard image formats.
"""

import struct
import sys
import os
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow", "numpy"])
    from PIL import Image
    import numpy as np


def read_sgfnd_raw(filepath):
    """Read SGFND RAW format with header: width, height, channels, bit_depth (uint32 each)"""
    with open(filepath, 'rb') as f:
        header = f.read(16)
        if len(header) != 16:
            raise ValueError("Invalid RAW file: header too small")
        
        width, height, channels, bit_depth = struct.unpack('IIII', header)
        print(f"RAW Header: {width}x{height}, {channels} channels, {bit_depth}-bit")
        
        pixel_count = width * height * channels
        data = f.read()
        
        if bit_depth == 8:
            expected = pixel_count
            dtype = np.uint8
            divisor = 255.0
        elif bit_depth == 16:
            expected = pixel_count * 2
            dtype = np.uint16
            divisor = 65535.0
        else:
            expected = pixel_count * 4
            dtype = np.float32
            divisor = 1.0
        
        if len(data) < expected:
            raise ValueError(f"RAW file truncated: expected {expected} bytes, got {len(data)}")
        
        arr = np.frombuffer(data[:expected], dtype=dtype).astype(np.float32) / divisor
        arr = arr.reshape((height, width, channels))
        
        return arr, width, height, channels


def convert_raw_to_image(raw_path, output_path=None, format='PNG'):
    """Convert SGFND RAW to PNG/JPEG"""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        print(f"Error: {raw_path} not found")
        return False
    
    if output_path is None:
        output_path = raw_path.with_suffix(f'.{format.lower()}')
    else:
        output_path = Path(output_path)
    
    try:
        arr, width, height, channels = read_sgfnd_raw(raw_path)
        
        if channels == 1:
            mode = 'L'
            img_arr = (arr[:, :, 0] * 255).astype(np.uint8)
        elif channels == 3:
            mode = 'RGB'
            img_arr = (arr * 255).astype(np.uint8)
        elif channels == 4:
            mode = 'RGBA'
            img_arr = (arr * 255).astype(np.uint8)
        else:
            print(f"Warning: Unsupported channel count {channels}, using first 3")
            mode = 'RGB'
            img_arr = (arr[:, :, :3] * 255).astype(np.uint8)
        
        img = Image.fromarray(img_arr, mode=mode)
        
        if format.upper() == 'JPEG' and mode == 'RGBA':
            bg = Image.new('RGB', img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        
        img.save(output_path, format=format.upper())
        print(f"Saved: {output_path} ({img.size[0]}x{img.size[1]}, {mode})")
        return True
        
    except Exception as e:
        print(f"Error converting {raw_path}: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python raw_to_image.py <input.raw> [output.png|output.jpg] [--format PNG|JPEG]")
        print("\nExamples:")
        print("  python raw_to_image.py generated_cuda_image.raw")
        print("  python raw_to_image.py generated_cuda_image.raw output.png")
        print("  python raw_to_image.py generated_cuda_image.raw output.jpg --format JPEG")
        sys.exit(1)
    
    raw_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    format_arg = 'PNG'
    
    for arg in sys.argv[2:]:
        if arg.startswith('--format'):
            format_arg = arg.split('=')[1] if '=' in arg else sys.argv[sys.argv.index(arg) + 1]
    
    success = convert_raw_to_image(raw_path, output_path, format_arg)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()