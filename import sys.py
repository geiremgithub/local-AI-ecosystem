import sys

print("=" * 50)
print(f"Kjører med Python-versjon: {sys.version.split()[0]}")
print("=" * 50)

# 1. Test Pillow (bildebehandling)
try:
    from PIL import Image, ImageDraw
    # Opprett et lite testbilde i minnet
    test_bilde = Image.new("RGB", (100, 100), color="blue")
    print("[OK] Pillow (PIL) fungerer og kan opprette bilder.")
except Exception as e:
    print(f"[FEIL] Kunne ikke bruke Pillow: {e}")

# 2. Test qwen-vl-utils
try:
    from qwen_vl_utils import process_vision_info
    print("[OK] qwen-vl-utils ble importert uten problemer.")
except Exception as e:
    print(f"[FEIL] Kunne ikke importere qwen-vl-utils: {e}")

# 3. Test PyTorch og CUDA (GPU)
try:
    import torch
    print(f"[OK] PyTorch versjon {torch.__version__} ble importert.")
    
    if torch.cuda.is_available():
        gpu_navn = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[OK] CUDA (GPU) er tilgjengelig!")
        print(f"     GPU: {gpu_navn}")
        print(f"     Minne (VRAM): {vram:.2f} GB")
    else:
        print("[ADVARSEL] CUDA er ikke tilgjengelig. PyTorch vil kjøre på CPU (dette vil gå tregere med visuelle modeller).")
except Exception as e:
    print(f"[FEIL] Kunne ikke laste inn PyTorch: {e}")

print("=" * 50)