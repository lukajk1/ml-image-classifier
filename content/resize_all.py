import os
from PIL import Image

MAX_SIZE = 256
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
ROOT = os.path.dirname(os.path.abspath(__file__))


def resize_image(path):
    img = Image.open(path).convert("RGB")
    if max(img.size) <= MAX_SIZE:
        return False
    img.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)
    img.save(path, "JPEG", quality=85)
    return True


def main():
    resized, skipped, failed = 0, 0, 0
    for dirpath, _, filenames in os.walk(ROOT):
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() not in EXTENSIONS:
                continue
            path = os.path.join(dirpath, filename)
            try:
                if resize_image(path):
                    print(f"  resized: {os.path.relpath(path, ROOT)}")
                    resized += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  failed:  {os.path.relpath(path, ROOT)}: {e}")
                failed += 1

    print(f"\nDone. {resized} resized, {skipped} already small enough, {failed} failed.")


if __name__ == "__main__":
    main()
