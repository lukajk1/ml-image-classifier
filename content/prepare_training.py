import os
import shutil

CONTENT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

SOURCES = {
    "training_photos": ["photos_fiftyone", "photos_mine"],
    "training_artwork": ["artwork"],
}


def copy_images(src_dir, dst_dir):
    if not os.path.exists(src_dir):
        print(f"  skipping (not found): {src_dir}")
        return 0
    count = 0
    for filename in os.listdir(src_dir):
        if os.path.splitext(filename)[1].lower() not in EXTENSIONS:
            continue
        src = os.path.join(src_dir, filename)
        dst = os.path.join(dst_dir, filename)
        if os.path.exists(dst):
            base, ext = os.path.splitext(filename)
            dst = os.path.join(dst_dir, f"{base}_{os.path.basename(src_dir)}{ext}")
        shutil.copy2(src, dst)
        count += 1
    return count


def main():
    for target, sources in SOURCES.items():
        dst_dir = os.path.join(CONTENT_DIR, target)
        os.makedirs(dst_dir, exist_ok=True)
        total = 0
        for src_name in sources:
            src_dir = os.path.join(CONTENT_DIR, src_name)
            n = copy_images(src_dir, dst_dir)
            print(f"  {src_name} -> {target}: {n} images")
            total += n
        print(f"  total in {target}: {total}\n")


if __name__ == "__main__":
    main()
