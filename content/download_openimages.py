import os
from PIL import Image
import fiftyone as fo
import fiftyone.zoo as foz

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloaded")
NUM_SAMPLES = 500
MAX_SIZE = 256  # longest edge in pixels
SPLITS = ["train", "validation", "test"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def resize_and_save(src_path, save_path, max_size):
    img = Image.open(src_path).convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    img.save(save_path, "JPEG", quality=85)


def main():
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        max_samples=NUM_SAMPLES,
        shuffle=True,
        seed=42,
        only_matching=True,
        label_types=[],  # no annotations
    )

    print(f"\nDownloading and resizing {len(dataset)} images to {MAX_SIZE}px max...\n")
    failed = 0

    for i, sample in enumerate(dataset):
        filename = f"{i:05d}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        try:
            resize_and_save(sample.filepath, save_path, MAX_SIZE)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(dataset)}")
        except Exception as e:
            import traceback
            print(f"  Failed {filename}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nDone. {len(dataset) - failed} saved to {OUTPUT_DIR}/  ({failed} failed)")
    fo.delete_dataset(dataset.name)


if __name__ == "__main__":
    main()
