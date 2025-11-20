from pathlib import Path
import json

import fire
from matplotlib import pyplot as plt

from generate_qa import draw_detections, extract_frame_info, extract_kart_objects

ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.
    
    # ------------------------------------------------------------
    # LOAD JSON
    # ------------------------------------------------------------
    info_path = Path(info_path)
    with open(info_path, "r") as f:
        info = json.load(f)


    # ------------------------------------------------------------
    # GET CORRECT VIEW DATA
    # ------------------------------------------------------------
    karts = extract_kart_objects(str(info_path), view_index, img_width, img_height)

    base = info_path.stem.replace("_info", "")
    image_file = f"{base}_{view_index:02d}_im.jpg"

    captions = []
    
    # -------------------------------------------------------
    # 1. EGO KART QUESTION
    # -------------------------------------------------------
    ego = next((k for k in karts if k["is_center_kart"]), None)
    if ego:
        captions.append({
            "image_file": image_file,
            "caption": f"{ego['kart_name']} is the ego car."
        })


    # -------------------------------------------------------
    # 2. COUNTING CAPTION
    # -------------------------------------------------------
    captions.append({
        "image_file": image_file,
        "caption": f"There are {len(karts)} karts in the scene."
    })

    # -------------------------------------------------------
    # 3. TRACK NAME CAPTION
    # -------------------------------------------------------
    track_name = info.get("track", "unknown track")
    captions.append({
        "image_file": image_file,
        "caption": f"The track is {track_name}."
    })

    # -------------------------------------------------------
    # 4. RELATIVE POSITION CAPTION
    # -------------------------------------------------------
    if ego:
        ego_x, ego_y = ego["center"]
        for kart in karts:
            if kart == ego:
                continue
            x, y = kart["center"]

            # left / right
            if x < ego_x:
                captions.append({
                    "image_file": image_file,
                    "caption": f"{kart['kart_name']} is left of the ego car."
                })
            else:
                captions.append({
                    "image_file": image_file,
                    "caption": f"{kart['kart_name']} is right of the ego car."
                })

            # front / behind
            if y < ego_y:
                captions.append({
                    "image_file": image_file,
                    "caption": f"{kart['kart_name']} is in front of the ego car."
                })
            else:
                captions.append({
                    "image_file": image_file,
                    "caption": f"{kart['kart_name']} is behind the ego car."
                })
    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()




def generate_all_captions(data_dir: str, output_file: str):
    data_dir = Path(data_dir)
    all_captions = []

    info_files = sorted(list(data_dir.glob("*_info.json")))

    for info_path in info_files:
        for view_index in range(10):
            try:
                captions = generate_caption(str(info_path), view_index)

                if captions:
                    all_captions.extend(captions)

            except Exception as e:
                print(f"[WARN] Skipping {info_path} view {view_index} due to error: {e}")
                continue

    with open(output_file, "w") as f:
        json.dump(all_captions, f, indent=2)

    print(f"Saved {len(all_captions)} captions to {output_file}")





"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({"check": check_caption,
               "generate": generate_all_captions})


if __name__ == "__main__":
    main()
