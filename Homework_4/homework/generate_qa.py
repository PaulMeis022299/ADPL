import json
from pathlib import Path
import os

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

import math

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    
    # ------------------------------------------------------------
    # LOAD JSON
    # ------------------------------------------------------------
    with open(info_path, "r") as f:
        data = json.load(f)

    kart_names = data["karts"]
    all_view_detections = data["detections"]
    detections = all_view_detections[view_index]

    # ------------------------------------------------------------
    # SET UP SCALE FROM img_width & img_heigth TO ORIGINAL
    # ------------------------------------------------------------
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # ------------------------------------------------------------
    # EXTRACT ALL KART OBJECTS
    # ------------------------------------------------------------
    kart_objects = []

    for det in detections:
        class_id, kart_id, x1, y1, x2, y2 = det

        # Filter out non karts
        if class_id != 1:
            continue

        kart_name = kart_names[kart_id] # Get kart_name from kart_id index

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Compute center
        cx = (x1_scaled + x2_scaled) / 2
        cy = (y1_scaled + y2_scaled) / 2

        # Must be within image bounds
        if not (0 <= cx < img_width and 0 <= cy < img_height):
            continue

        kart_objects.append({
            "instance_id": kart_id,     # should be kart_id in this frame
            "kart_name": kart_name,
            "center": (cx, cy),
            "is_center_kart": False
        })

    # ------------------------------------------------------------
    # IDENTIFY CENTER KART
    # ------------------------------------------------------------
    if kart_objects:
        img_center = (img_width / 2, img_height / 2)
        dists = [math.dist(kart["center"], img_center) for kart in kart_objects]
        center_idx = min(range(len(dists)), key=lambda i: dists[i])
        kart_objects[center_idx]["is_center_kart"] = True

    return kart_objects



def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """

    with open(info_path, "r") as f:
        data = json.load(f)
    track_name = data.get("track", None)

    return track_name


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    # ------------------------------------------------------------
    # HELPER FUNCTIONS
    # ------------------------------------------------------------
    def left_or_right(kart):
        x = kart["center"][0]
        return "left" if x < ego_x else "right"

    def front_or_behind(kart):
        y = kart["center"][1]
        return "front" if y < ego_y else "behind"


    qa_pairs = []

    # ------------------------------------------------------------
    # EXTRACT DATA FROM JSON
    # ------------------------------------------------------------
    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    # If no karts detected - return track-only QA pair
    # per Caitlin Tracht on ED for filtering QA
    if len(karts) == 0:
        qa_pairs.append({
            "question": "What track is this?",
            "answer": track_name
        })
        return qa_pairs

    ego = None
    for k in karts:
        if k["is_center_kart"]:
            ego = k

    if ego == None:
        return qa_pairs

    ego_name = ego["kart_name"]
    ego_x, ego_y = ego["center"]


    # -------------------------------------------------------
    # 1. EGO KART QUESTION
    # -------------------------------------------------------
    qa_pairs.append({
        "question": "What kart is the ego car?",
        "answer": ego_name
    })

    # -------------------------------------------------------
    # 2. TOTAL KARTS QUESTION
    # -------------------------------------------------------
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(karts))
    })

    # -------------------------------------------------------
    # 3. TRACK INFORMATION QUESTIONS
    # -------------------------------------------------------
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name
    })

    # -------------------------------------------------------
    # 4. RELATIVE POSITION QUESTIONS FOR EACH KART
    # -------------------------------------------------------
    left_count = 0
    right_count = 0
    front_count = 0
    behind_count = 0
    
    for kart in karts:
        if kart is ego:
            continue

        name = kart["kart_name"]
        left_right = left_or_right(kart)
        front_behind = front_or_behind(kart)
        relative = f"{front_behind} and {left_right}"

        # Left/Right
        qa_pairs.append({
            "question": f"Is {name} to the left or right of the ego car?",
            "answer": left_right
        })

        # Front/Behind
        qa_pairs.append({
            "question": f"Is {name} in front of or behind the ego car?",
            "answer": front_behind
        })

        # Relative question
        qa_pairs.append({
            "question": f"Where is {name} relative to the ego car?",
            "answer": relative
        })

        # To the left/right
        if left_right == "left":
            left_count += 1
        else:
            right_count += 1

        # To the front/behind
        if front_behind == "front":
            front_count += 1
        else:
            behind_count += 1

    # -------------------------------------------------------
    # 5. COUNTING QUESTIONS
    # -------------------------------------------------------
    qa_pairs.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(left_count)
    })
    qa_pairs.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(right_count)
    })
    qa_pairs.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(front_count)
    })
    qa_pairs.append({
        "question": "How many karts are behind the ego car?",
        "answer": str(behind_count)
    })
    
    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


def generate_all_qa_pairs(data_dir: str, output_file: str):
    """
    Generate QA pairs for all info.json files in a directory.
    """
    qa_output = []

    for file in sorted(os.listdir(data_dir)):
        if not file.endswith("_info.json"):
            continue

        info_path = os.path.join(data_dir, file)
        frame_id = file.split("_")[0]

        for view_index in range(10):

            qa_pairs = generate_qa_pairs(info_path, view_index)
            if len(qa_pairs) == 0:
                continue
            image_file = f"{frame_id}_{view_index:02d}_im.jpg"

            for qa in qa_pairs:
                qa_output.append({
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "image_file": image_file
                })

    
    with open(output_file, "w") as f:
        json.dump(qa_output, f, indent=2)

    print(f"Saved {len(qa_output)} training QA examples to {output_file}")



"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""


def main():
    fire.Fire({
        "check": check_qa_pairs,
        "generate": generate_all_qa_pairs
    })


if __name__ == "__main__":
    main()
