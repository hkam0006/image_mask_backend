from typing import Annotated
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from os.path import dirname, join
import base64
import numpy as np
import dataclasses
from PIL import Image
from google import genai
from google.genai import types
import io
import json
from io import BytesIO
from dotenv import load_dotenv
import os
import uvicorn
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import HTTPException, Depends

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

app = FastAPI()

ALLOWED_ORIGIN = os.environ.get("FRONTEND_URL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
  token = credentials.credentials
  if token != os.environ.get("AUTH_TOKEN"):  # Replace with your token validation logic
    raise HTTPException(status_code=401, detail="Invalid or missing token")

@app.middleware("http")
async def log_request_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    print(f"Incoming request headers: {request.headers}")
    response = await call_next(request)
    return response
  
@dataclasses.dataclass(frozen=True)
class SegmentationMask:
  # bounding box pixel coordinates (not normalized)
  y0: int # in [0..height - 1]
  x0: int # in [0..width - 1]
  y1: int # in [0..height - 1]
  x1: int # in [0..width - 1]
  mask: np.array # [img_height, img_width] with values 0..255
  label: str

def parse_json(json_output: str):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def parse_segmentation_masks(
    predicted_str: str, *, img_height: int, img_width: int
) -> list[SegmentationMask]:
  items = json.loads(parse_json(predicted_str))
  masks = []
  for item in items:
    raw_box = item["box_2d"]
    abs_y0 = int(item["box_2d"][0] / 1000 * img_height)
    abs_x0 = int(item["box_2d"][1] / 1000 * img_width)
    abs_y1 = int(item["box_2d"][2] / 1000 * img_height)
    abs_x1 = int(item["box_2d"][3] / 1000 * img_width)
    if abs_y0 >= abs_y1 or abs_x0 >= abs_x1:
      print("Invalid bounding box", item["box_2d"])
      continue
    print(item)
    label = item["label"]
    png_str = item["mask"]
    if not png_str.startswith("data:image/png;base64,"):
      print("Invalid mask")
      continue
    png_str = png_str.removeprefix("data:image/png;base64,")
    png_str = base64.b64decode(png_str)
    mask = Image.open(io.BytesIO(png_str))
    bbox_height = abs_y1 - abs_y0
    bbox_width = abs_x1 - abs_x0
    if bbox_height < 1 or bbox_width < 1:
      print("Invalid bounding box")
      continue
    mask = mask.resize((bbox_width, bbox_height), resample=Image.Resampling.BILINEAR)
    np_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    np_mask[abs_y0:abs_y1, abs_x0:abs_x1] = mask
    masks.append(SegmentationMask(abs_y0, abs_x0, abs_y1, abs_x1, np_mask, label))
  return masks

def overlay_mask_on_img(
    img: Image,
    mask_layer: Image,
    mask: np.ndarray,
    color: tuple = (255, 0, 0),
    alpha: float = 0.2
) -> Image.Image:
    """
    Overlays a single mask onto a PIL Image using a named color.

    The mask image defines the area to be colored. Non-zero pixels in the
    mask image are considered part of the area to overlay.

    Args:
        img: The base PIL Image object.
        mask: A PIL Image object representing the mask.
              Should have the same height and width as the img.
              Modes '1' (binary) or 'L' (grayscale) are typical, where
              non-zero pixels indicate the masked area.
        color: A standard color name string (e.g., 'red', 'blue', 'yellow').
        alpha: The alpha transparency level for the overlay (0.0 fully
               transparent, 1.0 fully opaque). Default is 0.7 (70%).

    Returns:
        A new PIL Image object (in RGBA mode) with the mask overlaid.

    Raises:
        ValueError: If color name is invalid, mask dimensions mismatch img
                    dimensions, or alpha is outside the 0.0-1.0 range.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("Alpha must be between 0.0 and 1.0")

    # Convert the color name string to an RGB tuple
    try:
        color_rgb = color
    except ValueError as e:
        # Re-raise with a more informative message if color name is invalid
        raise ValueError(f"Invalid color name '{color}'. Supported names are typically HTML/CSS color names. Error: {e}")

    # Prepare the base image for alpha compositing
    img_rgba = img.convert("RGBA")
    mask_layer_rgba = mask_layer.convert("RGBA")
    width, height = img_rgba.size

    # Create the colored overlay layer
    # Calculate the RGBA tuple for the overlay color
    alpha_int = int(alpha * 255)
    overlay_color_rgba = color_rgb + (alpha_int,)

    # Create an RGBA layer (all zeros = transparent black)
    colored_mask_layer_np = np.zeros((height, width, 4), dtype=np.uint8)

    # Mask has values between 0 and 255, threshold at 127 to get binary mask.
    mask_np_logical = mask > 127

    # Apply the overlay color RGBA tuple where the mask is True
    colored_mask_layer_np[mask_np_logical] = overlay_color_rgba

    # Convert the NumPy layer back to a PIL Image
    colored_mask_layer_pil = Image.fromarray(colored_mask_layer_np, 'RGBA')

    # Composite the colored mask layer onto the base image
    result_img = Image.alpha_composite(img_rgba, colored_mask_layer_pil)
    result_mask_only = Image.alpha_composite(mask_layer_rgba, colored_mask_layer_pil)    

    return result_img, result_mask_only

def plot_segmentation_masks(img: Image, segmentation_masks: list[SegmentationMask], color: tuple = (255, 0, 0), alpha: float = 0.2):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img: The PIL.Image.
        segmentation_masks: A string encoding as JSON a list of segmentation masks containing the name of the object,
         their positions in normalized [y1 x1 y2 x2] format, and the png encoded segmentation mask.
    """
    width, height = img.size
    mask_only = Image.fromarray(np.zeros((height, width, 4), dtype=np.uint8), 'RGBA')
    
    for _, mask in enumerate(segmentation_masks):
      img, mask_only = overlay_mask_on_img(img, mask_only, mask.mask, color, alpha=alpha)
    return img, mask_only

@app.get("/")
async def health_check():
  return {"status": "healthy"}
  
@app.post("/detect_objects/")
async def detect_objects(
  req: Request,
  red: int, 
  green: int, 
  blue:int, 
  alpha: int, 
  object: str, 
  file: UploadFile = File(...),
  credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):  
  prompt = f"Give the segmentation masks for all {object} in this image. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key 'box_2d', the segmentation mask in key 'mask', and the text label in the key 'label'. Use descriptive labels."
  im = Image.open(BytesIO(await file.read()))
  im.thumbnail((1024, 1024), Image.LANCZOS)
  
  color = (red, green,blue, alpha / 100)
  
  response = client.models.generate_content(
    model="gemini-2.5-flash-preview-04-17",
    contents=[
      prompt,
      im
    ],
    config = types.GenerateContentConfig(
        temperature=0.5,
        safety_settings=safety_settings,
    )
  )
  
  seg_mask = parse_segmentation_masks(
    response.text, 
    img_height=im.size[1], 
    img_width=im.size[0]
  )
  image_with_mask, mask_only = plot_segmentation_masks(
      im, 
      seg_mask,
      color=(red, green, blue),
      alpha=alpha / 100
    )
  
  # send the image with mask to the frontend
  buffered = BytesIO()
  image_with_mask.save(buffered, format="PNG")
  img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
  img_str = f"data:image/png;base64,{img_str}"
  
  # send the mask only to the frontend
  buffered = BytesIO()
  mask_only.save(buffered, format="PNG")
  mask_only_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
  mask_only_str = f"data:image/png;base64,{mask_only_str}"
  
  return {
    "image": img_str,
    "mask": mask_only_str,
  }
  
if __name__ == "__main__":
  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)