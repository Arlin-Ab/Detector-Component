from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import cv2
import json
import easyocr
from detect_compo import ip_region_proposal as ip
import os

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
     allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def filter_components(components, min_width=40, min_height=20, min_area=800):
    return [
        comp for comp in components
        if comp['width'] >= min_width
        and comp['height'] >= min_height
        and (comp['width'] * comp['height']) >= min_area
    ]

def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    return resize_length if height > width else int(resize_length * (height / width))

def detect_text_easyocr(img_path):
    reader = easyocr.Reader(["es", "en"], gpu=False)
    results = reader.readtext(img_path)
    texts = []
    for (bbox, text, confidence) in results:
        if confidence < 0.5 or not text.strip():
            continue  
        (top_left, top_right, bottom_right, bottom_left) = bbox
        x_coords = [top_left[0], top_right[0], bottom_right[0], bottom_left[0]]
        y_coords = [top_left[1], top_right[1], bottom_right[1], bottom_left[1]]
        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        width = int(max(x_coords)) - x_min
        height = int(max(y_coords)) - y_min
        if width > 5 and height > 5:
            texts.append({
                "text": text.strip(),
                "x": x_min,
                "y": y_min,
                "width": width,
                "height": height
            })
    return texts

def associate_texts_to_components(components, texts):
    associated = []

    for comp in components:
        comp_center_x = comp['column_min'] + comp['width'] / 2
        comp_center_y = comp['row_min'] + comp['height'] / 2

        best_match = None
        min_distance = float('inf')

        for text in texts:
            text_center_x = text['x'] + text['width'] / 2
            text_center_y = text['y'] + text['height'] / 2

            distance = ((comp_center_x - text_center_x) ** 2 + (comp_center_y - text_center_y) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                best_match = text

        # Si el texto más cercano está razonablemente cerca, asociarlo
        if best_match and min_distance < max(comp['width'], comp['height']) * 1.5:  
            assigned_text = best_match['text']
        else:
            assigned_text = ""

        associated.append({
            "id": comp['id'],
            "type": guess_component_type(comp, assigned_text),
            "text": assigned_text,
            "x": comp['column_min'],
            "y": comp['row_min'],
            "width": comp['width'],
            "height": comp['height'],
        })

    return associated


def guess_component_type(comp, text):
    width = comp['width']
    height = comp['height']
    text = (text or "").lower()

    
    if "login" in text or "entrar" in text or "aceptar" in text or "submit" in text or "button" in "registrar" in text:
        return "button"
    if "input" in text or "ingrese" in text or "escriba" in text:
        return "input"
    if "textarea" in text:
        return "textarea"
    if "card" in text or "bloque" in text:
        return "card"
    if "etiqueta" in text or (len(text.strip()) <= 10 and width < 200 and height < 100):
        return "label"

   
    if width > 250 and height > 100:
        return "card"
    if height > 50 and width < 300:
        return "textarea"
    if width < 150 and height < 50:
        return "label"

    return "input"




@app.post("/api/detect-components")
async def detect_components(file: UploadFile = File(...)):
    temp_filename = None
    compo_path = None
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = os.path.join(temp_dir, f"{uuid.uuid4()}.png")

        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_root = "output"
        os.makedirs(os.path.join(output_root, "ip"), exist_ok=True)

        resized_height = resize_height_by_longest_edge(temp_filename)

        key_params = {'min-grad': 10, 'ffl-block': 5, 'min-ele-area': 50,
                      'merge-contained-ele': True, 'merge-line-to-paragraph': False, 'remove-bar': True}

        name = os.path.splitext(os.path.basename(temp_filename))[0]

        ip.compo_detection(temp_filename, output_root, key_params,
                           classifier=None, resize_by_height=resized_height, show=False)

        compo_path = os.path.join(output_root, 'ip', f"{name}.json")
        with open(compo_path, 'r') as f:
            data = json.load(f)

        components = data.get('compos', [])
        filtered_components = filter_components(components)

        texts = detect_text_easyocr(temp_filename)

        final_components = associate_texts_to_components(filtered_components, texts)

        return JSONResponse(content={"components": final_components})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    finally:
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)
        if compo_path and os.path.exists(compo_path):
            os.remove(compo_path)
            
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

