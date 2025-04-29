from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import cv2
import json
from detect_compo import ip_region_proposal as ip

app = FastAPI()

# --- Configuración CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Funciones auxiliares ---
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

def guess_component_type(comp):
    width = comp['width']
    height = comp['height']

    if width > 250 and height > 100:
        return "card"
    if height > 50 and width < 300:
        return "textarea"
    if width < 150 and height < 50:
        return "label"

    return "input"

def format_components(components):
    formatted = []
    for comp in components:
        formatted.append({
            "id": comp['id'],
            "type": guess_component_type(comp),
            "text": "",
            "x": comp['column_min'],
            "y": comp['row_min'],
            "width": comp['width'],
            "height": comp['height'],
        })
    return formatted

# --- Endpoint principal ---
@app.post("/api/detect-components")
async def detect_components(file: UploadFile = File(...)):
    temp_filename = None
    compo_path = None
    try:
        # Guardar imagen temporal
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_filename = os.path.join(temp_dir, f"{uuid.uuid4()}.png")

        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Detectar componentes
        output_root = "output"
        os.makedirs(os.path.join(output_root, "ip"), exist_ok=True)

        resized_height = resize_height_by_longest_edge(temp_filename)

        key_params = {'min-grad': 10, 'ffl-block': 5, 'min-ele-area': 50,
                      'merge-contained-ele': True, 'merge-line-to-paragraph': False, 'remove-bar': True}

        name = os.path.splitext(os.path.basename(temp_filename))[0]

        ip.compo_detection(temp_filename, output_root, key_params,
                           classifier=None, resize_by_height=resized_height, show=False)

        # Procesar resultados
        compo_path = os.path.join(output_root, 'ip', f"{name}.json")
        with open(compo_path, 'r') as f:
            data = json.load(f)

        components = data.get('compos', [])
        filtered_components = filter_components(components)
        final_components = format_components(filtered_components)

        return JSONResponse(content={"components": final_components})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        # Limpiar archivos temporales
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)
        if compo_path and os.path.exists(compo_path):
            os.remove(compo_path)
