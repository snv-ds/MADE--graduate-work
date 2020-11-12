from fastapi import FastAPI, File
from starlette.responses import JSONResponse

from model import get_model, get_tags

model = get_model()

app = FastAPI(title="MADE graduate project",
              description='''Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
                           Visit this URL at port 8501 for the streamlit interface.''',
              version="0.0.1",
              )


@app.post("/predict")
def get_s(file: bytes = File(...)):
    '''Get segmentation maps from image file'''
    tags = get_tags(model, file)
    # bytes_io = io.BytesIO()
    # tags.save(bytes_io, format='PNG')
    return JSONResponse(tags)
