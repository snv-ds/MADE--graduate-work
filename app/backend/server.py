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
    tags = get_tags(model, file)
    return JSONResponse(tags)
#
# @app.post("/predict_test")
# def get_s_test(text):
#     print(text)
#     file = np.fromfile(text, dtype=np.uint8)
#     tags = get_tags(model, file)
#     return JSONResponse(tags)

