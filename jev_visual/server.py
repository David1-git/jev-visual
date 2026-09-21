import os
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .schema import Request


@asynccontextmanager
async def lifespan(app):
    backend = os.environ.get("JEV_VISUAL_BACKEND", "mlx")
    if backend == "mlx":
        from .engine import Engine
        factory = lambda path: Engine(path)
    elif backend == "torch_internvl":
        from .internvl import build_internvl_engine
        factory = lambda path: build_internvl_engine(path)
    elif backend == "torch_qwen35":
        from .qwen35 import build_qwen35_engine
        factory = lambda path: build_qwen35_engine(path)
    else:
        raise ValueError(f"unknown JEV_VISUAL_BACKEND: {backend}")
    # MLX streams are thread-local; torch backends are thread-safe per request
    # through Engine's lock. Load and infer on the SAME dedicated worker,
    # never on FastAPI's arbitrary request threadpool.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev-visual") as executor:
        app.state.executor = executor
        app.state.engine = await asyncio.get_running_loop().run_in_executor(
            executor, partial(factory, os.environ.get("JEV_VISUAL_MODEL_PATH"))
        )
        yield


app = FastAPI(title="Jev Visual (local Qwen prototype)", lifespan=lifespan)


@app.get("/health")
def health():
    return {"ready": hasattr(app.state, "engine"), "calibrated": False}


@app.post("/v1/judge")
async def judge(request: Request):
    try:
        return await asyncio.get_running_loop().run_in_executor(
            app.state.executor, partial(app.state.engine.judge, request, allow_path=False)
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def home():
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Jev Visual</title>
<style>body{max-width:900px;margin:40px auto;font:16px system-ui;background:#f6f7fa;color:#172331}textarea{width:100%;height:260px}button{padding:12px 24px;margin:15px 0}pre{white-space:pre-wrap;background:white;padding:20px}img{max-width:360px;max-height:250px}small{color:#536275}</style>
<h1>Jev Visual</h1><p><a href="/demo/">Visual inference demos</a></p><p>Choose an image and ask several decision questions.</p><input id="file" type="file" accept="image/*"><p><img id="preview"></p>
<textarea id="questions">{"subject":{"type":"choice","instructions":"What is the main subject in the image?","criteria":{"person":"A person","animal":"An animal","object":"An object","other":"Something else"}},"has_text":{"type":"noul","instructions":"Is there readable text in the image?"}}</textarea>
<button id="run">Analyze image</button><small>Candidate probabilities are relative to the supplied options, not calibrated.</small><pre id="result">Waiting for an image</pre>
<script>let image;const $=id=>document.getElementById(id);$('file').onchange=()=>{const f=$('file').files[0];if(!f)return;const r=new FileReader();r.onload=()=>{image=r.result;$('preview').src=image};r.readAsDataURL(f)};
$('run').onclick=async()=>{if(!image){$('result').textContent='Choose an image first.';return}$('run').disabled=true;$('result').textContent='Analyzing…';try{const questions=JSON.parse($('questions').value);const r=await fetch('/v1/judge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image,questions})});$('result').textContent=JSON.stringify(await r.json(),null,2)}catch(e){$('result').textContent=String(e)}finally{$('run').disabled=false}};</script></html>'''


# Source-checkout demos share the API origin and the existing model worker.
_demo_dir = Path(__file__).resolve().parent.parent / "demo"
if _demo_dir.is_dir():
    app.mount("/demo", StaticFiles(directory=_demo_dir, html=True), name="demo")
