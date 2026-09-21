import base64
import io
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from jev_visual.preprocessing import read_image
from jev_visual.server import app


def image_url():
    buf = io.BytesIO()
    Image.new("RGB", (900, 450), "blue").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_image_decoding_and_bounded_resolution():
    img = read_image(image_url(), allow_path=False)
    assert img.size == (768, 384)
    assert img.mode == "RGB"
    with pytest.raises(ValueError, match="data URL"):
        read_image("/etc/passwd", allow_path=False)
    with pytest.raises(ValueError):
        read_image("data:image/png;base64,not!valid", allow_path=False)


def test_api_validation_and_data_url_policy():
    class FakeEngine:
        def judge(self, request, *, allow_path):
            assert not allow_path
            read_image(request.image, allow_path=allow_path)
            return {"answers": {"red": {"type": "noul", "noul": .2}}}
    app.state.engine = FakeEngine()
    app.state.executor = ThreadPoolExecutor(max_workers=1)
    client = TestClient(app)
    questions = {"red": {"type": "noul", "instructions": "Is it red?"}}
    assert client.get("/").status_code == 200
    assert client.post("/v1/judge", json={"image": image_url(), "questions": questions}).json()["answers"]["red"]["noul"] == .2
    assert client.post("/v1/judge", json={"image": "/etc/passwd", "questions": questions}).status_code == 422
    assert client.post("/v1/judge", json={"image": image_url(), "questions": {}}).status_code == 422
    app.state.executor.shutdown()


def test_demo_pages_and_static_modules():
    client = TestClient(app)
    for path, expected in [
        ('/demo/', 'Visual inference demos'),
        ('/demo/rubik/', 'game.js'),
        ('/demo/factory/', 'factory.js'),
        ('/demo/breakout/', 'game.js'),
        ('/demo/gestures/', 'gestures.js'),
        ('/demo/shared.js', 'mountDemo'),
    ]:
        response = client.get(path)
        assert response.status_code == 200
        assert expected in response.text

    assert client.get("/demo/snake/").status_code == 404
