# AquaGauge 24-Hour Hackathon Starter Kit

This is a **prep kit and backend seed**, not a completed product. It includes a small API for the reference-height calculation, illustrative risk-policy defaults, preflight checks, image assets extracted from the team's existing presentation, and a 24-hour build plan.

## Product scope

A demo should take a selected/uploaded road image, show vehicle/water annotations, let the operator correct the waterline/reference if automatic perception is uncertain, estimate depth from a known reference height, show a configurable risk status, and generate an incident/route-change view.

**Important:** the included flood photos are sample UI inputs, not a labelled training/evaluation set. The sample depth and policy output are illustrative. Do not describe them as validated measurements or real-world safety limits.

## Start the backend

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r backend/requirements-core.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive API docs.

Test the calculation core:

```bash
python -m pytest backend/tests -q
```

Example API request (assuming the backend is running):

```bash
curl -X POST http://127.0.0.1:8000/api/estimate \
  -H 'Content-Type: application/json' \
  -d '{"reference_height_cm":60,"y_top":100,"y_base":200,"y_waterline":147,"vehicle_class":"light_vehicle"}'
```

Edit the clearly illustrative thresholds in `backend/app/policy.json`; do not treat them as real safety standards.

Optional computer-vision stack:

```bash
python -m pip install -r backend/requirements-vision.txt
python backend/scripts/check_env.py
```

Then test pretrained vehicle detection before wiring it into the UI:

```bash
python - <<'PY'
from ultralytics import YOLO
model = YOLO('yolo26n.pt')
results = model('assets/demo_images/flooded_vehicle_underpass.jpg', verbose=False)
print([(int(b.cls[0]), round(float(b.conf[0]), 2)) for r in results for b in r.boxes])
PY
```

This model detects common objects such as cars/buses; it does **not** by itself segment floodwater. Don't represent object detection as flood segmentation.

## Recommended demo path

1. Start with the supplied image gallery and a clear uploaded-image path.
2. Add pretrained vehicle detection if the install/download works.
3. Add a human-in-the-loop waterline/reference annotation for dependable depth geometry.
4. Apply the demo policy and create an incident event.
5. Draw the blocked road and alternative route on a simple map. Add online tiles only if time/network permit.
6. If auto water segmentation is reliable, make it a bonus; always keep manual correction/abstention.

## Assets

- `assets/demo_images/`: three flood scene images extracted from the team's uploaded AquaGauge deck.
- `assets/reference_graphics/`: existing wheel/reference geometry, depth-policy illustration, and route schematic.
- `assets/media_contact_sheet.jpg`: quick view of the presentation's embedded media.

The original deck did not contain complete per-image source/licence metadata. Treat extracted photos as existing team presentation assets and confirm source/permission before publishing or redistributing them. For a dataset with explicit terms, see `docs/IMAGE_DATA_GUIDE.md`.
