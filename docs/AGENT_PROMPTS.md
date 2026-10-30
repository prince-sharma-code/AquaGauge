# AI-Agent Workflow: Copy/Paste Prompts

## Global rules for every agent

> Work only in your assigned files/branch. Read `README.md` and `docs/HACKATHON_PLAN.md` first. Do not change API schemas or dependencies without telling the integrator. Before finishing, run relevant tests/build, list exact files changed, commands run, known gaps, and a one-sentence merge instruction. Do not claim a model works unless you ran it on the bundled images. Prefer a small working feature over architecture abstractions. Keep all secret/API keys out of Git. The app must retain a deterministic demo fallback if inference or internet is unavailable.

## Vision agent (Python)

> Build a focused `vision.py` module to run a small pretrained vehicle detector on an uploaded/local image. Return JSON boxes with `x1,y1,x2,y2,class_name,confidence` normalized or in pixels (choose one and document it). Add a helper to draw the boxes into a saved image. Do not call the outputs flood segmentation. Test on every image in `assets/demo_images/`. If weights fail to download, provide a stub/fallback that uses a clearly labelled precomputed demo result. Do not train a model. Ensure CPU inference works.

## Frontend agent (React + TypeScript)

> Build a polished dark municipal flood-operations dashboard. Core flow: select bundled image/upload image → show image canvas → show vehicle boxes if returned → allow operator to mark reference top, reference base, and waterline (or draw an editable water polygon) → call `/api/estimate` → render estimated depth, ratio, reference sufficiency, vehicle risk status, incident card, and blocked/detour route schematic. Explicitly label operator input and demo thresholds; never imply the water mask is always model-generated. Add a local static fallback so the whole dashboard is usable when backend is down. Use accessible text controls, avoid adding major dependencies, and run `npm run build` before finishing.

## Backend/API agent (FastAPI)

> Preserve the `/api/health` and `/api/estimate` schemas documented in this repo. Add any new endpoints with typed Pydantic models, validation, and tests. Load risk thresholds from a config file and label them illustrative. When geometry is invalid, return an explicit abstain/review response rather than a number. Do not hardcode a claim of centimeter-level accuracy. Add sample camera/incident fixtures. Run `pytest` and provide a curl example for the frontend.

## GIS/operations agent

> Implement the municipal response stage only: a road segment can be `OPEN`, `HIGH_RISK`, or `BLOCKED`; creating an incident appends a timestamped record; a route view shows the original route and alternate route around one blocked segment. Prefer a local SVG/schematic if online maps add fragility. Do not claim an actual city traffic API or real ambulance dispatch. Keep JSON structures documented and tests for risk-to-status mapping.

## Integrator prompt (Codex / Antigravity)

> Inspect the current repository and build the thinnest end-to-end vertical slice. First report the current state and propose a <=6-step plan. Do not rewrite the whole project. Keep one component per file, write small tests for geometry/risk rules, and ensure the clean-start commands work. If another agent owns a file, don't edit it; define the interface and ask for the exact exported function/schema instead. Never install or migrate dependencies without approval. At the end, run all tests plus a production build and report any remaining manual steps.

## Gemini Deep Research prompt (only if needed; timebox to 15 minutes)

> Find a dataset/model that can be legally used in a student hackathon prototype for street-level urban floodwater segmentation or vehicle-submersion estimation. Prioritize official dataset/model cards, clear licences, image download size, inference requirements, and direct relevance to fixed-angle CCTV rather than satellite/aerial imagery. Return no more than 3 options in a table, note domain mismatch and license caveats, and separate verified facts from suggestions. Do not recommend training from scratch or a multi-hour setup. Cite primary sources. We already know about FloodDET (CC BY 4.0, 10,262 images, ~5.74 GB) and the Floodwater Dataset (water masks, aerial/oblique UAV, GPL-3.0); only return a new option if it materially improves the 24-hour build.
