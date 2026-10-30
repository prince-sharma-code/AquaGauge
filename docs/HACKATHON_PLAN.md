# AquaGauge: 24-Hour Build Plan

## The demo promise

**Given a road image, AquaGauge produces a reviewable visual flood-risk event:** vehicle detections, water/reference annotations, an illustrative or calibrated depth estimate, a configurable vehicle/road status, and a proposed route adaptation.

The demo must be honest about what is automatic and what is operator-confirmed. Never claim production-ready precision, universal road-safety thresholds, or live ICCC/GIS integration unless actually implemented and tested.

## Must-have by the end

- One command starts backend and one starts frontend.
- A gallery with at least three preloaded flood images plus upload support.
- Visible image annotations; label vehicle boxes as model-detected and waterline/reference marks as automatic or operator-confirmed.
- A working reference geometry calculation and a clear abstain state for invalid/insufficient geometry.
- A configurable, explicitly illustrative risk policy.
- Incident/event panel and a route visual that marks a blocked segment and shows an alternative.
- A reliable scripted demo that runs without downloading a new model or map tiles live.

## Defer until the core works

- Training a new model.
- Real CCTV RTSP integration; use uploaded images or a prerecorded local clip first.
- Live municipal APIs, live emergency dispatch, complex routing backends, VLM verification, authentication, database deployment, Docker, or cloud hosting.
- Claims of centimetre accuracy or guaranteed vehicle passability.

## Four-person work split

### Person 1 — Tech lead / integrator (suggested: you)
- Own the API contract, Git integration, working demo path, final acceptance checklist, and judging narrative.
- Keep main branch runnable. Merge only tested branches.
- Build the demo sequence and backup screen recording/static outputs.
- Don't become the sole implementer of every module; delegate the bounded tasks below.

### Person 2 — Computer vision
- First: run pretrained vehicle detection on the supplied images.
- Return boxes, classes, confidence, and annotated image. Keep file handling separate from the frontend.
- Add water segmentation only if it works on multiple representative images; otherwise implement a clearly labelled manual waterline/polygon correction flow with an abstain fallback.
- Test one dry negative image and multiple different flood scenes.

### Person 3 — Frontend / interaction
- Build the dark municipal dashboard: image gallery/upload, canvas overlay, reference object selection, waterline editing, depth badge, vehicle-risk cards, incident feed.
- Make operator clicks usable and show when a value is manual/illustrative.
- Use a static demo path when backend/AI is unavailable so the main flow never crashes.

### Person 4 — Backend / risk / map
- Own FastAPI endpoints, typed response schemas, configurable policy, camera/event mock data, incident log.
- Add route display: blocked segment (red) and alternative route (green). Start with a schematic/local route; online tiles are a bonus.
- Write API tests and provide fixture JSON for the frontend.

## Timebox

### T- Tonight (before hackathon)
- Confirm every teammate can run Git, Node, Python, and the app scaffold.
- Install Python core requirements and Node dependencies.
- Run the CUDA check, but do not spend more than 30 minutes repairing GPU setup; CPU + precomputed results are acceptable for a static-image demo.
- Open the bundled flood photos and verify they load. Add one dry-road negative image from a source you can use.
- Create the repository, branches, `.gitignore`, and a shared API contract.
- Download/cache the small pretrained vehicle model if your connection allows; test on one image, save output, and stop if it stalls.

### Hours 0–1: Lock scope and prove the skeleton
- Each teammate says what will be working at hour 4.
- Start backend health route and blank dashboard.
- Agree on JSON schema before parallel coding.
- Tag a baseline commit.

### Hours 1–4: Independent foundations
- Vision: inference script producing bounding boxes plus saved annotated JPEG.
- Frontend: image gallery/upload and canvas that displays a selected image.
- Backend: health + depth-estimate + policy endpoints and tests.
- Integrator: end-to-end local launch instructions and realistic demo sequence.

### Hours 4–8: First vertical slice
- User selects image → API/vision runs → annotations visible → depth endpoint returns JSON → risk card renders.
- At hour 8, stop adding features until that path works.

### Hours 8–12: Make the interaction real
- Operator can adjust reference top/base/waterline or select a simple water polygon.
- The app shows the ratio/calculation, whether the reference is sufficient, and why it might abstain.
- An incident can be created with camera ID, time, depth estimate, and road status.

### Hours 12–16: Close the loop
- Road-status panel and blocked/detour route visualization.
- Policy values configurable from a JSON file or UI.
- Add sample cases for low/moderate/high risk, each clearly marked illustrative.

### Hours 16–19: Reliability pass
- Test bad uploads, no vehicle detected, unclear waterline, missing CUDA, model unavailable, API down, and offline map.
- Keep static/fallback demo state if any optional service fails.
- Don't add new major dependencies after hour 16.

### Hours 19–22: Judge-ready polish
- Clean typography, labels, confidence/source tags, clear status flow.
- Save annotated output images and a 30–60 second backup screen recording.
- Prepare an architecture diagram and 2-minute demo narration.

### Hours 22–24: Freeze and rehearse
- Freeze features.
- Start clean on a fresh terminal and follow the README exactly.
- Rehearse twice; rehearse once without internet.
- Ensure no fabricated metrics are displayed as measured results.

## Definition of done

- [ ] Starts from clean terminal using documented commands.
- [ ] At least 3 local flood sample images load offline.
- [ ] A dry/negative scene does not automatically become a high-risk alert.
- [ ] Depth math is testable and visible.
- [ ] Invalid/uncertain geometry yields `ABSTAIN / REVIEW`, not a fabricated number.
- [ ] Risk thresholds are configurable and explicitly illustrative.
- [ ] Road status can create an incident and show a detour.
- [ ] There is a static fallback in case model inference fails.
- [ ] Every teammate can explain the piece they owned and the current limitations.
