# Image/Data Plan for the Hackathon

## First: use the bundled images

`assets/demo_images/` contains three flood photographs extracted from the team's existing AquaGauge presentation. They are ready for upload/visualisation smoke tests; they are **not** annotated ground-truth data. The source/licence for each embedded photo was not recorded in the PPTX metadata, so confirm permission before redistributing/publishing them.

Make an image set of 6–10 representative frames before the event:

1. shallow water / mostly visible wheel,
2. moderate wheel submersion,
3. deep water / vehicle partly obscured,
4. dry road negative,
5. reflective wet road but no standing floodwater,
6. poor lighting or occlusion,
7. curb visible with clear waterline,
8. curb/wheel missing or ambiguous (expected abstention).

Keep a tiny `demo_manifest.csv` with `filename, source, rights, scenario, reference_height_cm, waterline_status, expected_demo_status`. Do not invent measured depths; set expected depth blank unless independently labelled.

## Dataset options (verified online)

### FloodDET — closest task match
- URL: https://data.mendeley.com/datasets/px48m7vwvf/2
- 10,262 images, COCO-style annotations, non-flooded negatives, and 11 levels of visible submersion for person, bicycle, car, and bus.
- Dataset page lists CC BY 4.0.
- Current archive reported around 5.74 GB; **do not download the full archive unless bandwidth/storage is known to be adequate**. It is useful for later experiments, not a reason to train overnight.
- Cite: Singha & Neog (2026), “FloodDET,” Mendeley Data, DOI 10.17632/px48m7vwvf.2.

### Floodwater Dataset — for binary water masks, domain mismatch
- URL: https://decide.ugent.be/datasets/floodwater/
- 700 manually annotated image/mask pairs plus a much larger oblique UAV video dataset.
- The video archive is about 5.3 GB; data are from flood-affected areas in Belgium and use aerial/oblique UAV views, not fixed road CCTV.
- Page lists GPL-3.0. Use primarily for research/prototyping, and review the terms before reuse.

### FloodNet — useful but aerial
- URL: https://github.com/BinaLab/FloodNet-Supervised_v1.0
- 2,343 post-Hurricane Harvey UAV images, with classes including flooded road, water, vehicle and background.
- Repo states the dataset is released under the Community Data License Agreement (permissive).
- The aerial viewpoint differs from a fixed urban CCTV camera, so don't claim it validates street-level depth measurement.

## Model choice for a 24-hour build

- **Vehicle objects:** a small pretrained detector (for example, a small Ultralytics model) can identify cars/buses/motorcycles. Test it on the team's images before integrating. It will not automatically identify the flood-water boundary.
- **Water mask:** do not assume a generic COCO detector labels water. Options: use an existing water segmentation model only if it works on several test scenes; otherwise let the operator draw/adjust a water polygon/line and make it a clearly stated human-in-the-loop fallback.
- **Depth:** use reference geometry with explicit site calibration and editable top/base/waterline coordinates. If the reference is not sufficiently visible, return `ABSTAIN / REVIEW`.
- **Demo:** label outputs as “estimated” and use a “demo policy” badge. No claims of validated centimetre accuracy from three sample pictures.

## Pre-event image sanity checklist

- [ ] Images open locally without internet.
- [ ] No one relies on a link that requires login on demo day.
- [ ] There is at least one dry/negative image.
- [ ] UI can handle portrait/landscape sizes and corrupt uploads.
- [ ] Masks/boxes have visible provenance labels: `MODEL`, `OPERATOR`, or `PRECOMPUTED DEMO`.
- [ ] Original sources and licence are retained in the manifest.
