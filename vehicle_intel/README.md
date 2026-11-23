# Vehicle intelligence (type, colour, number plate)

1. Put `vehicle_intel/` AND `routes_vehicles.py` next to `api.py`. Your api.py already auto-loads `routes_vehicles`.
2. Install: `pip install ultralytics opencv-python easyocr`  (or `pytesseract` + the Tesseract program)
3. For each camera video:
       python -m vehicle_intel.process_video --video cameras/cam1_topleft.mp4 --camera cam1 --out outputs/ --time_offset 0
   (use the same camera id and the same --time_offset you gave track.py, so vehicle and person times line up)
   -> writes outputs/cam1_vehicles.json (one row per tracked vehicle: type, colour, plate, first/last time)
4. Restart the API; http://localhost:8000/modules should list routes_vehicles.

Tests (no YOLO weights needed): `python -m vehicle_intel.test_process_video` and `pytest -q vehicle_intel`.

## Endpoints
- GET /vehicles?vehicle_type=car&color=red&plate=MH12          filter / search (plate: partial, or full with OCR-error tolerance)
- GET /vehicles/journey?plate=MH12AB1234                       same vehicle across cameras, time-ordered, with gap analysis

## How it works
- Type: YOLO class (car / motorcycle / bus / truck), majority vote over the track.
- Colour: HSV pixel vote on the central body region, weighted vote over the track.
- Plate: classical plate localisation -> OCR -> Indian-format cleaning (fixes O/0, I/1, S/5, B/8 by position, checks the
  state code) -> vote across frames; valid plates beat confident garbage.

## Known limits (state these in your report)
- Tested on synthetic images and a simulated detector only; never on real CCTV footage. Real CCTV plates need enough pixels (roughly 100+ px plate width) and a front/rear view.
- Plate localisation is classical (edges + shape). A trained plate detector would raise recall a lot.
- Colour is affected by lighting (night / sodium lamps) and glass/shadows; names are coarse.
- Plate/vehicle data is personal data: restrict access and keep to authorised footage.
