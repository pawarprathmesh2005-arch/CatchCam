import json, glob, collections
for f in glob.glob("outputs/*_observations.json"):
    rows = json.load(open(f, encoding="utf-8-sig")); by = collections.defaultdict(list)
    for r in rows: by[r["track_id"]].append(r)
    print(f, len(rows), "rows,", len(by), "tracks")
    for t, rs in sorted(by.items()):
        rs.sort(key=lambda r: r["frame"]); b = [r["bbox_xyxy"] for r in rs]
        h = sum(x[3]-x[1] for x in b)/len(b); cx = [(x[0]+x[2])/2 for x in b]; cy = [(x[1]+x[3])/2 for x in b]
        mv = ((max(cx)-min(cx))**2 + (max(cy)-min(cy))**2) ** .5
        print(f"track {t:4d} {rs[0]['timestamp']:7.1f}s-{rs[-1]['timestamp']:7.1f}s frames {len(rs):4d} conf {sum(r['confidence'] for r in rs)/len(rs):.2f} height {h:5.0f}px moved {mv:5.0f}px")
