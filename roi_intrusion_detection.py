import cv2, numpy as np, os, re, time
from ultralytics import YOLO


YOLO_MODEL = YOLO("yolov8n.pt")   
SCRIPT_TXT = "cv_hw2_mp4script.txt"
INPUT_DIR  = "."
OUTPUT_DIR = "Result"
EVENT_TXT  = "event_detection_result.txt"


NIGHT_VIDEOS = {"test_snownight.mp4", "test_snownight_1.mp4"}


CONF_TH       = 0.35   
INSIDE_FRAC   = 0.85   
ON_SEC        = 1.0    
OFF_SEC       = 0.5   
COAST         = 2.5    
MATCH_DIST    = 120    
STATIC_SPREAD = 60    
SKIP          = 2     


def clahe(f):
    
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def detect(frame):
   
    r = YOLO_MODEL(frame, conf=CONF_TH, classes=[0], verbose=False)[0]
    out = []
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(box.conf[0])))
    return out


def inside_frac(box, roi):
  
    x, y, w, h = box[:4]
    if w <= 0 or h <= 0:
        return 0.0
    inside = total = 0
    for fx in (0.1, 0.5, 0.9):           
        for fy in (0.0, 0.25, 0.5, 0.75, 1.0):  
            px, py = x + fx * w, y + fy * h
            total += 1
            if cv2.pointPolygonTest(roi, (float(px), float(py)), False) >= 0:
                inside += 1
    return inside / total

def is_in(box, roi):
    return inside_frac(box, roi) >= INSIDE_FRAC

def center(box):
    x, y, w, h = box[:4]
    return (x + w // 2, y + h // 2)

def fmt(sec):
    return f"{int(sec)//60:02d}:{int(sec)%60:02d}"


def process(vid_path, roi, out_path, night):
    roi = np.array(roi, dtype=np.int32)
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (320, 180))

    tracks = []
    tid = [0]
    alarm = False
    pend = None
    events = []
    inc = outc = 0
    on_need = int(fps * ON_SEC)
    off_need = int(fps * OFF_SEC)
    coast_frames = int(fps * COAST)

    def moving(t):
        h = t.get('hist', [])
        if not night:
            return True              
        if len(h) < 8:
            return False           
        xs = [p[0] for p in h]
        ys = [p[1] for p in h]
        return (max(xs) - min(xs)) + (max(ys) - min(ys)) >= STATIC_SPREAD

    for fi in range(total):
        ret, fr = cap.read()
        if not ret:
            break

        if fi % SKIP == 0:
        
            proc = clahe(fr) if night else fr
            dets = detect(proc)


            um = list(range(len(dets)))
            for t in tracks:
                tc = center(t['box'])
                best, bd = -1, MATCH_DIST
                for di in um:
                    dc = center(dets[di])
                    d = ((tc[0]-dc[0])**2 + (tc[1]-dc[1])**2) ** 0.5
                    if d < bd:
                        bd, best = d, di
                if best >= 0:                  
                    t['box'] = dets[best]
                    t['coast'] = coast_frames
                    t['in_roi'] = is_in(dets[best], roi)
                    h = t.get('hist', []); h.append(center(dets[best])); t['hist'] = h[-30:]
                    um.remove(best)
                else:                               
                    t['coast'] -= SKIP
           
            for di in um:
                tid[0] += 1
                tracks.append({'id': tid[0], 'box': dets[di], 'coast': coast_frames,
                               'in_roi': is_in(dets[di], roi), 'hist': [center(dets[di])]})
            tracks = [t for t in tracks if t['coast'] > 0]
        else:
            for t in tracks:
                t['coast'] -= 1
            tracks = [t for t in tracks if t['coast'] > 0]

        
        any_in = any(t['in_roi'] and moving(t) for t in tracks)
        if any_in:
            inc += 1; outc = 0
        else:
            outc += 1; inc = 0
        if not alarm and inc >= on_need:           
            alarm = True
            pend = f"Alarm ON  : {fmt(fi/fps)}"
            inc = 0
        if alarm and outc >= off_need:             
            alarm = False
            if pend:
                events.append([pend, f"Alarm OFF : {fmt(fi/fps)}"])
                pend = None
            outc = 0

     
        vis = fr.copy()
        ov = vis.copy(); cv2.fillPoly(ov, [roi], (0, 200, 0))
        vis = cv2.addWeighted(ov, 0.18, vis, 0.82, 0)
        cv2.polylines(vis, [roi], True, (0, 220, 0), 2)
        for t in tracks:
            if not moving(t):                    
                continue
            x, y, w, h = t['box'][:4]
            ins = t['in_roi']
            col = (0, 0, 255) if ins else (0, 140, 255)
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
            cv2.putText(vis, ("IN " if ins else "") + f"ID{t['id']}", (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        cv2.putText(vis, "!!ALARM ON!!" if alarm else "ALARM OFF", (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if alarm else (180, 180, 180), 2)
        cv2.putText(vis, fmt(fi/fps), (6, oh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 200), 1)
        wr.write(cv2.resize(vis, (320, 180)))

    if alarm and pend:
        events.append([pend, f"Alarm OFF : {fmt(total/fps)}"])
    cap.release(); wr.release()
    print(f"DONE {os.path.basename(vid_path)}: {len(events)} events")
    for e in events:
        print(f"  {e[0]} / {e[1]}")
    return events


def parse_script(path):
    """cv_hw2_mp4script.txt → [(mp4명, np.array([[x,y],...])), ...]"""
    items, cur, pts = [], None, []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        if line.lower().endswith(".mp4"):
            if cur and pts:
                items.append((cur, np.array(pts, dtype=np.int32)))
            cur, pts = line, []
        else:
            m = re.findall(r"(\d+)\s*,\s*(\d+)", line)
            if m:
                pts.append([int(m[0][0]), int(m[0][1])])
    if cur and pts:
        items.append((cur, np.array(pts, dtype=np.int32)))
    return items

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    items = parse_script(SCRIPT_TXT)
    all_events = {}
    for name, roi in items:
        vid = os.path.join(INPUT_DIR, name)
        if not os.path.exists(vid):
            print(f"[skip] {name} 없음")
            continue
        out = os.path.join(OUTPUT_DIR, name.replace(".mp4", "_box.mp4"))
        night = name in NIGHT_VIDEOS
        print(f"처리중: {name} (night={night}) ...")
        t0 = time.time()
        ev = process(vid, roi, out, night)
        all_events[name] = ev
        print(f"  완료 {time.time()-t0:.0f}s -> {out}\n")

    with open(EVENT_TXT, "w", encoding="utf-8") as f:
        for name, ev in all_events.items():
            f.write(f"[{name}]\n")
            if ev:
                for on, off in ev:
                    f.write(f"{on}    {off}\n")
            else:
                f.write("(침입 이벤트 없음)\n")
            f.write("\n")
    print(f"결과 저장: {EVENT_TXT}")

if __name__ == "__main__":
    main()
