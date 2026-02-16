"""
Multi-Label Image Labeler — Head, Shemagh, Right Place
Run: python3 labeler.py
Open: http://localhost:8765
Keyboard: H=Head, S=Shemagh, R=Right Place, →=Next, ←=Prev, J=Jump unlabeled
Results saved to labels.txt (format: filename,head,shemagh,right_place)
"""
import http.server
import json
import os
import urllib.parse

TEST_DIR = "data/dal-shemagh-detection-challenge/images/test"
OUTPUT_FILE = "labels.txt"

images = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')],
                key=lambda x: int(x.replace('.jpg','')))

# Load existing labels: {filename: {head:0/1, shemagh:0/1, right_place:0/1}}
labels = {}
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) == 4:
                    labels[parts[0]] = {
                        'head': int(parts[1]),
                        'shemagh': int(parts[2]),
                        'right_place': int(parts[3])
                    }

print(f"Found {len(images)} images, {len(labels)} already labeled")

HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Shemagh Labeler</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { 
    background: #0d1117; color: #fff; font-family: 'Segoe UI', sans-serif;
    display: flex; flex-direction: column; align-items: center; height: 100vh;
    user-select: none;
  }
  .top-bar {
    width: 100%; padding: 10px 24px; background: #161b22;
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 2px solid #30363d;
  }
  .filename { color: #58a6ff; font-size: 22px; font-weight: bold; }
  .progress { color: #f0883e; font-size: 16px; font-weight: bold; }
  .stats { color: #8b949e; font-size: 13px; }
  
  .image-container {
    flex: 1; display: flex; align-items: center; justify-content: center;
    padding: 12px; max-height: calc(100vh - 200px); overflow: hidden;
  }
  .image-container img {
    max-width: 100%; max-height: 100%; object-fit: contain;
    border-radius: 8px; box-shadow: 0 8px 32px rgba(0,0,0,0.6);
  }
  
  .controls {
    width: 100%; padding: 12px 24px; background: #161b22;
    border-top: 2px solid #30363d;
    display: flex; justify-content: center; align-items: center; gap: 20px;
  }
  
  .toggle-group {
    display: flex; flex-direction: column; align-items: center; gap: 4px;
  }
  .toggle-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
  .toggle-btn {
    width: 120px; height: 50px; border: 2px solid #30363d; border-radius: 10px;
    font-size: 16px; font-weight: bold; cursor: pointer; transition: all 0.15s;
    display: flex; align-items: center; justify-content: center; gap: 6px;
  }
  .toggle-btn:hover { transform: scale(1.05); }
  
  .toggle-off { background: #21262d; color: #8b949e; }
  .toggle-head-on { background: #1f6feb; color: #fff; border-color: #58a6ff; box-shadow: 0 0 12px rgba(31,111,235,0.4); }
  .toggle-shem-on { background: #238636; color: #fff; border-color: #3fb950; box-shadow: 0 0 12px rgba(35,134,54,0.4); }
  .toggle-rp-on { background: #da3633; color: #fff; border-color: #f85149; box-shadow: 0 0 12px rgba(218,54,51,0.4); }
  
  .nav-btn {
    padding: 12px 28px; border: 2px solid #30363d; border-radius: 10px;
    background: #21262d; color: #c9d1d9; font-size: 16px; font-weight: bold;
    cursor: pointer; transition: all 0.15s;
  }
  .nav-btn:hover { background: #30363d; transform: scale(1.05); }
  
  .divider { width: 1px; height: 50px; background: #30363d; }
  
  .kbd { 
    display: inline-block; padding: 2px 6px; background: #30363d; 
    border-radius: 4px; font-size: 11px; color: #8b949e; margin-top: 2px;
  }
  .bottom-hint {
    padding: 6px; text-align: center; color: #484f58; font-size: 12px;
  }
</style>
</head>
<body>
  <div class="top-bar">
    <span class="filename" id="filename">Loading...</span>
    <div style="text-align:right">
      <span class="progress" id="progress">0 / 0</span>
      <br><span class="stats" id="stats">Labeled: 0</span>
    </div>
  </div>
  
  <div class="image-container">
    <img id="img" src="" alt="Test Image">
  </div>
  
  <div class="controls">
    <button class="nav-btn" onclick="nav(-10)">⏪ -10</button>
    <button class="nav-btn" onclick="nav(-1)">← Prev</button>
    
    <div class="divider"></div>
    
    <div class="toggle-group">
      <div class="toggle-label">Head</div>
      <button class="toggle-btn toggle-off" id="btnHead" onclick="toggle('head')">
        👤 NO
      </button>
      <span class="kbd">H</span>
    </div>
    
    <div class="toggle-group">
      <div class="toggle-label">Shemagh</div>
      <button class="toggle-btn toggle-off" id="btnShem" onclick="toggle('shemagh')">
        🧣 NO
      </button>
      <span class="kbd">S</span>
    </div>
    
    <div class="toggle-group">
      <div class="toggle-label">Right Place</div>
      <button class="toggle-btn toggle-off" id="btnRP" onclick="toggle('right_place')">
        ✅ NO
      </button>
      <span class="kbd">R</span>
    </div>
    
    <div class="divider"></div>
    
    <button class="nav-btn" onclick="nav(1)">Next →</button>
    <button class="nav-btn" onclick="nav(10)">+10 ⏩</button>
  </div>
  <div class="bottom-hint">
    Keyboard: <b>H</b>=Head  <b>S</b>=Shemagh  <b>R</b>=Right Place  <b>←→</b>=Navigate  <b>J</b>=Jump unlabeled  <b>Space</b>=Next
  </div>
  
<script>
let idx = INITIAL_IDX;
const images = IMAGES_JSON;
const labels = LABELS_JSON;
const total = images.length;

function getLabel(fname) {
  return labels[fname] || null;
}

function render() {
  const fname = images[idx];
  document.getElementById('filename').textContent = fname;
  document.getElementById('img').src = '/img/' + fname;
  document.getElementById('progress').textContent = (idx+1) + ' / ' + total;
  
  const lbl = getLabel(fname);
  const h = lbl ? lbl.head : 0;
  const s = lbl ? lbl.shemagh : 0;
  const r = lbl ? lbl.right_place : 0;
  
  const bh = document.getElementById('btnHead');
  bh.className = 'toggle-btn ' + (h ? 'toggle-head-on' : 'toggle-off');
  bh.innerHTML = '👤 ' + (h ? 'YES' : 'NO');
  
  const bs = document.getElementById('btnShem');
  bs.className = 'toggle-btn ' + (s ? 'toggle-shem-on' : 'toggle-off');
  bs.innerHTML = '🧣 ' + (s ? 'YES' : 'NO');
  
  const br = document.getElementById('btnRP');
  br.className = 'toggle-btn ' + (r ? 'toggle-rp-on' : 'toggle-off');
  br.innerHTML = '✅ ' + (r ? 'YES' : 'NO');
  
  let labeled = Object.keys(labels).length;
  let headYes = 0, shemYes = 0, rpYes = 0;
  for (const f in labels) {
    if (labels[f].head) headYes++;
    if (labels[f].shemagh) shemYes++;
    if (labels[f].right_place) rpYes++;
  }
  document.getElementById('stats').textContent = 
    'Labeled: ' + labeled + '/' + total + ' | Heads: ' + headYes + ' | Shemagh: ' + shemYes + ' | RightPlace: ' + rpYes;
}

function toggle(field) {
  const fname = images[idx];
  if (!labels[fname]) labels[fname] = {head:0, shemagh:0, right_place:0};
  labels[fname][field] = labels[fname][field] ? 0 : 1;
  
  // Auto-logic: if right_place ON, head and shemagh must be ON
  if (field === 'right_place' && labels[fname].right_place === 1) {
    labels[fname].head = 1;
    labels[fname].shemagh = 1;
  }
  
  save(fname);
  render();
}

function save(fname) {
  fetch('/label', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({filename: fname, label: labels[fname]})
  });
}

function nav(delta) {
  // Auto-save current as "no labels" if untouched (mark as labeled with all NO)
  const fname = images[idx];
  if (!labels[fname]) {
    labels[fname] = {head:0, shemagh:0, right_place:0};
    save(fname);
  }
  idx = Math.max(0, Math.min(total-1, idx + delta));
  render();
}

function jumpUnlabeled() {
  for (let i = idx+1; i < total; i++) {
    if (!labels[images[i]]) { idx = i; render(); return; }
  }
  for (let i = 0; i <= idx; i++) {
    if (!labels[images[i]]) { idx = i; render(); return; }
  }
  alert('All ' + total + ' images labeled!');
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'h' || e.key === 'H') toggle('head');
  else if (e.key === 's' || e.key === 'S') toggle('shemagh');
  else if (e.key === 'r' || e.key === 'R') toggle('right_place');
  else if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); nav(1); }
  else if (e.key === 'ArrowLeft') nav(-1);
  else if (e.key === 'j' || e.key === 'J') jumpUnlabeled();
});

// Preload next image
setInterval(() => {
  if (idx < total - 1) {
    const next = new Image();
    next.src = '/img/' + images[idx+1];
  }
}, 500);

render();
</script>
</body>
</html>
"""

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    
    def do_GET(self):
        if self.path == '/':
            initial_idx = 0
            for i, fname in enumerate(images):
                if fname not in labels:
                    initial_idx = i
                    break
            
            html = HTML.replace('IMAGES_JSON', json.dumps(images))
            html = html.replace('LABELS_JSON', json.dumps(labels))
            html = html.replace('INITIAL_IDX', str(initial_idx))
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
        
        elif self.path.startswith('/img/'):
            fname = urllib.parse.unquote(self.path[5:])
            fpath = os.path.join(TEST_DIR, fname)
            if os.path.exists(fpath):
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Cache-Control', 'max-age=3600')
                self.end_headers()
                with open(fpath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/label':
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length))
            fname = data['filename']
            lbl = data['label']
            labels[fname] = lbl
            
            with open(OUTPUT_FILE, 'w') as f:
                f.write("# filename,head,shemagh,right_place\n")
                for img in images:
                    if img in labels:
                        l = labels[img]
                        f.write(f"{img},{l['head']},{l['shemagh']},{l['right_place']}\n")
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()

print(f"\n🏷️  Shemagh Labeler running at http://localhost:8765")
print(f"   {len(images)} images | {len(labels)} already labeled")
print(f"   Keys: H=Head, S=Shemagh, R=Right Place")
print(f"   Navigation: ←→, Space=Next, J=Jump unlabeled")
print(f"   Press Ctrl+C to stop\n")

server = http.server.HTTPServer(('0.0.0.0', 8765), Handler)
server.serve_forever()
