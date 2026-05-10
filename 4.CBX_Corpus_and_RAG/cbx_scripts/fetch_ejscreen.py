import requests
import json
from pathlib import Path

url = "https://ejamapi-84652557241.us-central1.run.app/report"

shape = {
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {},
    "geometry": {
      "coordinates": [[
        [-73.935238, 40.850555],
        [-73.924767, 40.847308],
        [-73.90411,  40.848564],
        [-73.887573, 40.846399],
        [-73.876987, 40.838781],
        [-73.864742, 40.838045],
        [-73.839737, 40.829646],
        [-73.843742, 40.82497 ],
        [-73.884197, 40.820726],
        [-73.898903, 40.808471],
        [-73.903366, 40.804486],
        [-73.91481,  40.797166],
        [-73.921963, 40.799678],
        [-73.916927, 40.806738],
        [-73.906227, 40.811199],
        [-73.892609, 40.823194],
        [-73.890835, 40.832893],
        [-73.88763,  40.836963],
        [-73.89587,  40.840512],
        [-73.915382, 40.841638],
        [-73.931862, 40.840945],
        [-73.940445, 40.844538],
        [-73.935238, 40.850555]
      ]],
      "type": "Polygon"
    }
  }]
}

params = {
    'shape': json.dumps(shape),
    'buffer': 0
}

print("Fetching EJScreen data for CBE corridor...")
r = requests.get(url, params=params, timeout=60)

print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('content-type')}")

# Save raw response regardless of format
raw_path = Path('./cbx_corpus/ejscreen_cbx_raw.json')
with open(raw_path, 'wb') as f:
    f.write(r.content)
print(f"Raw response saved: {len(r.content):,} bytes")

# Try to parse as JSON
try:
    data = r.json()
    
    # Save formatted JSON
    json_path = Path('./cbx_corpus/ejscreen_cbx_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"JSON saved: {json_path}")
    
    # Convert to readable text for RAG
    txt_path = Path('./cbx_corpus/ejscreen_cbx_corridor.txt')
    
    def flatten_to_text(obj, prefix='', lines=None):
        if lines is None:
            lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                flatten_to_text(v, 
                    f"{prefix}{k}" if prefix == '' 
                    else f"{prefix} — {k}", lines)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                flatten_to_text(item, 
                    f"{prefix} [{i}]", lines)
        else:
            if obj is not None and str(obj).strip():
                lines.append(f"{prefix}: {obj}")
        return lines
    
    lines = flatten_to_text(data)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("EJScreen Report — Cross Bronx "
                "Expressway Corridor\n")
        f.write("Source: EJAM API\n")
        f.write("Boundary: CBE corridor polygon\n")
        f.write("="*50 + "\n\n")
        f.write("\n".join(lines))
    
    print(f"TXT saved: {txt_path}")
    print(f"Total fields extracted: {len(lines)}")
    
    # Print first 20 lines to verify content
    print("\nFirst 20 fields:")
    for line in lines[:20]:
        print(f"  {line}")

except Exception as e:
    print(f"Not JSON or parse error: {e}")
    # Try as plain text
    try:
        text = r.content.decode('utf-8')
        txt_path = Path('./cbx_corpus/ejscreen_cbx_corridor.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("EJScreen Report — CBE Corridor\n")
            f.write("="*50 + "\n\n")
            f.write(text)
        print(f"Saved as plain text: {txt_path}")
        print(f"First 500 chars:\n{text[:500]}")
    except Exception as e2:
        print(f"Could not decode response: {e2}")