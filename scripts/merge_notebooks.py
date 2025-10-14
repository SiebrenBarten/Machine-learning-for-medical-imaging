#!/usr/bin/env python3
import sys
import json

def find_marker_index(cells, marker):
    marker = marker.lower()
    for i, cell in enumerate(cells):
        if cell.get('cell_type','').lower() == 'markdown':
            src = ''.join(cell.get('source',[])).lower()
            if marker in src:
                return i
    return None

def main():
    if len(sys.argv) != 5:
        print('Usage: merge_notebooks.py <ours.ipynb> <theirs.ipynb> <out.ipynb> <marker>')
        print('Example marker: "MNIST synthesis" or "Conditional image synthesis"')
        sys.exit(2)

    ours_path, theirs_path, out_path, marker = sys.argv[1:5]

    with open(ours_path, 'r', encoding='utf-8') as f:
        ours = json.load(f)
    with open(theirs_path, 'r', encoding='utf-8') as f:
        theirs = json.load(f)

    ours_cells = ours.get('cells', [])
    theirs_cells = theirs.get('cells', [])

    ours_idx = find_marker_index(ours_cells, marker)
    theirs_idx = find_marker_index(theirs_cells, marker)

    if ours_idx is None:
        print('Error: could not find "MNIST synthesis" cell in ours (Wessel) notebook')
        sys.exit(3)
    if theirs_idx is None:
        print('Error: could not find "MNIST synthesis" cell in theirs (Siebren_week5) notebook')
        sys.exit(3)

    # Keep up to and including the marker cell from ours
    merged_cells = []
    merged_cells.extend(ours_cells[:ours_idx+1])

    # Take the remainder after the MNIST synthesis cell from theirs
    merged_cells.extend(theirs_cells[theirs_idx+1:])

    merged = dict(ours)
    merged['cells'] = merged_cells

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f'Merged notebook written to {out_path}')

if __name__ == '__main__':
    main()
