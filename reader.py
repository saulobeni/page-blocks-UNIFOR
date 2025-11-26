
import re
from typing import Tuple, List


def read_arff(path: str) -> Tuple[List[str], list, list]:
    """Reads a simple ARFF file. Returns (attr_names, X_list, y_list).
    X_list is list of lists (floats), y_list is list of class labels (as strings)."""
    attr_names = []
    data_started = False
    X = []
    y = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '' or line.startswith('%'):
                continue
            if not data_started:
                if line.lower().startswith('@attribute'):
                    m = re.match(r"@attribute\s+('?\"?)([^\'\"\s]+)\1\s+(.*)", line, re.I)
                    if not m:
                        parts = line.split()
                        if len(parts) >= 2:
                            name = parts[1].strip("'\"")
                        else:
                            continue
                    else:
                        name = m.group(2)
                    attr_names.append(name)
                elif line.lower().startswith('@data'):
                    data_started = True
                else:
                    continue
            else:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 0:
                    continue
                features = parts[:-1]
                label = parts[-1]
                row = []
                for v in features:
                    try:
                        row.append(float(v))
                    except:
                        row.append(0.0)
                X.append(row)
                y.append(label)
    return attr_names, X, y