import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import cv2 


# Les inn bildet
path_image = "./Filtered_sudoku_Images/CNN_edge_detection.jpg"
image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)

# Juster Shi-Tomasi for å finne hjørner
corners = cv2.goodFeaturesToTrack(image, maxCorners=5000, qualityLevel=0.01, minDistance=15)

points = np.float32([i.ravel() for i in corners])

def lines_row_10bins(points, n_bins=10):
    """
    Deler punktsettet i 'n_bins' like store intervaller
    basert på y-verdien, og fitter én lineær regresjon i hver bin.
    
    Returnerer en liste av linjestykker:
      [((x1, y1), (x2, y2)), ((x3, y3), (x4, y4)), ...]
    hvor hvert linjestykke representerer én "rad".
    """
    lines = []

    # 1) Finn min/max y i hele datasettet
    y_all = points[:, 1]
    y_min_total = np.min(y_all)
    y_max_total = np.max(y_all)

    # 2) Lag bin-grenser (n_bins = 10 => 10 intervaller)
    bin_edges = np.linspace(y_min_total, y_max_total, n_bins + 1)
    # bin_edges = [ y_min_total, ..., y_max_total ]

    # 3) For hver bin => saml punkter, kjør regresjon
    for i in range(n_bins):
        low_edge  = bin_edges[i]
        high_edge = bin_edges[i + 1]

        # Filtrer punkter hvis y ligger i [low_edge, high_edge)
        bin_coords = points[(y_all >= low_edge) & (y_all < high_edge)]

        if len(bin_coords) < 2:
            # For få punkter til å definere en linje
            continue

        # X = x-verdier (2D), Y = y-verdier (1D)
        X = bin_coords[:, 0].reshape(-1, 1)
        Y = bin_coords[:, 1]

        model = LinearRegression()
        model.fit(X, Y)

        x_min, x_max = np.min(X), np.max(X)
        y_min = model.predict([[x_min]])[0]
        y_max = model.predict([[x_max]])[0]

        lines.append(((x_min, y_min), (x_max, y_max)))

    return lines

# KMeans(n_clusters=10) 

def lines_col_10bins(points, n_bins=10):
    """
    Deler punktsettet i 'n_bins' basert på x-verdien,
    og fitter lineær regresjon x = a*y + b i hver bin.
    """
    lines = []

    x_all = points[:, 0]
    x_min_total = np.min(x_all)
    x_max_total = np.max(x_all)

    bin_edges = np.linspace(x_min_total, x_max_total, n_bins + 1)

    for i in range(n_bins):
        low_edge  = bin_edges[i]
        high_edge = bin_edges[i + 1]

        # Punkter hvis x-verdi ligger i [low_edge, high_edge)
        bin_coords = points[(x_all >= low_edge) & (x_all < high_edge)]

        if len(bin_coords) < 2:
            continue

        # Nå: X = y-verdier (2D), Y = x-verdier (1D)
        X = bin_coords[:, 1].reshape(-1, 1)
        Y = bin_coords[:, 0]

        model = LinearRegression()
        model.fit(X, Y)

        y_min, y_max = np.min(X), np.max(X)
        x_min = model.predict([[y_min]])[0]
        x_max = model.predict([[y_max]])[0]

        lines.append(((x_min, y_min), (x_max, y_max)))

    return lines


def lines_row_kmeans(points, n_clusters=10):
    """
    Finner 10 rader basert på KMeans-klustering av y-verdiene,
    og fitter lineær regresjon for hver av disse gruppene.
    """
    lines = []
    
    # 1) Tren KMeans-modell på y-verdiene
    y_values = points[:, 1].reshape(-1, 1)  # Formater y som 2D-array
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(y_values)
    
    # 2) Få klyngesentrene og sorter dem (fra topp til bunn i bildet)
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())

    # 3) Gruppér punktene basert på hvilken klynge de tilhører
    labels = kmeans.labels_
    grouped_points = {c: [] for c in cluster_centers}

    for (x, y), label in zip(points, labels):
        grouped_points[cluster_centers[label]].append((x, y))

    # 4) Fitter lineær regresjon for hver y-klynge
    for y_val, coords in grouped_points.items():
        if len(coords) < 2:
            continue  # For få punkter til å lage linje

        X = np.array([c[0] for c in coords]).reshape(-1, 1)
        Y = np.array([c[1] for c in coords])

        model = LinearRegression()
        model.fit(X, Y)

        x_min, x_max = np.min(X), np.max(X)
        y_min = model.predict([[x_min]])[0]
        y_max = model.predict([[x_max]])[0]

        lines.append(((x_min, y_min), (x_max, y_max)))

    return lines

# KMeans(n_clusters=10) 

def lines_col_kmeans(points, n_clusters=10):
    """
    Finner 10 kolonner basert på KMeans-klustering av x-verdiene,
    og fitter lineær regresjon for hver av disse gruppene.
    """
    lines = []
    
    # 1) Tren KMeans-modell på x-verdiene
    x_values = points[:, 0].reshape(-1, 1)  # Formater x som 2D-array
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(x_values)
    
    # 2) Få klyngesentrene og sorter dem (fra venstre til høyre)
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())

    # 3) Gruppér punktene basert på hvilken klynge de tilhører
    labels = kmeans.labels_
    grouped_points = {c: [] for c in cluster_centers}

    for (x, y), label in zip(points, labels):
        grouped_points[cluster_centers[label]].append((x, y))

    # 4) Fitter lineær regresjon for hver x-klynge
    for x_val, coords in grouped_points.items():
        if len(coords) < 2:
            continue  # For få punkter til å lage linje

        X = np.array([c[1] for c in coords]).reshape(-1, 1)
        Y = np.array([c[0] for c in coords])

        model = LinearRegression()
        model.fit(X, Y)

        y_min, y_max = np.min(X), np.max(X)
        x_min = model.predict([[y_min]])[0]
        x_max = model.predict([[y_max]])[0]

        lines.append(((x_min, y_min), (x_max, y_max)))

    return lines


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

def extract_sudoku_cells(hor_lines, ver_lines):
    """
    Trekker ut hver celle fra skjæringspunktene mellom horisontale og vertikale linjer.
    Returnerer en liste av celler, der hver celle er definert av fire hjørnepunkter.
    """
    cells = []
    for i in range(len(hor_lines) - 1):
        for j in range(len(ver_lines) - 1):
            x1, y1 = ver_lines[j][0]  # Venstre øvre hjørne
            x2, y2 = ver_lines[j + 1][0]  # Høyre øvre hjørne
            x3, y3 = hor_lines[i][0]  # Venstre nedre hjørne
            x4, y4 = hor_lines[i + 1][0]  # Høyre nedre hjørne

            cell = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            cells.append(cell)

    return cells

hor_lines_clu = lines_row_kmeans(points, n_clusters=10)
ver_lines_clu = lines_col_kmeans(points, n_clusters=10)

hor_lines_bins = lines_row_10bins(points, n_bins=10)
ver_lines_bins = lines_col_10bins(points, n_bins=10)

# Overlapp bins og KMeans cluster gridene
fig, ax = plt.subplots(figsize=(8, 8))

# Plot punktene
ax.scatter(points[:, 0], points[:, 1], s=10, color='blue', alpha=0.5)

# Tegn bins grid (rød)
for (x1, y1), (x2, y2) in hor_lines_bins:
    ax.plot([x1, x2], [y1, y2], color='red', linestyle='dotted')
for (x1, y1), (x2, y2) in ver_lines_bins:
    ax.plot([x1, x2], [y1, y2], color='red', linestyle='dotted')

# Tegn clusters grid (grønn)
for (x1, y1), (x2, y2) in hor_lines_clu:
    ax.plot([x1, x2], [y1, y2], color='green', linestyle='solid')
for (x1, y1), (x2, y2) in ver_lines_clu:
    ax.plot([x1, x2], [y1, y2], color='green', linestyle='solid')

ax.set_title("Overlapp av Bins (rød) og KMeans clusters (grønn)")
ax.set_aspect('equal', 'box')
ax.invert_yaxis()
plt.show()

# Ekstraher celler basert på clusters-metoden (den mer presise)
sudoku_cells1 = extract_sudoku_cells(hor_lines_clu, ver_lines_clu)

with open("sudoku_cells_kmeans.txt", "w") as f:
    for i, cell in enumerate(sudoku_cells1):
        f.write(f"Cell {i + 1}: {cell}\n")
        
        
    # Ekstraher celler basert på clusters-metoden (den mer presise)
sudoku_cells2 = extract_sudoku_cells(hor_lines_bins, ver_lines_bins)

with open("sudoku_cells_bins.txt", "w") as f:
    for i, cell in enumerate(sudoku_cells2):
        f.write(f"Cell {i + 1}: {cell}\n")





