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

def sort_lines(lines, orientation="horizontal"):

    index_line = []
    if orientation == "horizontal":
        for line in lines:
            index_line.append((line[0][1], line))

    else:
        for line in lines:
            index_line.append((line[0][1], line))

    index_line.sort()

    sorted_lines = []
    for i, (_, line) in enumerate(index_line):
        sorted_lines.append((i, line))  # Hver linje får en indeks etter sortering

    return sorted_lines

def filter_close_lines(primary_lines, secondary_lines, threshold=5):
    """
    Filtrer ut linjer fra `secondary_lines` hvis de er for nær `primary_lines`.
    Beholder kun linjer i `secondary_lines` som ikke overlapper med `primary_lines`.
    
    - `primary_lines` er de viktigste linjene (KMeans).
    - `secondary_lines` er de mindre viktige linjene (Bins) som kun beholdes om nødvendig.
    """
    filtered_secondary = []
    
    for sec_line in secondary_lines:
        x1_sec, y1_sec = sec_line[0]
        x2_sec, y2_sec = sec_line[1]

        keep = True  # Vi antar at vi skal beholde linjen

        for prim_line in primary_lines:
            x1_prim, y1_prim = prim_line[0]
            x2_prim, y2_prim = prim_line[1]

            # Beregn avstand mellom linjene (gjennomsnitt av start- og sluttpunkt)
            dist_start = np.hypot(x1_sec - x1_prim, y1_sec - y1_prim)
            dist_end = np.hypot(x2_sec - x2_prim, y2_sec - y2_prim)

            if dist_start < threshold and dist_end < threshold:
                keep = False  # Fjern linjen fordi den overlapper en KMeans-linje
                break

        if keep:
            filtered_secondary.append(sec_line)

    return filtered_secondary

# Bruk funksjonen til å fjerne overlappende bins-linjer
filtered_hor_lines_bins = filter_close_lines(hor_lines_clu, hor_lines_bins, threshold=5)
filtered_ver_lines_bins = filter_close_lines(ver_lines_clu, ver_lines_bins, threshold=5)


# Tegn det filtrerte gridet
fig, ax = plt.subplots(figsize=(8, 8))

# Vis originalbildet i bakgrunnen
ax.imshow(image, cmap='gray')

# Tegn filtrert clusters grid (grønn, heltrukken)
for (x1, y1), (x2, y2) in hor_lines_clu:
    ax.plot([x1, x2], [y1, y2], color='green', linestyle='solid', linewidth=1)
for (x1, y1), (x2, y2) in ver_lines_clu:
    ax.plot([x1, x2], [y1, y2], color='green', linestyle='solid', linewidth=1)

# Tegn de gjenværende bins-linjene (rød, stiplet) – kun der KMeans ikke har linjer
for (x1, y1), (x2, y2) in filtered_hor_lines_bins:
    ax.plot([x1, x2], [y1, y2], color='red', linestyle='dotted', linewidth=1)
for (x1, y1), (x2, y2) in filtered_ver_lines_bins:
    ax.plot([x1, x2], [y1, y2], color='red', linestyle='dotted', linewidth=1)

ax.set_title("Optimalisert Sudoku-grid (kun unike linjer)")
ax.set_aspect('equal', 'box')
ax.invert_yaxis()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def fuse_lines(kmeans_lines, bins_lines, threshold=5, orientation="horizontal"):
    """
    Fuserer linjer fra KMeans og Bins hvis de er nærmere enn en terskelverdi.
    - `threshold`: Hvor nært linjer må være for å fusjoneres.
    - `orientation`: "horizontal" eller "vertical"
    """
    all_lines = kmeans_lines + bins_lines

    # Sorter linjene etter posisjon
    sorted_lines = sorted(all_lines, key=lambda line: line[0][1] if orientation == "horizontal" else line[0][0])

    fused_lines = []
    i = 0

    while i < len(sorted_lines):
        p1 = sorted_lines[i]
        j = i + 1

        # Sjekk for overlapp med neste linje
        while j < len(sorted_lines):
            p2 = sorted_lines[j]
            distance = abs(p2[0][1] - p1[0][1]) if orientation == "horizontal" else abs(p2[0][0] - p1[0][0])

            if distance < threshold:
                # Fusjonér linjene ved å ta gjennomsnittet
                avg_y = (p1[0][1] + p2[0][1]) / 2 if orientation == "horizontal" else p1[0][1]
                avg_x = (p1[0][0] + p2[0][0]) / 2 if orientation == "vertical" else p1[0][0]
                fused_lines.append(((avg_x, avg_y), (p1[1][0], p1[1][1])))  # Behold endepunktet fra p1
                j += 1
            else:
                break  # Ingen flere overlappende linjer

        # Hvis ingen fusjonering, legg til linjen uendret
        if j == i + 1:
            fused_lines.append(p1)

        i = j  # Hopp til neste linje

    return fused_lines


fused_hor_lines = fuse_lines(hor_lines_clu, filtered_hor_lines_bins, threshold=15, orientation="horizontal")
fused_ver_lines = fuse_lines(ver_lines_clu, filtered_ver_lines_bins, threshold=10, orientation="vertical")


fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image, cmap='gray')

# Tegn de fusjonerte linjene (blå)
for (x1, y1), (x2, y2) in fused_hor_lines:
    ax.plot([x1, x2], [y1, y2], color='blue', linestyle='solid', linewidth=1)
for (x1, y1), (x2, y2) in fused_ver_lines:
    ax.plot([x1, x2], [y1, y2], color='blue', linestyle='solid', linewidth=1)

ax.set_title("Fusjonert Sudoku-grid med optimaliserte linjer")
ax.set_aspect('equal', 'box')
plt.show()


















