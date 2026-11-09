"""Prepara el mapa de cantones de Costa Rica que usa la pestaña "Mapa cantonal" de la app.

Fuente: geoBoundaries, CRI ADM2 (cantones), versión simplificada. Licencia: dominio público (CC0).
https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/CRI/ADM2/geoBoundaries-CRI-ADM2_simplified.geojson

Uso:
    python src/preparar_mapa_cantones.py geoBoundaries-CRI-ADM2_simplified.geojson

Qué hace:
1. Simplifica las fronteras (algoritmo de Douglas-Peucker) y redondea las coordenadas,
   para que el archivo pese poco y el mapa cargue rápido.
2. Pone los nombres con tilde y corrige "Palmeras", que en el original es Palmares.
3. Ordena los polígonos en sentido horario, como los necesita Plotly.
4. Guarda data/raw/cantones_costa_rica.geojson.
"""
import json
import os
import sys
import unicodedata

TOLERANCIA = 0.001      # grados (~110 m): no se nota a la escala del país
DECIMALES = 4           # ~11 m

NOMBRES = {             # nombre del archivo original -> nombre correcto
    "Aserri": "Aserrí", "Belen": "Belén", "Canas": "Cañas", "Escazu": "Escazú",
    "Guacimo": "Guácimo", "Jimenez": "Jiménez", "La Union": "La Unión",
    "Leon Cortes Castro": "León Cortés Castro", "Limon": "Limón", "Palmeras": "Palmares",
    "Paraiso": "Paraíso", "Perez Zeledon": "Pérez Zeledón", "Poas": "Poás", "Pococi": "Pococí",
    "Puerto Jimenez": "Puerto Jiménez", "Rio Cuarto": "Río Cuarto", "San Jose": "San José",
    "San Ramon": "San Ramón", "Santa Barbara": "Santa Bárbara", "Sarapiqui": "Sarapiquí",
    "Sarchi": "Sarchí", "Tarrazu": "Tarrazú", "Tibas": "Tibás", "Tilaran": "Tilarán",
    "Vazquez de Coronado": "Vázquez de Coronado",
}


def clave(nombre):
    """Nombre en minúscula y sin tildes, para unir el mapa con los datos."""
    nombre = unicodedata.normalize("NFD", nombre.lower().strip())
    return "".join(c for c in nombre if unicodedata.category(c) != "Mn")


def simplificar(puntos, tolerancia):
    """Douglas-Peucker: quita los puntos que casi no cambian la forma."""
    if len(puntos) < 3:
        return puntos
    conservar = [False] * len(puntos)
    conservar[0] = conservar[-1] = True
    pendientes = [(0, len(puntos) - 1)]
    while pendientes:
        ini, fin = pendientes.pop()
        (x1, y1), (x2, y2) = puntos[ini], puntos[fin]
        dx, dy = x2 - x1, y2 - y1
        largo = (dx * dx + dy * dy) ** 0.5
        peor, indice = 0.0, None
        for i in range(ini + 1, fin):
            x, y = puntos[i]
            if largo == 0:
                d = ((x - x1) ** 2 + (y - y1) ** 2) ** 0.5
            else:
                d = abs(dy * x - dx * y + x2 * y1 - y2 * x1) / largo
            if d > peor:
                peor, indice = d, i
        if indice is not None and peor > tolerancia:
            conservar[indice] = True
            pendientes += [(ini, indice), (indice, fin)]
    return [p for p, c in zip(puntos, conservar) if c]


def area(anillo):
    """Área con signo (fórmula del zapatero): negativa si va en sentido horario."""
    return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(anillo, anillo[1:])) / 2


def limpiar_poligono(poligono):
    anillos = []
    for k, anillo in enumerate(poligono):
        anillo = simplificar(anillo, TOLERANCIA)
        anillo = [[round(x, DECIMALES), round(y, DECIMALES)] for x, y in anillo]
        anillo = [p for i, p in enumerate(anillo) if i == 0 or p != anillo[i - 1]]
        if len(anillo) < 4:
            if k == 0:
                return None          # la isla es demasiado pequeña: se descarta
            continue
        exterior = k == 0
        if (area(anillo) < 0) != exterior:   # exterior en sentido horario, huecos al revés
            anillo = anillo[::-1]
        anillos.append(anillo)
    return anillos


def main(origen, destino):
    with open(origen, encoding="utf-8") as f:
        mapa = json.load(f)

    salida = []
    for f in mapa["features"]:
        nombre = NOMBRES.get(f["properties"]["shapeName"], f["properties"]["shapeName"])
        geometria = f["geometry"]
        poligonos = geometria["coordinates"] if geometria["type"] == "MultiPolygon" else [geometria["coordinates"]]
        poligonos = [p for p in (limpiar_poligono(p) for p in poligonos) if p]
        if not poligonos:
            raise ValueError(f"{nombre} se quedó sin forma: baje la tolerancia")
        salida.append({
            "type": "Feature",
            "id": clave(nombre),
            "properties": {"canton": nombre, "clave": clave(nombre)},
            "geometry": {"type": "MultiPolygon", "coordinates": poligonos},
        })

    with open(destino, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": salida}, f, ensure_ascii=False, separators=(",", ":"))
    print(f"{len(salida)} cantones guardados en {destino} ({os.path.getsize(destino) / 1024:.0f} KB)")


if __name__ == "__main__":
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1], os.path.join(raiz, "data", "raw", "cantones_costa_rica.geojson"))
