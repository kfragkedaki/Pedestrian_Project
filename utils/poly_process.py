from shapely.geometry import Polygon, LineString, Point
from shapely.ops import linemerge, unary_union, polygonize
from typing import List


def _cut_poly_by_line(polygon: Polygon, line: LineString):
    merged = linemerge([polygon.boundary, line])
    borders = unary_union(merged)
    polygons = polygonize(borders)
    return list(polygons)

def crosswalk_poly_for_label(map) -> List[Polygon]:
    p1 = Polygon(map.crosswalk_poly.boundary.geoms[0])
    p2 = Polygon(map.crosswalk_poly.boundary.geoms[1])
    l1 = LineString([(5,43),(5,25),(6.8,8),(6.8,-3)])
    l2 = LineString([(20,34),(20,25),(22,7),(22,-3)])
    l3 = LineString([(33,7),(-3,7)])
    l4 = LineString([(33,26),(-3,26)])
    p3s = _cut_poly_by_line(p1, l1)
    p4s = []
    _crosswalks = [0]*4
    _points = [Point((0,16)), Point((14,3)), Point((27,16)), Point((14,30))]
    for p3 in p3s:
        p4s = [*p4s, *_cut_poly_by_line(p3,l2)]
        p5s = []
        for p4 in p4s:
            p5s = [*p5s, *_cut_poly_by_line(p4,l3)]
            p6s = []
            for p5 in p5s:
                p6s = [*p6s, *_cut_poly_by_line(p5,l4)]
    for p in p6s:
        _c = p.difference(p2)
        for i, _point in enumerate(_points):
            if _point.within(_c):
                _crosswalks[i] = _c
    return _crosswalks