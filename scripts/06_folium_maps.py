"""Step 6 - Folium maps of predictions on student_test trips.

Layers: raw GPS, nearby road network, reference route, classifier-only route,
classifier + HMM route, start/end; legend and layer control.
Output: results/maps/pred_trip_<id>.html
"""
import sys

import folium
import numpy as np

from _paths import ARTIFACTS, MAPS, STUDENT_TEST_DIR
from pathsnap.hmm import reconstruct_route
from pathsnap.io import list_trips, load_trip
from pathsnap.pipeline import MapMatcher, score

MAX_GPS_MARKERS = 1500


def segments(net, arc_ids):
    ll = net.nodes_lonlat
    return [[[round(ll[f, 1], 6), round(ll[f, 0], 6)], [round(ll[t, 1], 6), round(ll[t, 0], 6)]]
            for f, t in zip(net.frm[arc_ids], net.to[arc_ids])]


def build_map(matcher, trip_id):
    trip = load_trip(STUDENT_TEST_DIR, trip_id, with_route=True)
    res = matcher.match(load_trip(STUDENT_TEST_DIR, trip_id, with_route=False))
    m_scores = score(res, trip.route)
    net = res.net
    lo, hi = res.xy.min(axis=0) - 120, res.xy.max(axis=0) + 120
    near = np.flatnonzero((np.minimum(net.ax, net.bx) <= hi[0]) & (np.maximum(net.ax, net.bx) >= lo[0])
                          & (np.minimum(net.ay, net.by) <= hi[1]) & (np.maximum(net.ay, net.by) >= lo[1]))

    lat, lon = res.lonlat[:, 1], res.lonlat[:, 0]
    fmap = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=15, tiles="OpenStreetMap")
    layers = [
        ("Road network (nearby arcs)", segments(net, near), dict(color="#7f7f7f", weight=1.5, opacity=0.6), True),
        ("Reference route", segments(net, trip.route), dict(color="#1a9850", weight=7, opacity=0.65), True),
        ("Classifier only", segments(net, np.asarray(reconstruct_route(net, res.clf_arcs), dtype=np.int64)),
         dict(color="#7b3294", weight=3, opacity=0.9, dash_array="6 6"), False),
        ("Classifier + HMM (predicted)", segments(net, np.asarray(res.route_arcs, dtype=np.int64)),
         dict(color="#f2a900", weight=4, opacity=1.0), True),
    ]
    for name, segs, style, show in layers:
        g = folium.FeatureGroup(name=name, show=show)
        if segs:
            folium.PolyLine(segs, **style).add_to(g)
        g.add_to(fmap)

    g_gps = folium.FeatureGroup(name=f"Raw GPS points ({len(lat)})")
    step = max(1, len(lat) // MAX_GPS_MARKERS)
    for la, lo_, ts in zip(lat[::step], lon[::step], res.t[::step]):
        folium.CircleMarker([la, lo_], radius=2, color="#2166ac", fill=True, fill_opacity=0.8, weight=0,
                            tooltip=f"t = {ts:.0f} s").add_to(g_gps)
    g_gps.add_to(fmap)
    folium.CircleMarker([lat[0], lon[0]], radius=7, color="white", weight=2, fill=True, fill_color="#2e7d5b",
                        fill_opacity=1, tooltip="start").add_to(fmap)
    folium.CircleMarker([lat[-1], lon[-1]], radius=7, color="white", weight=2, fill=True, fill_color="#c0392b",
                        fill_opacity=1, tooltip="end").add_to(fmap)

    legend = (
        '<div style="position:fixed;bottom:28px;left:28px;z-index:9999;background:white;padding:9px 12px;'
        'border:1px solid #999;border-radius:6px;font:13px Arial">'
        f'<b>Student-test trip {trip_id}</b><br>'
        '<span style="color:#2166ac">&#9679;</span> raw GPS<br>'
        '<span style="color:#1a9850">&#9644;</span> reference route<br>'
        '<span style="color:#f2a900">&#9644;</span> classifier + HMM<br>'
        '<span style="color:#7b3294">- - -</span> classifier only (hidden by default)<br>'
        '<span style="color:#7f7f7f">&#9644;</span> road network<br>'
        f'<hr style="margin:5px 0">HMM accuracy {m_scores["hmm_accuracy"] * 100:.1f}% '
        f'(classifier only {m_scores["classifier_accuracy"] * 100:.1f}%)<br>'
        f'Route F1 {m_scores["route_f1"] * 100:.1f}%</div>'
    )
    fmap.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    out = MAPS / f"pred_trip_{trip_id}.html"
    fmap.save(str(out))
    print(f"{trip_id}: HMM {m_scores['hmm_accuracy']:.4f} classifier {m_scores['classifier_accuracy']:.4f} -> {out.name}")


def main():
    matcher = MapMatcher.load(ARTIFACTS)
    wanted = sys.argv[1:] or list_trips(STUDENT_TEST_DIR)
    for tid in wanted:
        build_map(matcher, tid)


if __name__ == "__main__":
    main()
