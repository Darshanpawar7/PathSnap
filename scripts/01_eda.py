"""Step 1 - dataset inventory and exploratory analysis (train + student_test only).

Outputs
  results/eda_trip_inventory.csv   one row per trip
  results/eda_summary.txt          headline numbers and data-quality notes
  results/figures/eda_*.png        plots
  results/maps/eda_trip_<id>.html  Folium map: GPS points, road network, reference route
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from _paths import FIGURES, MAPS, RESULTS, STUDENT_TEST_DIR, TRAIN_DIR
from pathsnap.config import EXCLUDED_TRAIN_TRIPS
from pathsnap.geo import LocalProjection
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import align_route
from pathsnap.network import RoadNetwork


def trip_stats(split: str, split_dir, trip_id: str) -> dict:
    trip = load_trip(split_dir, trip_id)
    tr, nodes, arcs, route = trip.track, trip.nodes, trip.arcs, trip.route
    lon, lat, t = tr[:, 0], tr[:, 1], tr[:, 2]

    nan_rows = int(np.isnan(tr).any(axis=1).sum())
    bad_coord = int(((np.abs(lon) > 180) | (np.abs(lat) > 90)).sum())
    dt = np.diff(t)
    inside_bbox = float(
        np.mean(
            (lon >= nodes[:, 0].min()) & (lon <= nodes[:, 0].max())
            & (lat >= nodes[:, 1].min()) & (lat <= nodes[:, 1].max())
        )
    )

    proj = LocalProjection.around(tr[:, :2])
    xy = proj.forward(tr[:, :2])
    step = np.hypot(*np.diff(xy, axis=0).T)
    net = RoadNetwork(nodes, arcs, proj)

    n = net.n_nodes
    deg = net.out_degree + net.in_degree
    und = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    n_comp, comp = connected_components(und, directed=False)
    comp_sizes = np.bincount(comp)

    route_valid = bool(len(route) and route.max() < net.n_arcs and route.min() >= 0)
    if route_valid and len(route) > 1:
        linked = net.to[route[:-1]] == net.frm[route[1:]]
        route_connected = float(linked.mean())
    else:
        route_connected = np.nan
    ref_arc, ref_dist, _ = align_route(net, xy, route) if route_valid else (None, np.full(len(xy), np.nan), None)

    stats = dict(
        split=split, trip_id=trip_id,
        n_points=len(tr), nan_rows=nan_rows, bad_coords=bad_coord,
        duration_s=float(t[-1] - t[0]) if len(t) else 0.0,
        dt_eq_1s_frac=float(np.mean(dt == 1)) if len(dt) else np.nan,
        dt_max_s=float(dt.max()) if len(dt) else np.nan,
        n_gaps_gt5s=int((dt > 5).sum()), n_gaps_gt30s=int((dt > 30).sum()),
        non_increasing_t=int((dt <= 0).sum()),
        track_length_km=float(step.sum() / 1000),
        max_step_m=float(step.max()) if len(step) else np.nan,
        mean_speed_kmh=float(step.sum() / max(t[-1] - t[0], 1) * 3.6),
        extent_ew_km=float(np.ptp(xy[:, 0]) / 1000), extent_ns_km=float(np.ptp(xy[:, 1]) / 1000),
        center_lon=float(np.mean(lon)), center_lat=float(np.mean(lat)),
        track_inside_network_bbox=inside_bbox,
        n_nodes=n, n_arcs=net.n_arcs,
        isolated_nodes=int((deg == 0).sum()),
        n_components=int(n_comp), largest_component_frac=float(comp_sizes.max() / max(n, 1)),
        two_way_arc_frac=float(np.mean(net.reverse_arc >= 0)),
        arc_len_median_m=float(np.median(net.length)), arc_len_p95_m=float(np.percentile(net.length, 95)),
        zero_length_arcs=int((net.length < 1e-6).sum()),
        n_route_arcs=len(route), route_ids_valid=route_valid,
        route_unique_arcs=int(len(np.unique(route))),
        route_consecutive_connected=route_connected,
        gps_to_ref_median_m=float(np.nanmedian(ref_dist)),
        gps_to_ref_p95_m=float(np.nanpercentile(ref_dist, 95)),
        gps_to_ref_max_m=float(np.nanmax(ref_dist)),
    )
    return stats, dt, ref_dist


def folium_trip_map(split_dir, trip_id: str, out_path):
    import folium

    trip = load_trip(split_dir, trip_id)
    proj = LocalProjection.around(trip.track[:, :2])
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    xy = proj.forward(trip.track[:, :2])
    lo, hi = xy.min(axis=0) - 150, xy.max(axis=0) + 150
    near = (
        (np.minimum(net.ax, net.bx) <= hi[0]) & (np.maximum(net.ax, net.bx) >= lo[0])
        & (np.minimum(net.ay, net.by) <= hi[1]) & (np.maximum(net.ay, net.by) >= lo[1])
    )
    ll = trip.nodes
    center = trip.track[:, [1, 0]].mean(axis=0).tolist()
    m = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")

    def segments(arc_ids):
        # one multi-line per layer keeps the HTML small
        return [[[round(ll[f, 1], 6), round(ll[f, 0], 6)], [round(ll[to, 1], 6), round(ll[to, 0], 6)]]
                for f, to in trip.arcs[arc_ids]]

    g_net = folium.FeatureGroup(name=f"Road network (arcs near track: {near.sum()})")
    folium.PolyLine(segments(np.flatnonzero(near)), color="#888888", weight=1.5, opacity=0.7).add_to(g_net)
    g_net.add_to(m)

    g_ref = folium.FeatureGroup(name="Reference route (ground truth)")
    folium.PolyLine(segments(trip.route), color="#1a9850", weight=5, opacity=0.8).add_to(g_ref)
    g_ref.add_to(m)

    g_gps = folium.FeatureGroup(name=f"Raw GPS points ({len(trip.track)})")
    for lon, lat, ts in trip.track[:: max(1, len(trip.track) // 1500)]:
        folium.CircleMarker([lat, lon], radius=2, color="#2166ac", fill=True, fill_opacity=0.8,
                            tooltip=f"t={ts:.0f}s").add_to(g_gps)
    g_gps.add_to(m)

    s, e = trip.track[0], trip.track[-1]
    folium.Marker([s[1], s[0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([e[1], e[0]], tooltip="end", icon=folium.Icon(color="red")).add_to(m)
    legend = (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;padding:8px 10px;'
        'border:1px solid #999;font-size:13px">'
        f"<b>Trip {trip_id}</b><br>"
        '<span style="color:#2166ac">&#9679;</span> raw GPS<br>'
        '<span style="color:#1a9850">&#9644;</span> reference route<br>'
        '<span style="color:#888">&#9644;</span> road network</div>'
    )
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[trip.track[:, 1].min(), trip.track[:, 0].min()], [trip.track[:, 1].max(), trip.track[:, 0].max()]])
    m.save(str(out_path))


def matplotlib_trip(ax, split_dir, trip_id):
    trip = load_trip(split_dir, trip_id)
    proj = LocalProjection.around(trip.track[:, :2])
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    xy = proj.forward(trip.track[:, :2])
    lo, hi = xy.min(axis=0) - 100, xy.max(axis=0) + 100
    near = (np.minimum(net.ax, net.bx) <= hi[0]) & (np.maximum(net.ax, net.bx) >= lo[0]) & \
           (np.minimum(net.ay, net.by) <= hi[1]) & (np.maximum(net.ay, net.by) >= lo[1])
    for a in np.flatnonzero(near):
        ax.plot([net.ax[a], net.bx[a]], [net.ay[a], net.by[a]], color="#bbbbbb", lw=0.6)
    for a in trip.route:
        ax.plot([net.ax[a], net.bx[a]], [net.ay[a], net.by[a]], color="#1a9850", lw=2.5)
    ax.scatter(xy[:, 0], xy[:, 1], s=3, color="#2166ac", zorder=3)
    ax.set_title(f"Trip {trip_id}: GPS (blue), reference route (green), network (grey)", fontsize=9)
    ax.set_xlabel("x (m, local projection)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])


def main():
    jobs = [("train", TRAIN_DIR, t) for t in list_trips(TRAIN_DIR)] + \
           [("student_test", STUDENT_TEST_DIR, t) for t in list_trips(STUDENT_TEST_DIR)]
    out = Parallel(n_jobs=8, verbose=0)(delayed(trip_stats)(*j) for j in jobs)
    inv = pd.DataFrame([o[0] for o in out])
    dts = np.concatenate([o[1] for o in out])
    ref_d_all = np.concatenate([o[2] for o in out if o[0]["split"] == "train"])
    ref_d = np.concatenate([o[2] for o in out if o[0]["split"] == "train"
                            and o[0]["trip_id"] not in EXCLUDED_TRAIN_TRIPS])
    inv.to_csv(RESULTS / "eda_trip_inventory.csv", index=False)

    # ---------------------------------------------------------------- figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for split, c in (("train", "#4575b4"), ("student_test", "#d73027")):
        sub = inv[inv.split == split]
        axes[0].hist(sub.n_points, bins=25, alpha=0.7, color=c, label=split)
        axes[1].hist(sub.track_length_km, bins=25, alpha=0.7, color=c, label=split)
        axes[2].hist(np.log10(sub.n_arcs), bins=25, alpha=0.7, color=c, label=split)
    axes[0].set_title("GPS points per trip"); axes[1].set_title("Trajectory length (km)")
    axes[2].set_title("Road arcs per trip (log10)")
    for a in axes:
        a.legend()
    fig.tight_layout(); fig.savefig(FIGURES / "eda_trip_sizes.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    vals, cnt = np.unique(np.clip(np.round(dts), 0, 10), return_counts=True)
    axes[0].bar(["<1" if v < 1 else ("10+" if v >= 10 else str(int(v))) for v in vals], cnt, color="#4575b4")
    axes[0].set_yscale("log"); axes[0].set_title("Sampling interval between GPS points (s), all trips")
    axes[1].hist(np.clip(ref_d, 0, 60), bins=60, color="#1a9850")
    axes[1].set_title("GPS point -> aligned reference arc (m), train excl. trips 33, 54")
    axes[1].axvline(np.percentile(ref_d, 95), color="k", ls="--", label=f"p95 = {np.percentile(ref_d, 95):.1f} m")
    axes[1].axvline(np.percentile(ref_d, 99), color="r", ls="--", label=f"p99 = {np.percentile(ref_d, 99):.1f} m")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(FIGURES / "eda_sampling_and_noise.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(inv.center_lon, inv.center_lat, c=(inv.split == "train").map({True: "#4575b4", False: "#d73027"}), s=25)
    ax.set_title("Trip locations (centre of each track): blue = train, red = student_test")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    fig.tight_layout(); fig.savefig(FIGURES / "eda_trip_locations.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    matplotlib_trip(axes[0], TRAIN_DIR, "00000003")
    matplotlib_trip(axes[1], TRAIN_DIR, "00000046")
    fig.tight_layout(); fig.savefig(FIGURES / "eda_trajectory_examples.png", dpi=130); plt.close(fig)

    folium_trip_map(TRAIN_DIR, "00000003", MAPS / "eda_trip_00000003.html")

    # ---------------------------------------------------------------- summary
    tr = inv[inv.split == "train"]
    lines = [
        f"Trips: train={len(tr)}, student_test={(inv.split == 'student_test').sum()}",
        f"GPS points: train={tr.n_points.sum():,}, student_test={inv[inv.split == 'student_test'].n_points.sum():,}",
        f"Points per trip (all): min={inv.n_points.min()}, median={inv.n_points.median():.0f}, max={inv.n_points.max()}",
        f"NaN rows: {inv.nan_rows.sum()}, out-of-range coords: {inv.bad_coords.sum()}, "
        f"non-increasing timestamps: {inv.non_increasing_t.sum()}",
        f"Share of 1 s steps: {np.mean(dts == 1):.3f}; steps > 5 s: {(dts > 5).sum()}; > 30 s: {(dts > 30).sum()}; max gap {dts.max():.0f} s",
        f"Track inside network bbox (mean): {inv.track_inside_network_bbox.mean():.3f}  -> column order lon,lat confirmed",
        f"Arcs per trip: min={inv.n_arcs.min()}, median={inv.n_arcs.median():.0f}, max={inv.n_arcs.max()}",
        f"Two-way arc share (median): {inv.two_way_arc_frac.median():.2f}",
        f"Isolated nodes (total): {inv.isolated_nodes.sum()}, components per trip median={inv.n_components.median():.0f}, "
        f"largest component share median={inv.largest_component_frac.median():.3f}",
        f"Route ids valid in all trips: {bool(inv.route_ids_valid.all())}; consecutive route arcs connected (mean): "
        f"{inv.route_consecutive_connected.mean():.4f}",
        f"GPS->aligned reference arc distance (all train): median={np.median(ref_d_all):.1f} m, "
        f"p95={np.percentile(ref_d_all, 95):.1f} m, share > 50 m={(ref_d_all > 50).mean():.4f}",
        f"Same, excluding trips {list(EXCLUDED_TRAIN_TRIPS)}: median={np.median(ref_d):.1f} m, "
        f"p95={np.percentile(ref_d, 95):.1f} m, p99={np.percentile(ref_d, 99):.1f} m, max={ref_d.max():.1f} m, "
        f"share > 50 m={(ref_d > 50).mean():.4f}",
        "Trip 00000033: GPS ~7 m from roads but ~122 m from the reference route -> route does not describe the track",
        "Trip 00000054: GPS ~84 m from any road in its network -> network incomplete for this track",
        f"Largest time gap: trip {inv.loc[inv.dt_max_s.idxmax(), 'trip_id']} ({inv.dt_max_s.max():.0f} s)",
        f"Trip centres: lon {inv.center_lon.min():.2f}..{inv.center_lon.max():.2f}, lat {inv.center_lat.min():.2f}..{inv.center_lat.max():.2f}",
    ]
    anomalies = inv[(inv.dt_max_s > 30) | (inv.gps_to_ref_p95_m > 30) | (inv.route_consecutive_connected < 0.99)]
    lines.append("Trips flagged (gap > 30 s, p95 distance > 30 m, or route not fully connected): "
                 + json.dumps(anomalies.trip_id.tolist()))
    text = "\n".join(lines)
    (RESULTS / "eda_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
