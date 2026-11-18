/* PathSnap evaluator UI (laptop 2). Talks to the FastAPI server on laptop 1. */
(function () {
  "use strict";

  const STORE_KEY = "pathsnap.apiUrl";
  const $ = (id) => document.getElementById(id);
  const els = {
    form: $("connect-form"), url: $("api-url"), btn: $("connect-btn"), status: $("status"),
    trips: $("trips"), result: $("result-block"), resultTrip: $("result-trip"),
    hmm: $("m-hmm"), body: $("metrics-body"), overlay: $("overlay"), overlayText: $("overlay-text"),
  };

  let api = "";
  let busy = false;

  // ---------------------------------------------------------------- map
  const map = L.map("map", { preferCanvas: true, zoomControl: true }).setView([20, 0], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  const groups = {
    gps: L.layerGroup().addTo(map),
    clf: L.layerGroup().addTo(map),
    hmm: L.layerGroup().addTo(map),
    ends: L.layerGroup().addTo(map),
  };
  L.control.layers(null, {
    "Raw GPS points": groups.gps,
    "Classifier only": groups.clf,
    "Predicted route (HMM)": groups.hmm,
    "Start / end": groups.ends,
  }, { collapsed: false, position: "topright" }).addTo(map);

  // ---------------------------------------------------------------- helpers
  function readStoredUrl() {
    try { return localStorage.getItem(STORE_KEY) || ""; } catch (e) { return ""; }
  }
  function storeUrl(v) {
    try { localStorage.setItem(STORE_KEY, v); } catch (e) { /* private mode: ignore */ }
  }
  function setStatus(text, kind) {
    els.status.textContent = text;
    els.status.className = "status" + (kind ? " " + kind : "");
  }
  async function request(path, options, timeoutMs) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs || 10000);
    try {
      const res = await fetch(api + path, Object.assign({ signal: ctrl.signal }, options || {}));
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.detail || ("HTTP " + res.status));
      return body;
    } catch (err) {
      if (err.name === "AbortError") throw new Error("timed out - is the API running and reachable?");
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }
  const pct = (v) => (v == null ? "-" : (100 * v).toFixed(1) + "%");
  const num = (v, d) => (v == null ? "-" : Number(v).toFixed(d == null ? 3 : d));

  // ---------------------------------------------------------------- connect
  async function connect(evt) {
    if (evt) evt.preventDefault();
    api = els.url.value.trim().replace(/\/+$/, "");
    if (!api) return;
    storeUrl(api);
    els.btn.disabled = true;
    setStatus("Connecting to " + api + " ...");
    try {
      const health = await request("/health", null, 8000);
      const data = await request("/trips", null, 15000);
      renderTrips(data.trips || []);
      setStatus("Connected: " + health.model + ", " + data.trips.length + " trips available", "ok");
    } catch (err) {
      els.trips.innerHTML = "";
      const p = document.createElement("p");
      p.className = "hint";
      p.textContent = "Could not load trips.";
      els.trips.appendChild(p);
      setStatus("Connection failed: " + err.message, "err");
    } finally {
      els.btn.disabled = false;
    }
  }

  function renderTrips(trips) {
    els.trips.innerHTML = "";
    trips.forEach((t) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "trip";
      b.dataset.trip = t.trip_id;
      b.title = "Trip " + t.trip_id;
      const s = document.createElement("strong");
      s.textContent = String(parseInt(t.trip_id, 10));
      const n = document.createElement("span");
      n.textContent = t.n_points != null ? t.n_points + " pts" : "";
      b.append(s, n);
      b.addEventListener("click", () => predict(t.trip_id));
      els.trips.appendChild(b);
    });
  }

  // ---------------------------------------------------------------- predict
  async function predict(tripId) {
    if (busy) return;
    busy = true;
    document.querySelectorAll(".trip").forEach((b) => {
      b.disabled = true;
      b.classList.toggle("active", b.dataset.trip === tripId);
    });
    els.overlayText.textContent = "Matching trip " + parseInt(tripId, 10) + " on the server ...";
    els.overlay.hidden = false;
    const t0 = performance.now();
    try {
      const res = await request("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trip_id: tripId }),
      }, 120000);
      const roundTrip = (performance.now() - t0) / 1000;
      draw(res);
      showMetrics(res, roundTrip);
      setStatus("Trip " + tripId + " matched in " + roundTrip.toFixed(1) + " s", "ok");
    } catch (err) {
      setStatus("Prediction failed for trip " + tripId + ": " + err.message, "err");
    } finally {
      els.overlay.hidden = true;
      document.querySelectorAll(".trip").forEach((b) => { b.disabled = false; });
      busy = false;
    }
  }

  function draw(res) {
    Object.values(groups).forEach((g) => g.clearLayers());
    res.gps.forEach((p) => {
      L.circleMarker(p, { radius: 2.5, color: "#2166ac", weight: 0, fillOpacity: 0.75 }).addTo(groups.gps);
    });
    res.classifier_route.forEach((line) => {
      L.polyline(line, { color: "#7b3294", weight: 3, opacity: 0.85, dashArray: "6 6" }).addTo(groups.clf);
    });
    res.predicted_route.forEach((line) => {
      L.polyline(line, { color: "#1b262c", weight: 8, opacity: 0.55 }).addTo(groups.hmm);
      L.polyline(line, { color: "#f2a900", weight: 4.5, opacity: 1 }).addTo(groups.hmm);
    });
    if (res.gps.length) {
      const first = res.gps[0], last = res.gps[res.gps.length - 1];
      L.circleMarker(first, { radius: 7, color: "#fff", weight: 2, fillColor: "#2e7d5b", fillOpacity: 1 })
        .bindTooltip("start").addTo(groups.ends);
      L.circleMarker(last, { radius: 7, color: "#fff", weight: 2, fillColor: "#c0392b", fillOpacity: 1 })
        .bindTooltip("end").addTo(groups.ends);
      map.fitBounds(L.latLngBounds(res.gps), { padding: [30, 30] });
    }
  }

  function showMetrics(res, roundTrip) {
    const m = res.metrics, t = res.timings_s;
    els.result.hidden = false;
    els.resultTrip.textContent = "trip " + res.trip_id;
    els.hmm.textContent = pct(m.hmm_accuracy);
    const rows = [
      ["Point accuracy", null],
      ["Classifier + HMM", pct(m.hmm_accuracy)],
      ["Classifier only", pct(m.classifier_accuracy)],
      ["HMM, direction ignored", pct(m.hmm_accuracy_undirected)],
      ["Route", null],
      ["Route precision", pct(m.route_precision)],
      ["Route recall", pct(m.route_recall)],
      ["Route F1 (HMM / classifier only)", pct(m.route_f1) + " / " + pct(m.classifier_route_f1)],
      ["Continuity (HMM / classifier only)", num(m.hmm_continuity) + " / " + num(m.classifier_continuity)],
      ["Chain restarts (gaps)", String(m.n_breaks)],
      ["Run", null],
      ["GPS points", String(res.n_points)],
      ["Candidate arcs scored", String(res.n_candidates)],
      ["Server time", num(t.total, 2) + " s"],
      ["Round trip", num(roundTrip, 2) + " s"],
    ];
    els.body.innerHTML = "";
    rows.forEach(([label, value]) => {
      const tr = document.createElement("tr");
      const a = document.createElement("td");
      a.textContent = label;
      if (value === null) {
        tr.className = "section";
        a.colSpan = 2;
        tr.appendChild(a);
      } else {
        const b = document.createElement("td");
        b.textContent = value;
        tr.append(a, b);
      }
      els.body.appendChild(tr);
    });
  }

  // ---------------------------------------------------------------- start
  const params = new URLSearchParams(location.search);
  const fromQuery = params.get("api");
  const autoTrip = params.get("trip");
  els.url.value = fromQuery || readStoredUrl() || "http://localhost:8000";
  els.form.addEventListener("submit", connect);
  if (fromQuery || readStoredUrl()) {
    connect().then(() => { if (autoTrip && api) predict(autoTrip); });
  }
})();
