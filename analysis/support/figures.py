"""Figure previews from saved numerical data. Style is versioned separately."""
import os
from pathlib import Path
import tempfile

from common import file_hash, utc_now, write_new
from analysis.support.source_record import presentation_code_record

# Edit this configuration for appearance, then increment plot_version.
# Scientific quantities and intervals come from the numerical result bundle.
FIGURE_SETTINGS = {
    "plot_version": "1.51", "style_version": "1.1", "font": "Arial",
    "width_inches": 6.5, "category_limit": 5,
    "colors": {"teal": "#5F828E", "dark_teal": "#2E5661", "orange": "#E6A052"},
    # Distinct model colors requested for the combined confidence figure.
    "confidence_colors": [
        "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6F3C9F",
        "#A6761D", "#E31A1C", "#008B8B", "#556B2F", "#377EB8",
        "#B21869", "#222222", "#B8860B",
    ],
    "resource_marker_area": 75,
    "resource_label_offsets": {},  # Figure name -> configuration -> (dx, dy) in points.
    # Opt in from the working notebook; historical report callers retain their layout.
    "report_layout": "original",
    "revision_sizes": {"combined": (210/25.4, 297/25.4), "combined_3d": (10, 14), "probability": (7.2, 7.8),
                       "probability_main": (210/25.4, 297/25.4), "probability_multiples": (7.2, 11.5),
                       "repeatability": (7.2, 7.6)},
    # Physical points on the A4 canvas; the old 18-inch figure shrank to ~7 pt.
    "figure1_fontsize": 9.5,
    "figure1_label_fontsize": 9,
    "figure1_schematic_fontsize": 9.5,
    "resource_full_frontier_names": False,  # IDs map to the full names in panel b.
    "main_fontsize": 11,
    "main_tick_fontsize": 10,
    "main_panel_fontsize": 13,
    "revision_fontsize": 9,
    "revision_label_fontsize": 8,
    "revision_legend_columns": 2,
    "revision_auroc_in_legend": False,
    "revision_3d_view": (35, -135),
    "rc_params": {
        "axes.formatter.use_locale": False, "font.size": 11, "figure.dpi": 180, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#404040", "axes.linewidth": .6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "grid.color": "#DDDDDD",
        "grid.linewidth": .6, "lines.linewidth": 1.2, "patch.edgecolor": "#404040",
        "patch.linewidth": .6, "legend.fontsize": 10, "legend.fancybox": False,
        "legend.framealpha": 1., "legend.edgecolor": "#404040", "xtick.direction": "out",
        "ytick.direction": "out", "xtick.major.width": .6, "ytick.major.width": .6,
        "svg.fonttype": "none", "svg.hashsalt": "jev-analysis-figures-v1",
    },
}
# Historical callers retain their default font and style names.
STYLE_VERSION, FONT = FIGURE_SETTINGS["style_version"], FIGURE_SETTINGS["font"]


def configure_style(font=None, *, settings=None):
    settings = FIGURE_SETTINGS if settings is None else settings
    font = settings["font"] if font is None else font
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="jev-analysis-mpl-"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(os.environ["MPLCONFIGDIR"])/"font-cache"))
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import findfont
    try:
        font_path = Path(findfont(font, fallback_to_default=False))
    except ValueError as exc:
        raise ValueError("Selected font unavailable: choose an explicit sans-serif font in FIGURE_SETTINGS") from exc
    plt.rcdefaults()
    plt.rcParams.update({**settings["rc_params"], "font.family": font, "axes.formatter.use_locale": False})
    return {"style_guide_version": settings["style_version"], "font": font,
            "font_file": font_path.name, "font_sha256": file_hash(font_path)}


def presentation_plot_data(summary, plot_data, *, confidence_display="auroc", revision=False):
    """Copy saved plots; show AUROC points or ROC rates from saved threshold counts."""
    from copy import deepcopy
    from math import isclose
    from analysis.support.notebook import report_figure_data
    if confidence_display not in ("auroc", "roc"):
        raise ValueError("Choose confidence_display='auroc' or 'roc'")
    data = {k: v for k, v in report_figure_data(plot_data).items() if not k.endswith("repeatability_sankey")}
    rows = []
    labels = {"J01": "JEV", "D01": "Luna Decisions"}
    if summary.get('configuration_namespace'):
        from benchmarking.configuration_ids import canonical_configuration, CURRENT_NAMESPACE
        labels = {cid: canonical_configuration(CURRENT_NAMESPACE, cid)['display_name']
                  for cid in summary.get('decision_models', {})}
    for cid, model in summary.get("decision_models", {}).items():
        confidence = model["confidence"]
        if confidence["auroc"] is None:
            continue
        row = {"configuration": cid, "model": labels.get(cid, cid),
               "eligible_n": confidence["eligible_n"], "auroc": confidence["auroc"],
               "interval95": deepcopy(confidence["auroc_bootstrap"]["interval95"])}
        if confidence_display == "roc":
            correct_n = confidence["score_support"]["correct_n"]
            incorrect_n = confidence["score_support"]["incorrect_n"]
            if correct_n <= 0 or incorrect_n <= 0 or correct_n+incorrect_n != row["eligible_n"]:
                raise ValueError(f"{cid}: ROC requires both correctness classes and matching eligible counts")
            points = [{"threshold_exact": None, "retained": 0, "errors": 0,
                       "false_positive_rate": 0., "true_positive_rate": 0.}]
            previous_correct, previous_errors = 0, 0
            for threshold in confidence["risk_coverage"]:
                retained, errors = threshold["retained"], threshold["errors"]
                correct = retained-errors
                if not (0 <= previous_correct <= correct <= correct_n and
                        0 <= previous_errors <= errors <= incorrect_n):
                    raise ValueError(f"{cid}: saved threshold counts are inconsistent or unordered")
                points.append({"threshold_exact": threshold["threshold_exact"],
                               "retained": retained, "errors": errors,
                               "false_positive_rate": errors/incorrect_n,
                               "true_positive_rate": correct/correct_n})
                previous_correct, previous_errors = correct, errors
            if previous_correct != correct_n or previous_errors != incorrect_n:
                raise ValueError(f"{cid}: saved thresholds do not cover all eligible answers")
            area = sum((b["false_positive_rate"]-a["false_positive_rate"])*
                       (a["true_positive_rate"]+b["true_positive_rate"])/2
                       for a, b in zip(points, points[1:]))
            if not isclose(area, row["auroc"], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{cid}: plotted ROC area differs from the saved AUROC")
            row.update({"correct_n": correct_n, "incorrect_n": incorrect_n,
                        "tie_rule": confidence["tie_rule"], "points": points})
        rows.append(row)
    if rows:
        data["confidence_"+confidence_display] = rows
    curves = [{"configuration": cid, "model": labels.get(cid, cid),
               "eligible_n": model["confidence"]["eligible_n"],
               "curve": deepcopy(model["confidence"]["risk_coverage"])}
              for cid, model in summary.get("decision_models", {}).items()
              if model["confidence"].get("risk_coverage")]
    if curves:
        data["decision_risk_coverage"] = curves
    # Derive this presentation from the saved summary, including older bundles
    # whose frozen figure_data has no cutoff figure. Do not gate it on AUROC.
    cutoff_curves = [{"configuration": cid, "model": labels.get(cid, cid),
                      "eligible_n": model["confidence"]["eligible_n"],
                      "curve": deepcopy(model["confidence"]["risk_coverage"])}
                     for cid, model in summary.get("decision_models", {}).items()]
    if cutoff_curves:
        data["confidence_cutoff"] = cutoff_curves
    confidence_names = ("confidence_roc", "decision_risk_coverage", "confidence_cutoff")
    if all(name in data for name in confidence_names):
        data["confidence_panel"] = {name: deepcopy(data[name]) for name in confidence_names}
    resource_names = ("accuracy_cost", "accuracy_latency")
    if all(name in data for name in resource_names):
        data["resource_panel"] = {name: deepcopy(data[name]) for name in resource_names}
    repeatability_names = ("repeatability_confidence", "repeatability_vector")
    if all(name in data for name in repeatability_names):
        data["repeatability_panel"] = {name: deepcopy(data[name]) for name in repeatability_names}
    if revision:
        data.update(_revision_plot_data(summary, data))
    return data


def _revision_plot_data(summary, data):
    """Assemble presentation candidates from saved estimates; never refit statistics."""
    from copy import deepcopy
    from math import isclose, isfinite
    expected = {f"G{i:02}" for i in range(1, 16)} | {f"D{i:02}" for i in range(1, 14)}
    decisions = {f"D{i:02}" for i in range(1, 14)}
    if set(summary['primary']) != expected or set(summary['decision_models']) != decisions:
        raise ValueError('Revision candidates require all 28 configurations and 13 decision models')
    for cid, row in summary['primary'].items():
        if not row['complete'] or row['planned_n'] != 1273 or row['wilson95'] is None:
            raise ValueError(f'{cid}: complete saved accuracy and Wilson interval required')
    frontier = summary.get('frontiers', {}).get('analysis_usd', {}).get('three_dimensional')
    if frontier is None or frontier['excluded']:
        raise ValueError('Saved approved-cost 3D frontier missing or incomplete; do not recompute in render mode')
    points3d = []
    for name in ('accuracy', 'accuracy_cost', 'accuracy_latency'):
        rows = data[name] if name == 'accuracy' else data[name]['points']
        if len(rows) != 28 or {r['configuration'] for r in rows} != expected:
            raise ValueError(f'{name}: saved plot must cover all 28 configurations exactly once')
    for cid, row in summary['primary'].items():
        op = row['operations']
        cost = op.get('analysis_usd', {}).get('mean_per_attempt')
        duration = op.get('analysis_latency', {}).get('median')
        if any(v is None or not isfinite(v) or v <= 0 for v in (cost, duration)):
            raise ValueError(f'{cid}: positive saved cost/duration required for logarithmic resource panels')
        # Check the two saved projections against the primary summary before joining them.
        for name, value in (('accuracy_cost', cost), ('accuracy_latency', duration)):
            point = next(p for p in data[name]['points'] if p['configuration'] == cid)
            if not isclose(point['x'], value, rel_tol=1e-12, abs_tol=0) or point['accuracy'] != row['accuracy']:
                raise ValueError(f'{cid}: {name} differs from saved primary results')
        points3d.append({'configuration': cid, 'accuracy': row['accuracy'],
                         'cost_usd_per_1000': 1000*cost, 'duration_seconds': duration,
                         'nondominated': cid in frontier['nondominated']})
    schematic = {'questions': 1273, 'decision_models': 13, 'generative_configurations': 15,
                 'primary_runs_per_configuration': 1, 'additional_decision_runs': 2}
    combined = {'schematic': schematic, 'accuracy': deepcopy(data['accuracy']),
                'cost_basis': 'Selected final attempts; observed plus approved within-run mean imputation'}
    candidate_a = {**deepcopy(combined), 'accuracy_cost': deepcopy(data['accuracy_cost']),
                   'accuracy_latency': deepcopy(data['accuracy_latency'])}
    for point in candidate_a['accuracy_cost']['points']:
        point['x'] *= 1000
    candidate_a['accuracy_cost']['display_unit'] = 'USD per 1,000 questions'
    resource_3d = {'accuracy_cost_duration': points3d}
    probability = deepcopy(data['confidence_panel'])
    for name, rows in probability.items():
        if len(rows) != 13 or {r['configuration'] for r in rows} != decisions:
            raise ValueError(f'{name}: all 13 decision models must be retained')
    calibration = []
    for cid, model in summary['decision_models'].items():
        c = model['confidence']
        bins = c.get('reliability')
        if bins is None or len(bins) != 10 or sum(b['n'] for b in bins) != c['eligible_n']:
            raise ValueError(f'{cid}: ten saved calibration bins and eligible denominator required')
        if any(b['n'] and (b['accuracy_wilson95'] is None or b['mean_probability'] is None
                          or b['observed_accuracy'] is None) for b in bins):
            raise ValueError(f'{cid}: nonempty calibration bin is missing its estimate or uncertainty')
        calibration.append({'configuration': cid, 'eligible_n': c['eligible_n'], 'bins': deepcopy(bins)})
    probability['calibration'] = calibration
    # Display selection is bound to the saved joint frontier, never a projection
    # or a ranking of the probability outcomes being displayed.
    frontier_order = ['D01', 'D02', 'D05', 'D08', 'D09', 'D11']
    if decisions.intersection(frontier['nondominated']) != set(frontier_order):
        raise ValueError('Saved joint frontier does not match the six requested decision models')
    def paired_payload(keys):
        return {'model_order': frontier_order.copy(),
                'selection': 'analysis_usd.three_dimensional.nondominated',
                **{key: [deepcopy(next(r for r in probability[key] if r['configuration'] == cid))
                         for cid in frontier_order] for key in keys}}
    repeatability = deepcopy(data['repeatability_panel'])
    expected_pairs = {(cid, a, b) for cid in decisions for a, b in ((1, 2), (1, 3), (2, 3))}
    for name, rows in repeatability.items():
        actual = {(r['configuration'], r['earlier_run'], r['later_run']) for r in rows}
        if len(rows) != 39 or actual != expected_pairs:
            raise ValueError(f'{name}: all 13 models and three run pairs must be retained')
    return {'combined_candidate_a': candidate_a, 'resource_3d': resource_3d,
            'probability_roc_calibration': paired_payload(('confidence_roc', 'calibration')),
            'probability_errors': paired_payload(('decision_risk_coverage', 'confidence_cutoff')),
            'probability_coverage': probability['decision_risk_coverage'],
            'probability_calibration': calibration,
            'probability_roc_supplement': probability['confidence_roc'],
            'probability_cutoff_supplement': probability['confidence_cutoff'],
            'repeatability_supplement': repeatability}


def _confidence_cutoff_steps(row):
    """Map saved whole-tie rows to left-continuous cutoff coordinates only."""
    from fractions import Fraction
    from math import nan
    curve = list(reversed(row["curve"]))  # Saved thresholds descend.
    if not curve:
        if row["eligible_n"] != 0:
            raise ValueError("Eligible confidence answers require saved threshold rows")
        return [0., 1.], [nan, nan], [0., 0.]
    thresholds = [Fraction(point["threshold_exact"]) for point in curve]
    if (not all(0 <= value <= 1 for value in thresholds)
            or any(a >= b for a, b in zip(thresholds, thresholds[1:]))
            or curve[0]["retained"] != row["eligible_n"]):
        raise ValueError("Saved confidence thresholds must be ordered, unique and cover eligible answers")
    # Use exact boundaries for ordering and endpoint decisions; convert only
    # at the Matplotlib boundary. Never round, bin or combine score ties.
    risk = [100*point["risk"] if point["retained"] else nan for point in curve]
    coverage = [100*point["coverage"] for point in curve]
    if thresholds[0] > 0:
        thresholds.insert(0, Fraction(0))
        risk.insert(0, risk[0]); coverage.insert(0, coverage[0])
    if thresholds[-1] < 1:
        thresholds.append(Fraction(1))
        risk.append(nan); coverage.append(0.)
    x = [float(value) for value in thresholds]
    if any(a >= b for a, b in zip(x, x[1:])):
        raise ValueError("Exact confidence boundaries are too close for distinct plotting coordinates")
    return x, risk, coverage


def _revision_resource_labels(ax, points, settings, name, model_names=None):
    """Prioritize full frontier names; show other IDs only if their bounds fit."""
    from matplotlib.transforms import Bbox
    from matplotlib.text import Text
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    radius = 5*fig.dpi/72
    occupied = []
    for p in points:
        x, y = ax.transData.transform((p['x'], 100*p['accuracy']))
        occupied.append(Bbox.from_extents(x-radius, y-radius, x+radius, y+radius))
    offsets = sorted(((x, y) for x in (-60, -48, -36, -24, -18, -12, -6, 0, 6, 12, 18, 24, 36, 48, 60)
                      for y in (-48, -36, -24, -18, -12, -6, 0, 6, 12, 18, 24, 36, 48)
                      if x*x+y*y >= 12**2),
                     key=lambda p: (p[0]**2+p[1]**2, -p[1], -p[0]))
    overrides = settings.get('resource_label_offsets', {}).get(name, {})
    warnings = []
    def area(a, b):
        overlap = Bbox.intersection(a, b)
        return 0 if overlap is None else overlap.width*overlap.height
    for p in sorted(points, key=lambda p: (not p['nondominated'], -p['accuracy'], p['configuration'])):
        cid = p['configuration']
        from textwrap import fill
        label_text = fill((model_names or {}).get(cid, cid), width=24, break_long_words=False,
                          break_on_hyphens=False) if (p['nondominated'] and settings.get('resource_full_frontier_names', True)) else cid
        label = ax.annotate(label_text, (p['x'], 100*p['accuracy']), xytext=(6, 6),
                            textcoords='offset points', fontsize=settings['figure1_label_fontsize'],
                            arrowprops={'arrowstyle': '-', 'lw': .7, 'color': '#777777',
                                        'shrinkA': 2, 'shrinkB': 7}, zorder=5)
        best = None
        for dx, dy in ([overrides[cid]] if cid in overrides else offsets):
            label.set_position((dx, dy))
            label.set_ha('left' if dx > 0 else 'right' if dx < 0 else 'center')
            label.set_va('bottom' if dy > 0 else 'top' if dy < 0 else 'center')
            label.update_positions(renderer)
            box = Text.get_window_extent(label, renderer).padded(fig.dpi/72)
            score = (max(0, box.width*box.height-area(box, ax.bbox)),
                     sum(area(box, other) for other in occupied), dx*dx+dy*dy)
            if best is None or score < best[0]:
                best = (score, (dx, dy), label.get_ha(), label.get_va(), box)
        score, position, ha, va, box = best
        if not p['nondominated'] and (score[0] > 1 or score[1] > 1):
            label.remove()
            continue
        label.set_position(position); label.set_ha(ha); label.set_va(va)
        occupied.append(box)
        if score[0] > 1 or score[1] > 1:
            warnings.append(f'{name}: review crowded label {cid}; adjust resource_label_offsets or dimensions')
    return warnings


def interactive_resource_view_v3(points, model_names, settings, *, standalone=False):
    """Supplementary v3 HTML using saved coordinates and frontier membership.

    The triangulated frontier is an interpolated visual guide, not a fitted model. Markers
    are constant-depth billboards tinted by nearer portions of the frontier surface.
    """
    import json
    from html import escape
    from math import log10
    from math import isfinite
    from IPython.display import HTML
    from matplotlib.tri import Triangulation

    if not points or len({p['configuration'] for p in points}) != len(points):
        raise ValueError('Unique saved model coordinates are required')
    for p in points:
        if any(not isfinite(p[k]) or p[k] <= 0
               for k in ('cost_usd_per_1000', 'duration_seconds')):
            raise ValueError('Log axes require positive finite saved cost and duration')
        if not isfinite(p['accuracy']) or not 0 <= p['accuracy'] <= 1:
            raise ValueError('Saved accuracy must be in [0, 1]')
        if not isinstance(p['nondominated'], bool):
            raise ValueError('Saved 3D frontier membership must be boolean')
    coordinates = [[log10(p['cost_usd_per_1000']), log10(p['duration_seconds']),
                    100*p['accuracy']] for p in points]
    vertices = sorted({tuple(xyz) for p, xyz in zip(points, coordinates) if p['nondominated']})
    if len(vertices) < 3:
        raise ValueError('Frontier surface requires at least three distinct points')
    triangulation = Triangulation([v[0] for v in vertices], [v[1] for v in vertices])
    payload = {
        'vertices': vertices, 'triangles': triangulation.triangles.tolist(),
        'points': [{**p, 'xyz': xyz, 'name': model_names.get(p['configuration'], p['configuration']),
                    'color': settings['colors']['orange' if p['configuration'].startswith('D') else 'teal']}
                   for p, xyz in zip(points, coordinates)],
        'view': [30, -135], 'version': 3,
    }
    # Escape HTML delimiters even inside JSON embedded in a script element.
    encoded = json.dumps(payload).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    document = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Supplementary resource figure · version 3</title><style>
body{margin:0;font:14px Arial,sans-serif;color:#333;background:white}
.controls{padding:10px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
canvas{width:100%;height:640px;touch-action:none;cursor:grab;display:block}
input{width:70px} #detail{min-height:35px;padding:8px}code{user-select:all}
</style></head><body>
<div class="controls">
<label>Elevation <input id="elev" type="number" min="-89" max="89" step="1"></label>
<label>Azimuth <input id="azim" type="number" step="1"></label>
<button id="reset">Reset view</button><button id="out" aria-label="Zoom out">−</button><button id="in" aria-label="Zoom in">+</button><button id="download">Download HTML</button><label><input id="mesh" type="checkbox" checked>Pareto surface</label>
<label><input id="labels" type="checkbox">IDs</label></div>
<div class="controls"><code id="angles"></code><span>Drag or use arrow keys to rotate; scroll or use +/− to zoom. Hover for values. Orange: decision models; teal: generative configurations. Diamonds: 3D Pareto frontier; circles: dominated.</span></div>
<canvas id="plot" tabindex="0" aria-label="Interactive accuracy, cost and duration plot"></canvas><div id="detail" role="status">Hover over a model for details.</div>
<script>
const data=__PAYLOAD__, canvas=document.getElementById('plot'), ctx=canvas.getContext('2d');
const el=document.getElementById('elev'), az=document.getElementById('azim');
const mesh=document.getElementById('mesh'), labels=document.getElementById('labels');
const detail=document.getElementById('detail');
let elevation=data.view[0], azimuth=data.view[1], zoom=1, drag=null, projected=[], panes=[];
const ranges=[0,1,2].map(k=>{const a=data.points.map(p=>p.xyz[k]);
  let lo=Math.min(...a),hi=Math.max(...a),pad=(hi-lo||1)*.08;return [lo-pad,hi+pad];});
const rad=Math.PI/180;
function project(v){
  const q=v.map((n,k)=>(n-ranges[k][0])/(ranges[k][1]-ranges[k][0])-.5);q[2]*=.8;
  const a=azimuth*rad,e=elevation*rad;
  const right=-Math.sin(a)*q[0]+Math.cos(a)*q[1];
  const up=-Math.sin(e)*Math.cos(a)*q[0]-Math.sin(e)*Math.sin(a)*q[1]+Math.cos(e)*q[2];
  const depth=Math.cos(e)*Math.cos(a)*q[0]+Math.cos(e)*Math.sin(a)*q[1]+Math.sin(e)*q[2];
  const w=canvas.clientWidth,h=canvas.clientHeight,s=zoom*Math.min(w*.64,h*.70);
  return [w/2+s*right,h/2-s*up,depth];
}
function line(a,b,color='#aaa'){const p=project(a),q=project(b);ctx.strokeStyle=color;
 ctx.beginPath();ctx.moveTo(p[0],p[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();}
function draw(){
 const dpr=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;
 canvas.width=w*dpr;canvas.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
 ctx.clearRect(0,0,w,h);ctx.font='12px Arial';ctx.lineWidth=.7;
 el.value=elevation;az.value=azimuth;
 document.getElementById('angles').textContent=`View ${elevation}°, ${azimuth}° · ${zoom.toFixed(2)}×`;
 const base=ranges.map(r=>r[0]);
 const names=['Cost (USD/1,000; log scale)','Duration (s; log scale)','Accuracy (%)'];
 for(let k=0;k<3;k++){
  const end=base.slice();end[k]=ranges[k][1];line(base,end,'#444');
  const p=project(end);ctx.fillStyle='#333';ctx.textAlign='center';ctx.fillText(names[k],p[0],p[1]-20);
  const ticks=k<2?Array.from({length:Math.max(0,Math.floor(ranges[k][1])-Math.ceil(ranges[k][0])+1)},(_,i)=>Math.ceil(ranges[k][0])+i)
   :Array.from({length:5},(_,i)=>ranges[k][0]+i*(ranges[k][1]-ranges[k][0])/4);
  for(const t of ticks){const v=base.slice();v[k]=t;const s=project(v);
   ctx.fillText(k<2?(10**t).toPrecision(2):t.toFixed(0),s[0]+12,s[1]+13);}
 }
 // Triangles are anchored exclusively to saved 3D nondominated estimates.
 panes=data.triangles.map(t=>t.map(i=>project(data.vertices[i])))
  .sort((a,b)=>a.reduce((s,p)=>s+p[2],0)-b.reduce((s,p)=>s+p[2],0));
 function polygon(f){ctx.beginPath();f.forEach((q,i)=>i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]));ctx.closePath();}
 if(mesh.checked)for(const f of panes){polygon(f);ctx.fillStyle='rgba(173,198,207,.28)';ctx.fill();ctx.strokeStyle='rgba(64,64,64,.55)';ctx.lineWidth=1;ctx.stroke();}
 // Clip each pane to the portion nearer than a marker's constant-depth billboard.
 function inFront(face,depth){
  const out=[];
  for(let i=0;i<face.length;i++){
   const a=face[i],b=face[(i+1)%face.length],ai=a[2]>depth,bi=b[2]>depth;
   if(ai)out.push(a);
   if(ai!==bi){const t=(depth-a[2])/(b[2]-a[2]);out.push(a.map((v,k)=>v+t*(b[k]-v)));}
  }return out;
 }
 projected=data.points.map(p=>({...p,screen:project(p.xyz)})).sort((a,b)=>a.screen[2]-b.screen[2]);
 for(const p of projected){const [x,y]=p.screen;ctx.beginPath();
  if(p.nondominated){ctx.moveTo(x,y-6);ctx.lineTo(x+6,y);ctx.lineTo(x,y+6);ctx.lineTo(x-6,y);ctx.closePath();}
  else ctx.arc(x,y,4,0,2*Math.PI);
  ctx.fillStyle=p.color;ctx.fill();ctx.strokeStyle='#404040';ctx.lineWidth=p.nondominated?1.7:.8;ctx.stroke();
  ctx.save();ctx.clip();
  if(mesh.checked)for(const f of panes){const front=inFront(f,p.screen[2]);
   if(front.length>=3){polygon(front);ctx.fillStyle='rgba(173,198,207,.28)';ctx.fill();}}
  ctx.restore();ctx.lineWidth=.7;
  if(labels.checked){ctx.fillStyle='#333';ctx.textAlign='left';ctx.fillText(p.configuration,x+7,y-7);}
 }
}
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY,elevation,azimuth];canvas.setPointerCapture(e.pointerId);};
canvas.onpointerup=()=>{drag=null;};canvas.onpointercancel=()=>{drag=null;};
canvas.onpointermove=e=>{
 if(drag){azimuth=Math.round(drag[3]-(e.clientX-drag[0])*.4);
  elevation=Math.max(-89,Math.min(89,Math.round(drag[2]+(e.clientY-drag[1])*.4)));draw();return;}
 const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
 const near=projected.map(p=>[Math.hypot(x-p.screen[0],y-p.screen[1]),p]).sort((a,b)=>a[0]-b[0])[0];
 if(near&&near[0]<18){const p=near[1];detail.textContent=`${p.configuration}: ${p.name} | Accuracy ${(100*p.accuracy).toFixed(2)}% | Cost $${p.cost_usd_per_1000.toPrecision(4)}/1,000 | Duration ${p.duration_seconds.toPrecision(4)} s${p.nondominated?' | 3D Pareto frontier':' | Dominated'}`;}else detail.textContent='Hover over a model for details.';
};
el.onchange=()=>{const v=Number(el.value);if(Number.isFinite(v))elevation=Math.max(-89,Math.min(89,v));draw();};
az.onchange=()=>{const v=Number(az.value);if(Number.isFinite(v))azimuth=v;draw();};
mesh.onchange=labels.onchange=draw;
function setZoom(z){zoom=Math.max(.55,Math.min(2.5,z));draw();}
canvas.addEventListener('wheel',e=>{e.preventDefault();setZoom(zoom*Math.exp(-e.deltaY*.001));},{passive:false});
document.getElementById('in').onclick=()=>setZoom(zoom*1.15);
document.getElementById('out').onclick=()=>setZoom(zoom/1.15);
canvas.onkeydown=e=>{
 if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','Home'].includes(e.key))return;
 e.preventDefault();
 if(e.key==='Home'){[elevation,azimuth]=data.view;zoom=1;}
 if(e.key==='ArrowLeft')azimuth-=5;if(e.key==='ArrowRight')azimuth+=5;
 if(e.key==='ArrowUp')elevation=Math.min(89,elevation+5);if(e.key==='ArrowDown')elevation=Math.max(-89,elevation-5);
 if(e.key==='+')zoom=Math.min(2.5,zoom*1.15);if(e.key==='-')zoom=Math.max(.55,zoom/1.15);draw();
};
canvas.onpointerleave=()=>{if(!drag)detail.textContent='Hover over a model for details.';};
document.getElementById('reset').onclick=()=>{[elevation,azimuth]=data.view;zoom=1;draw();};
document.getElementById('download').onclick=()=>{
 const url=URL.createObjectURL(new Blob(['<!doctype html>'+document.documentElement.outerHTML],{type:'text/html'}));
 const a=document.createElement('a');a.href=url;a.download='supplementary-resource-v3.html';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
// Choose a shallow front-quadrant camera minimizing projected point overlaps.
let best=Infinity,chosen=data.view;
for(const e of [20,25,30,35,40])for(const a of [-160,-150,-140,-130,-120,-110]){
 elevation=e;azimuth=a;const ps=data.points.map(p=>project(p.xyz));let score=0;
 for(let i=0;i<ps.length;i++)for(let j=0;j<i;j++){
  const d=Math.hypot(ps[i][0]-ps[j][0],ps[i][1]-ps[j][1]);score+=Math.max(0,22-d)**2;
 }
 score+=.002*((e-30)**2+(a+135)**2);
 if(score<best){best=score;chosen=[e,a];}
}
data.view=chosen;[elevation,azimuth]=chosen;
new ResizeObserver(draw).observe(canvas);draw();
</script></body></html>'''.replace('__PAYLOAD__', encoded)
    if standalone:
        return document
    return HTML('<iframe title="Supplementary resource figure version 3" sandbox="allow-scripts allow-downloads" '
                'style="width:100%;height:900px;border:1px solid #ddd" srcdoc="'
                + escape(document, quote=True) + '"></iframe>')


def _build_revision_figures(summary, data, settings, model_names, figure_names):
    """Review figures at physical publication size, using saved quantities."""
    from copy import deepcopy
    from math import log10, floor, ceil, isfinite, nan
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    settings = deepcopy(settings)
    settings['rc_params'].update({'font.size': settings['revision_fontsize'],
                                  'legend.fontsize': settings['revision_label_fontsize'],
                                  'lines.linewidth': .9})
    metadata = configure_style(settings=settings)
    wanted = set(current_report_figures(summary, revision=True) if figure_names is None else figure_names)
    unknown = wanted - set(current_report_figures(summary, revision=True))
    if unknown:
        raise ValueError(f'Unknown revision figures: {sorted(unknown)}')
    figures, evidence, warnings = {}, {}, []
    model_names = model_names or {}
    from benchmarking.configuration_ids import canonical_configuration, CURRENT_NAMESPACE
    names = {cid: model_names.get(cid, canonical_configuration(CURRENT_NAMESPACE, cid)['display_name'])
             for cid in summary['primary']}
    gray = settings['rc_params']['axes.edgecolor']
    category_color = lambda cid: settings['colors']['orange' if cid.startswith('D') else 'teal']
    ids = sorted(summary['decision_models'])
    patterns = ('-', '--', '-.', ':')
    markers = ('o', 's', '^', 'D', 'v', 'P', 'X')
    styles = {cid: {'color': settings['confidence_colors'][i % len(settings['confidence_colors'])],
                    'linestyle': patterns[i % len(patterns)], 'linewidth': .9}
              for i, cid in enumerate(ids)}
    def letter(ax, value):
        # Journal panel lettering is an approved exception to the house default.
        ax.text2D(-.08, 1.02, value, transform=ax.transAxes, weight='bold') if hasattr(ax, 'text2D') else ax.text(
            -.08, 1.02, value, transform=ax.transAxes, weight='bold', va='bottom')
    def capture(name, fig):
        figures[name] = fig
        evidence[name] = {'size_inches': list(fig.get_size_inches()), 'plotted_values': data[name]}
    def model_key(fig, rows=None):
        handles = []
        for cid in ids:
            label = names[cid]
            if rows is not None and settings['revision_auroc_in_legend']:
                r = next(r for r in rows if r['configuration'] == cid)
                interval = r['interval95']
                label += f"\nAUROC {r['auroc']:.3f}"
                if interval is not None:
                    label += f" ({interval[0]:.3f}–{interval[1]:.3f})"
            handles.append(Line2D([], [], **styles[cid], label=label))
        legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .01),
                            ncol=settings['revision_legend_columns'], handlelength=3,
                            columnspacing=1.5, borderpad=.6)
        fig.canvas.draw()
        return legend.get_window_extent(fig.canvas.get_renderer()).height / (fig.dpi*fig.get_figheight()) + .08
    for name in ('combined_candidate_a',):
        if name not in wanted:
            continue
        payload = data[name]
        size = settings['revision_sizes']['combined']
        fig = plt.figure(figsize=size)
        schematic = fig.add_axes((.012, .405, .285, .565))
        schematic.set(xlim=(0, 1), ylim=(0, 1)); schematic.axis('off')
        fig.text(.02, .978, 'a', weight='bold', fontsize=settings['main_panel_fontsize'], va='bottom')
        fig.text(.33, .978, 'b', weight='bold', fontsize=settings['main_panel_fontsize'], va='bottom')
        from matplotlib.patches import Rectangle, FancyArrowPatch
        from matplotlib.path import Path as VectorPath
        # A tall study diagram fits beside all 28 accuracy rows at page size.
        # The decision-only branch bypasses the primary/sensitivity boxes.
        def box(x, y, width, height, text, color='#F1F2F3'):
            schematic.add_patch(Rectangle((x-width/2, y-height/2), width, height,
                                          facecolor=color, edgecolor=gray, linewidth=.9, zorder=2))
            schematic.text(x, y, text, ha='center', va='center', fontsize=settings['figure1_schematic_fontsize'],
                           color='black', linespacing=1.25, zorder=3)
        def arrow(*vertices):
            schematic.add_patch(FancyArrowPatch(
                path=VectorPath(vertices, [VectorPath.MOVETO]+[VectorPath.LINETO]*(len(vertices)-1)),
                arrowstyle='-|>', mutation_scale=11, color=gray, lw=1, zorder=1))
        box(.5, .94, .70, .085, 'MedQA\n1,273 questions', 'white')
        box(.31, .77, .48, .13, '13 decision\nmodels\n3 runs', '#F5DDC1')
        box(.81, .77, .37, .13, '15 generative\nconfigurations\n1 run', '#DCE8EB')
        arrow((.5, .8975), (.5, .87), (.31, .87), (.31, .835))
        arrow((.5, .87), (.81, .87), (.81, .835))
        box(.54, .57, .85, .13, 'Primary comparison\nRun 1\nAccuracy, cost and duration')
        arrow((.31, .705), (.31, .635))
        arrow((.81, .705), (.81, .635))
        box(.54, .405, .85, .085, 'Sensitivity analyses')
        arrow((.54, .505), (.54, .4475))
        box(.54, .235, .85, .13, 'Probability quality, run 1\nDiscrimination, calibration\nand selective prediction')
        box(.54, .065, .85, .10, 'Repeatability, runs 1-3\nAnswers and probabilities')
        arrow((.07, .77), (.025, .77), (.025, .235), (.115, .235))
        arrow((.025, .235), (.025, .065), (.115, .065))
        ax = fig.add_axes((.63, .405, .35, .565))
        rows = sorted(payload['accuracy'], key=lambda r: (-r['accuracy'], r['configuration']))
        for i, r in enumerate(rows):
            value, (lo, hi) = r['accuracy'], r['wilson95']
            ax.barh(i, 100*value, height=.65, color=category_color(r['configuration']),
                    edgecolor=gray, linewidth=.7, zorder=2)
            ax.errorbar(100*value, i, xerr=[[100*(value-lo)], [100*(hi-value)]],
                        fmt='none', ecolor=gray, elinewidth=1.2, capsize=1.8, zorder=3)
            primary = summary['primary'][r['configuration']]
            ax.text(2, i, f"{100*value:.1f}% ({primary['correct']}/{primary['planned_n']})",
                    ha='left', va='center', fontsize=9,
                    color='white' if r['configuration'].startswith('G') else 'black', zorder=4)
        from textwrap import fill
        figure1_names = {}
        for r in rows:
            cid = r['configuration']
            config = canonical_configuration(CURRENT_NAMESPACE, cid)
            label = config['display_name']
            if cid.startswith('G'):
                label += f" ({config['effort']})"
            figure1_names[cid] = label
        labels = [f"{r['configuration']}  {fill(figure1_names[r['configuration']], 33)}" for r in rows]
        ax.set(yticks=range(len(rows)), yticklabels=labels, xlabel='Accuracy (%)',
               xlim=(0, 100))
        ax.tick_params(axis='y', labelsize=settings['figure1_fontsize'], length=0, pad=7)
        ax.tick_params(axis='x', labelsize=settings['figure1_fontsize'])
        ax.xaxis.label.set_size(settings['figure1_fontsize'])
        ax.set_ylim(len(rows)-.4, -.6); ax.grid(axis='x')
        handles = [Line2D([], [], marker='o', color='none', markerfacecolor=category_color(cid),
                          markeredgecolor=gray, label=label)
                   for cid, label in (('G01', 'Generative configurations'), ('D01', 'Decision models'))]
        handles.append(Line2D([], [], marker='D', color=gray, linestyle='none', label='Nondominated'))
        for handle in handles:
            handle.set_markersize(6)
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.003),
                   ncol=3, fontsize=settings['figure1_fontsize'],
                   handletextpad=.5, columnspacing=.9, borderpad=.6)
        all_points = payload['accuracy_cost']['points'] + payload['accuracy_latency']['points']
        accuracy_floor = 5*floor(min(p['accuracy']*100 for p in all_points)/5)-5
        for row_index, (key, xlabel) in enumerate((('accuracy_cost', 'Cost (USD/1,000 questions)'),
                                                  ('accuracy_latency', 'Median duration (s)'))):
            panel_height = .345*size[0]/size[1]
            ax = fig.add_axes((.075 + .50*row_index, .105, .345, panel_height))
            ax.set_box_aspect(1)
            fig.text(.02+.50*row_index, .12+panel_height, 'cd'[row_index], weight='bold', fontsize=settings['main_panel_fontsize'])
            points = payload[key]['points']
            for p in points:
                ax.scatter(p['x'], p['accuracy']*100, color=category_color(p['configuration']),
                           # Matplotlib's D path has unit area; its o path has
                           # area pi/4. Match visible area rather than nominal s.
                           marker='D' if p['nondominated'] else 'o',
                           s=36*.7853981633974483 if p['nondominated'] else 36,
                           edgecolors=gray, linewidths=.9, zorder=3)
            frontier = sorted({(p['x'], p['accuracy']*100) for p in points if p['nondominated']})
            if frontier:
                ax.plot(*zip(*frontier), color=gray, linestyle='--', linewidth=1.5)
            ax.set_xscale('log'); ax.minorticks_off()
            ax.set(xlabel=xlabel+'\n(log scale)', ylabel='Accuracy (%)',
                   ylim=(accuracy_floor, 102), yticks=range(max(0, int(accuracy_floor)), 101, 10))
            ax.margins(x=.20); ax.grid(axis='both', linewidth=.8)
            ax.tick_params(labelsize=settings['figure1_fontsize'])
            ax.xaxis.label.set_size(settings['figure1_fontsize'])
            ax.yaxis.label.set_size(settings['figure1_fontsize'])
            warnings.extend(_revision_resource_labels(ax, points, settings, key, figure1_names))
        from matplotlib.text import Text
        for text in fig.findobj(Text):
            if text.get_color() != 'white':
                text.set_color('black')
        capture(name, fig)
    probability_names = {**names, 'D01': 'JEV', 'D02': 'GPT-6 Luna Decisions'}
    # Main labels are stable whether the caller supplies table labels or not.
    main_probability_names = {cid: canonical_configuration(CURRENT_NAMESPACE, cid)['display_name']
                              for cid in ids}
    main_probability_names.update({'D01': 'JEV', 'D02': 'GPT-6 Luna Decisions'})
    panel_types = {
        'confidence_roc': ('ROC', 'False-positive rate', 'True-positive rate'),
        'calibration': ('Calibration', 'Mean selected-answer\nprobability', 'Observed accuracy'),
        'decision_risk_coverage': ('Error vs coverage', 'Eligible coverage (%)', 'Error rate (%)'),
        'confidence_cutoff': ('Error vs cutoff', 'Probability cutoff', 'Error rate (%)'),
    }

    def probability_panel(ax, cid, kind, row, panel_label, *, main=False):
        from textwrap import fill
        # Equal physical axis lengths, including panels with unlike data units.
        ax.set_box_aspect(1)
        color = styles[cid]['color']
        label = (main_probability_names if main else probability_names)[cid]
        ax.set_title(fill(label, width=22), x=.13 if main else .14,
                     loc='left', fontsize=settings['main_fontsize'] if main else 9, linespacing=1.2, pad=8 if main else 6)
        ax.annotate(panel_label if main else f'{panel_label})', (0, 1), xycoords='axes fraction',
                    xytext=(0, 8) if main else (0, 6), textcoords='offset points',
                    fontsize=settings['main_panel_fontsize'] if main else 9, fontweight='bold', va='bottom')
        ax.tick_params(labelsize=settings['main_tick_fontsize'] if main else 8, labelbottom=True, labelleft=True)
        ax.grid(axis='both')
        ax.set(xlim=(0, 1), ylim=(0, 1), xticks=(0, .5, 1), yticks=(0, .5, 1))
        if kind == 'calibration':
            bins = [p for p in row['bins'] if p['n']]
            errors = [[p['observed_accuracy']-p['accuracy_wilson95'][0] for p in bins],
                      [p['accuracy_wilson95'][1]-p['observed_accuracy'] for p in bins]]
            if any(e < -1e-12 for side in errors for e in side):
                raise ValueError('Calibration Wilson interval does not contain observed accuracy')
            ax.errorbar([p['mean_probability'] for p in bins],
                        [p['observed_accuracy'] for p in bins],
                        yerr=[[max(0., e) for e in side] for side in errors],
                        fmt='o', linestyle='none', color=color, markersize=3.5,
                        elinewidth=1.3 if main else .55, capthick=1.3 if main else None,
                        capsize=1.5, clip_on=False)
            ax.plot([0, 1], [0, 1], ':', color=gray, linewidth=1.3 if main else .7)
        elif kind == 'confidence_roc':
            ax.plot([p['false_positive_rate'] for p in row['points']],
                    [p['true_positive_rate'] for p in row['points']],
                    color=color, linewidth=1.3 if main else .9, marker=None)
            ax.plot([0, 1], [0, 1], ':', color=gray, linewidth=1.3 if main else .7)
            interval = row['interval95']
            interval_text = (f"95% CI {interval[0]:.3f}–{interval[1]:.3f}"
                             if interval is not None else '95% CI unavailable')
            ax.text(.96, .04, f"AUROC {row['auroc']:.3f}\n{interval_text}",
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=9.5 if main else 7.5,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': 1})
        elif kind == 'decision_risk_coverage':
            ax.plot([p['coverage']*100 for p in row['curve']],
                    [p['risk']*100 for p in row['curve']],
                    color=color, linewidth=1.3 if main else .9, marker=None)
            ax.set(xlim=(0, 100), ylim=(0, 100), xticks=(0, 50, 100), yticks=(0, 50, 100))
        else:
            x, risk, _ = _confidence_cutoff_steps(row)
            ax.step(x, risk, where='pre', color=color, linewidth=1.3 if main else .9, marker=None)
            ax.set(ylim=(0, 100), yticks=(0, 50, 100))
        if main:
            _, xlabel, ylabel = panel_types[kind]
            # Shared labels are placed once per six-model block below.
            if kind in ('decision_risk_coverage', 'confidence_cutoff'):
                from matplotlib.ticker import PercentFormatter
                ax.set(ylim=(0, 25), yticks=(0, 10, 20, 25))
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            else:
                ax.set(xticks=(0, .5, 1), yticks=(0, .5, 1))

    def probability_row_labels(axes, kind):
        # One centered x label and one left y label per three-model row.
        _, xlabel, ylabel = panel_types[kind]
        for row_axes in axes:
            row_axes[0].set_ylabel(ylabel, fontsize=8)
            row_axes[1].set_xlabel(xlabel, fontsize=8)
            for ax in row_axes[1:]:
                ax.tick_params(labelleft=False)

    for name, kinds in (
            ('probability_roc_calibration', ('confidence_roc', 'calibration')),
            ('probability_errors', ('decision_risk_coverage', 'confidence_cutoff'))):
        if name not in wanted:
            continue
        payload = data[name]
        fig = plt.figure(figsize=settings['revision_sizes']['probability_main'])
        rows = {kind: {r['configuration']: r for r in payload[kind]} for kind in kinds}
        headings = {'decision_risk_coverage': 'Error versus coverage',
                    'confidence_cutoff': 'Error versus probability cutoff'}
        # Two blocks, each with six models in two rows and three columns.
        # Read panels a-f across the upper block, then g-l across the lower block.
        for block, kind in enumerate(kinds):
            top, bottom = ((.925, .565) if block == 0 else (.435, .075))
            grid = fig.add_gridspec(2, 3, left=.115, right=.97, bottom=bottom, top=top,
                                   wspace=.30, hspace=.60)
            for index, cid in enumerate(payload['model_order']):
                ax = fig.add_subplot(grid[index//3, index%3])
                probability_panel(ax, cid, kind, rows[kind][cid],
                                  chr(97+6*block+index), main=True)
            heading_y = .984 if block == 0 else .494
            fig.text(.54, heading_y, headings.get(kind, panel_types[kind][0]),
                     ha='center', va='top', fontsize=settings['main_panel_fontsize'])
            _, xlabel, ylabel = panel_types[kind]
            fig.text(.035, (top+bottom)/2, ylabel, ha='center', va='center',
                     rotation=90, fontsize=settings['main_fontsize'])
            fig.text(.54, bottom-.043, xlabel.replace('\n', ' '), ha='center', va='top',
                     fontsize=settings['main_fontsize'])
        capture(name, fig)

    # Complete model-by-model supplements, with the same axes and drawing code.
    for name, kind in (
            ('probability_roc_supplement', 'confidence_roc'),
            ('probability_calibration', 'calibration'),
            ('probability_coverage', 'decision_risk_coverage'),
            ('probability_cutoff_supplement', 'confidence_cutoff')):
        if name not in wanted:
            continue
        rows = {r['configuration']: r for r in data[name]}
        fig, axes = plt.subplots(5, 3, sharex='row', sharey='row',
                                 figsize=settings['revision_sizes'].get('probability_multiples', (7.2, 11.5)),
                                 layout='constrained')
        fig.suptitle(panel_types[kind][0], fontsize=11)
        for index, cid in enumerate(ids):
            probability_panel(axes.flat[index], cid, kind, rows[cid], chr(97+index))
        for ax in list(axes.flat)[len(ids):]:
            ax.set_visible(False)
        probability_row_labels(axes, kind)
        # The last supplementary row contains only one model.
        axes[-1, 0].set_xlabel(panel_types[kind][1], fontsize=8)
        capture(name, fig)
    if 'repeatability_supplement' in wanted:
        payload = data['repeatability_supplement']
        fig, axes = plt.subplots(2, 3, figsize=settings['revision_sizes']['repeatability'], sharex=True, sharey=True)
        xmax = min(1., max([p['change'] for rows in payload.values() for r in rows for p in r['ecdf']]+[.01])*1.05)
        for row_index, key in enumerate(('repeatability_confidence', 'repeatability_vector')):
            for column, pair in enumerate(((1,2), (1,3), (2,3))):
                ax = axes[row_index,column]; letter(ax, 'abcdef'[row_index*3+column])
                for r in payload[key]:
                    if (r['earlier_run'], r['later_run']) != pair or not r['ecdf']:
                        continue
                    curve = r['ecdf']
                    ax.step([0]+[p['change'] for p in curve]+[xmax],
                            [0]+[100*p['fraction'] for p in curve]+[100*curve[-1]['fraction']],
                            where='post', **styles[r['configuration']])
                ax.set(xlim=(0,xmax), ylim=(0,102)); ax.grid(axis='both')
                if row_index == 0:
                    ax.text(.5, 1.04, f'Runs {pair[0]}–{pair[1]}', transform=ax.transAxes, ha='center')
            axes[row_index,0].set_ylabel('Cumulative eligible questions (%)')
            axes[row_index,1].set_xlabel('Absolute own-selected probability change' if row_index == 0
                                         else 'Total variation of normalized option probabilities', labelpad=10)
        bottom = model_key(fig)
        fig.subplots_adjust(left=.11, right=.98, top=.94, bottom=bottom, wspace=.18, hspace=.5)
        capture('repeatability_supplement', fig)
    return figures, {**metadata, 'settings': settings, 'provenance': summary['provenance'],
                     'figures': evidence, 'layout_review': warnings}


def presentation_notebook_record(path):
    """Bind review-copy cell sources independently of execution outputs."""
    import json
    from common import fingerprint
    path = Path(path)
    notebook = json.loads(path.read_text())
    cells = [{"cell_type": c["cell_type"], "source": "".join(c["source"])}
             for c in notebook["cells"]]
    return {"file": path.name, "cell_sources_sha256": fingerprint(cells)}


def build_figures(summary, *, font=None, width=None, category_limit=None, plot_data=None, settings=None,
                  model_names=None, figure_names=None, _model_panel=False):
    """Draw saved coordinates and intervals; return previews and plotted-value evidence."""
    from copy import deepcopy
    from matplotlib import pyplot as plt
    revision = (settings or FIGURE_SETTINGS).get('report_layout') == 'coauthor-20261021'
    if summary.get('configuration_namespace') and not _model_panel:
        from analysis.support.notebook import figure_data
        plot_data = presentation_plot_data(
            summary, figure_data(summary) if plot_data is None else plot_data,
            confidence_display='roc', revision=revision)
        if figure_names is None:
            figure_names = current_report_figures(summary, revision=revision)
    settings = deepcopy(FIGURE_SETTINGS if settings is None else settings)
    for key, value in (("font", font), ("width_inches", width), ("category_limit", category_limit)):
        if value is not None:
            settings[key] = value
    if plot_data is None:
        # Existing historical CLI callers can still supply their numerical summary.
        from analysis.support.notebook import figure_data
        plot_data = figure_data(summary)
    if revision:
        return _build_revision_figures(summary, plot_data, settings, model_names, figure_names)
    plot_data = {k: deepcopy(v) for k, v in plot_data.items()
                 if not k.endswith(('_repeatability_provider_confidence', 'repeatability_sankey'))}
    if figure_names is not None:
        requested = set(figure_names)
        plot_data = {name: values for name, values in plot_data.items() if name in requested}
    confidence_panel = plot_data.get("confidence_panel")
    if confidence_panel:
        plot_data.update(deepcopy(confidence_panel))
    resource_panel = plot_data.get("resource_panel")
    if resource_panel:
        plot_data.update(deepcopy(resource_panel))
    model_names = {} if model_names is None else model_names
    model_name = lambda cid: model_names.get(cid, cid)
    metadata = configure_style(settings=settings)
    width, category_limit = settings["width_inches"], settings["category_limit"]
    teal, dark_teal, orange = (settings["colors"][c] for c in ("teal", "dark_teal", "orange"))
    gray = settings["rc_params"]["axes.edgecolor"]
    model_color = lambda cid: orange if (cid.startswith("D") or cid == "J01") else teal
    figures, evidence = {}, {}

    def capture(fig, name, *, layout=True):
        if layout:
            fig.tight_layout()
        figures[name] = fig
        evidence[name] = {"size_inches": list(fig.get_size_inches()), "plotted_values": plot_data[name]}

    rows = plot_data.get("accuracy", [])
    if rows:
        fig, ax = plt.subplots(figsize=(width, max(3, .3*len(rows)+1.4)))
        rows = sorted(rows, key=lambda row: row["accuracy"], reverse=True)
        names, accuracies = [model_name(r["configuration"]) for r in rows], [100*r["accuracy"] for r in rows]
        errors = [[100*(r["accuracy"]-r["wilson95"][0]) for r in rows],
                  [100*(r["wilson95"][1]-r["accuracy"]) for r in rows]]
        colors = [model_color(r["configuration"]) for r in rows]
        value_labels = []
        for row in rows:
            counts = summary["primary"][row["configuration"]]
            percentage = f"{100*row['accuracy']:.2f}"
            value_labels.append(f"{percentage}% ({counts['correct']}/{counts['planned_n']})")
        if len(rows) > category_limit:
            bars = ax.barh(names, accuracies, height=.65, color=colors, edgecolor=gray)
            annotations = ax.bar_label(bars, labels=value_labels, label_type="center", fontsize=10, zorder=4)
            ax.errorbar(accuracies, range(len(rows)), xerr=errors, fmt="none", ecolor=gray, elinewidth=.8)
            ax.set(xlabel="Accuracy (%)", xlim=(0, 100))
            ax.invert_yaxis(); ax.grid(axis="x")
        else:
            bars = ax.bar(names, accuracies, width=.65, color=colors, edgecolor=gray)
            annotations = ax.bar_label(bars, labels=value_labels, label_type="center", fontsize=10,
                                       rotation=90, zorder=4)
            ax.errorbar(range(len(rows)), accuracies, yerr=errors, fmt="none", ecolor=gray, elinewidth=.8)
            ax.set(ylabel="Accuracy (%)", ylim=(0, 100)); ax.grid(axis="y")
        for annotation, row in zip(annotations, rows):
            annotation.set_color(gray if (row["configuration"].startswith("D") or row["configuration"] == "J01") else "white")
        capture(fig, "accuracy")
    for family in ("S1", "S2"):
        name = family.lower()+"_paired_effects"
        contrasts = plot_data.get(name, [])
        if not contrasts:
            continue
        fig, ax = plt.subplots(figsize=(width, max(3, .3*len(contrasts)+1.4)))
        ax.scatter([r["difference_pp"] for r in contrasts], range(len(contrasts)), s=28, color=dark_teal)
        for i, row in enumerate(contrasts):
            if row["interval95_pp"] is not None:
                ax.plot(row["interval95_pp"], [i, i], color=dark_teal)
        ax.axvline(0, color=gray, linewidth=.6)
        labels = [f"{model_name(r['positive_configuration'])} − {model_name(r['negative_configuration'])}"
                  for r in contrasts]
        ax.set(yticks=range(len(contrasts)), yticklabels=labels,
               xlabel=("Positive minus negative configuration accuracy (percentage points)" if family == "S1" and "decision_models" in summary else
                       "JEV minus LLM accuracy (percentage points)" if family == "S1" else
                       "Upper minus lower native setting accuracy (percentage points)"))
        ax.invert_yaxis(); ax.grid(axis="x")
        capture(fig, name)
    resource_axes = {}
    if resource_panel:
        resource_fig, axes = plt.subplots(2, 1, figsize=(max(width, 11.), 14.))
        # Fix both axes before placing labels in display coordinates. Later
        # layout passes would invalidate the collision checks for panel a.
        resource_fig.subplots_adjust(left=.10, right=.98, bottom=.06, top=.96, hspace=.30)
        resource_axes = dict(zip(("accuracy_cost", "accuracy_latency"), axes))
        for ax, label in zip(axes, ("a)", "b)")):
            ax.text(0, 1.03, label, transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=14, fontweight="bold")
    for name, label in (("accuracy_cost", "Mean final-attempt charge (USD)"),
                        ("accuracy_estimated_cost", "Mean estimated charge (USD)"),
                        ("accuracy_latency", "Median request-wall latency (seconds)")):
        saved = plot_data.get(name)
        if saved is None:
            continue
        points = saved["points"]
        fig, ax = ((resource_fig, resource_axes[name]) if name in resource_axes
                   else plt.subplots(figsize=(max(width, 11.), 7.)))
        for p in points:
            ax.scatter([p["x"]], [p["accuracy"]*100], color=model_color(p["configuration"]),
                       marker="D" if p["nondominated"] else "o",
                       s=settings.get("resource_marker_area", 75), edgecolors=gray, linewidths=.6, zorder=3)
        frontier = sorted({(p["x"], p["accuracy"]*100) for p in points if p["nondominated"]})
        if frontier:
            ax.plot([p[0] for p in frontier], [p[1] for p in frontier],
                    color=gray, linestyle="--", linewidth=.9, zorder=2)
        if all(p["x"] > 0 for p in points):
            ax.set_xscale("log")
            label += " (log scale)"
        accuracies = [100*p["accuracy"] for p in points]
        ax.set(xlabel=label, ylabel="Accuracy (%)",
               ylim=(max(0, min(accuracies)-4), min(100, max(accuracies)+4)))
        from matplotlib.lines import Line2D
        ax.legend(handles=[Line2D([], [], color=gray, marker="D", linestyle="--",
                                  label="Pareto frontier (point estimates)")], loc="lower left")
        ax.grid(axis="both"); ax.margins(x=.25)
        # Choose nearby label positions using actual text bounds, in display space.
        if name not in resource_axes:
            fig.tight_layout()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        from matplotlib.text import Text
        from matplotlib.transforms import Bbox
        from matplotlib.path import Path as DisplayPath
        occupied = [ax.get_legend().get_window_extent(renderer)]
        # Reserve space for every marker, including nearby points whose labels
        # have not yet been placed. Allow for the larger diamond and padding.
        radius = (settings.get("resource_marker_area", 75)**.5 / 2**.5 + 3) * fig.dpi / 72
        for p in points:
            x, y = ax.transData.transform((p["x"], p["accuracy"]*100))
            occupied.append(Bbox.from_extents(x-radius, y-radius, x+radius, y+radius))

        def overlap_area(first, second):
            intersection = Bbox.intersection(first, second)
            return 0. if intersection is None else intersection.width*intersection.height

        overrides = settings.get("resource_label_offsets", {}).get(name, {})
        # Search outward in points, independently of logarithmic data scaling.
        offsets = sorted(((dx, dy) for dx in (-80, -56, -32, -16, 0, 16, 32, 56, 80)
                          for dy in (-80, -56, -32, -16, 0, 16, 32, 56, 80)
                          if dx or dy), key=lambda offset: (offset[0]**2+offset[1]**2, -offset[1], -offset[0]))
        ordered = sorted(points, key=lambda p: (-p["accuracy"], p["x"], p["configuration"]))
        leaders = []
        for p in ordered:
            anchor = ax.transData.transform((p["x"], p["accuracy"]*100))
            own_marker = occupied[1 + points.index(p)]
            annotation = ax.annotate(model_name(p["configuration"]), (p["x"], p["accuracy"]*100),
                                     xytext=(7, 7), textcoords="offset points", fontsize=10, zorder=4,
                                     arrowprops={"arrowstyle": "-", "color": gray, "linewidth": .6,
                                                 "shrinkA": 2, "shrinkB": 6,
                                                 "connectionstyle": "arc3,rad=0"})
            candidates = ([overrides[p["configuration"]]] if p["configuration"] in overrides else
                          offsets)
            best = None
            for dx, dy in candidates:
                annotation.set_position((dx, dy))
                annotation.set_ha("left" if dx > 0 else "right" if dx < 0 else "center")
                annotation.set_va("bottom" if dy > 0 else "top" if dy < 0 else "center")
                annotation.update_positions(renderer)
                # Check text alone so leader lines do not inflate label bounds.
                bounds = Text.get_window_extent(annotation, renderer).padded(3*fig.dpi/72)
                outside = max(0., bounds.width*bounds.height-overlap_area(bounds, ax.bbox))
                overlap = sum(overlap_area(bounds, other) for other in occupied)
                # Join the nearest label edge to the actual data point. Prefer
                # connectors that do not cross markers, labels or other leaders.
                endpoint = (min(max(anchor[0], bounds.x0), bounds.x1),
                            min(max(anchor[1], bounds.y0), bounds.y1))
                leader = DisplayPath([anchor, endpoint])
                crossings = sum(leader.intersects_bbox(other, filled=False)
                                for other in occupied if other is not own_marker)
                crossings += sum(leader.intersects_path(other, filled=False)
                                 or other.intersects_bbox(bounds, filled=False) for other in leaders)
                score = (outside, overlap, crossings, dx*dx+dy*dy)
                if best is None or score < best[0]:
                    best = (score, (dx, dy), annotation.get_ha(), annotation.get_va(), bounds, leader)
                if outside == 0 and overlap == 0 and crossings == 0:
                    break
            # If the region is crowded, keep the least-overlapping candidate
            # instead of blindly accepting the final position searched.
            _, position, horizontal, vertical, bounds, leader = best
            annotation.set_position(position)
            annotation.set_ha(horizontal)
            annotation.set_va(vertical)
            annotation.update_positions(renderer)
            occupied.append(bounds)
            leaders.append(leader)
        # A second layout pass would invalidate the display-space checks above.
        if name not in resource_axes:
            capture(fig, name, layout=False)
    if resource_panel:
        capture(resource_fig, "resource_panel", layout=False)
    models = summary.get("decision_models", {"J01": {}})
    for cid in models:
        prefix, model_label = ("jev", model_name(cid)) if cid == "J01" else (cid.lower(), model_name(cid))
        name = prefix+"_risk_coverage"
        curve = plot_data.get(name)
        if curve:
            fig, ax = plt.subplots(figsize=(width, 4.))
            ax.plot([r["coverage"]*100 for r in curve], [r["risk"]*100 for r in curve],
                    marker="o", markersize=5, color=teal, drawstyle="steps-post")
            ax.set(xlabel="Coverage within eligible valid-score answers (%)", ylabel="Error risk (%)",
                   xlim=(0, 100), ylim=(0, 100)); ax.grid(axis="y")
            capture(fig, name)
        name = prefix+"_reliability"
        bins = plot_data.get(name)
        if bins is not None:
            fig, ax = plt.subplots(figsize=(width, 4.))
            populated = [b for b in bins if b["n"]]
            ax.plot([0, 1], [0, 1], color=gray, linestyle="--")
            ax.plot([b["mean_probability"] for b in populated], [b["observed_accuracy"] for b in populated],
                    marker="o", markersize=5, color=teal)
            for b in populated:
                if b.get("accuracy_wilson95") is not None:
                    ax.plot([b["mean_probability"]]*2, b["accuracy_wilson95"], color=teal, linewidth=.6)
                ax.annotate(f"n={b['n']}", (b["mean_probability"], b["observed_accuracy"]),
                            xytext=(4, 4), textcoords="offset points", fontsize=10)
            ax.set(xlabel="Mean selected-option probability", ylabel="Observed correctness fraction",
                   xlim=(0, 1), ylim=(0, 1)); ax.grid(axis="y")
            capture(fig, name)
        name = prefix+"_stability_accuracy"
        scores = plot_data.get(name)
        if scores:
            fig, ax = plt.subplots(figsize=(width, 3.5))
            ax.bar(["Run "+str(r["replicate"]) for r in scores], [r["accuracy"]*100 for r in scores],
                   width=.65, color=teal, edgecolor=gray)
            for i, r in enumerate(scores):
                if r["wilson95"] is not None:
                    ax.plot([i, i], [100*x for x in r["wilson95"]], color=gray, linewidth=.6)
            ax.set(ylabel=model_label+" accuracy (%)", ylim=(0, 100)); ax.grid(axis="y")
            capture(fig, name)
        name = prefix+"_stability_agreement"
        pairs = plot_data.get(name)
        if pairs:
            fig, ax = plt.subplots(figsize=(width, 3.5))
            labels = [f"Runs {p['earlier_run']}–{p['later_run']}\nvalid n={p['valid_pair_n']}" for p in pairs]
            ax.bar(labels, [p["answer_agreement"]*100 for p in pairs], width=.65, color=teal, edgecolor=gray)
            ax.set(ylabel="Agreement among valid label pairs (%)", ylim=(0, 100)); ax.grid(axis="y")
            capture(fig, name)
    repeatability_panel = plot_data.get("repeatability_panel")
    confidence_ids = sorted(set(summary.get('decision_models', {})) | {
        row['configuration'] for rows in (repeatability_panel or {}).values() for row in rows
    })
    confidence_colors = settings.get('confidence_colors', FIGURE_SETTINGS['confidence_colors'])
    confidence_linewidth = plt.rcParams['lines.linewidth']
    confidence_styles = {
        cid: {'color': confidence_colors[i % len(confidence_colors)],
              'linestyle': '-',
              'linewidth': confidence_linewidth}
        for i, cid in enumerate(confidence_ids)
    }
    if repeatability_panel:
        from matplotlib.lines import Line2D
        from textwrap import fill
        fig, axes = plt.subplots(2, 3, figsize=(max(width, 14.), 8.5), sharex=True, sharey=True)
        metric_names = ("repeatability_confidence", "repeatability_vector")
        row_labels = ("Absolute selected-answer probability change",
                      "Total variation of normalized option distributions")
        all_rows = [row for name in metric_names for row in repeatability_panel[name]]
        xmax = min(1., max([p["change"] for row in all_rows for p in row["ecdf"]]+[.01])*1.05)
        counts = {}
        for row_index, (name, row_label) in enumerate(zip(metric_names, row_labels)):
            for column, pair in enumerate(((1, 2), (1, 3), (2, 3))):
                ax = axes[row_index, column]
                panel_label = "abcdef"[3*row_index+column]
                for row in repeatability_panel[name]:
                    if (row["earlier_run"], row["later_run"]) != pair:
                        continue
                    cid, curve = row["configuration"], row["ecdf"]
                    counts.setdefault(cid, {})[panel_label] = row["eligible_n"]
                    if not curve:
                        continue
                    # Extend the terminal cumulative value across the common
                    # domain; retain every saved jump, including zero changes.
                    ax.step([0]+[p["change"] for p in curve]+[xmax],
                            [0]+[100*p["fraction"] for p in curve]+[100*curve[-1]["fraction"]],
                            where="post", **confidence_styles[cid])
                ax.set(xlim=(0, xmax), ylim=(0, 100))
                if row_index == 0:
                    ax.text(.5, 1.04, f"Runs {pair[0]}–{pair[1]}", transform=ax.transAxes,
                            ha="center", va="bottom")
                ax.grid(axis="both")
                ax.text(0, 1.04, panel_label+")", transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=14, fontweight="bold")
            axes[row_index, 1].set_xlabel(row_label, labelpad=12)
            axes[row_index, 0].set_ylabel(
                "Cumulative eligible matched questions (%)",
                fontsize=axes[row_index, 1].xaxis.label.get_fontsize())
        handles = []
        for cid, panel_counts in counts.items():
            eligible = (f"eligible n={next(iter(panel_counts.values())):,}"
                        if len(panel_counts) == 6 and len(set(panel_counts.values())) == 1 else
                        "eligible n: "+", ".join(f"{label}) {n:,}" for label, n in panel_counts.items()))
            handles.append(Line2D([], [], **confidence_styles[cid],
                                  label=fill(f"{model_name(cid)} ({eligible})", width=48,
                                             break_long_words=False, break_on_hyphens=False)))
        legend = fig.legend(handles=handles, loc="lower center", ncol=3, mode="expand",
                            bbox_to_anchor=(.05, 0, .90, 0), borderaxespad=0,
                            fontsize=9, handlelength=3, labelspacing=.6)
        # Measure the three-column legend so only its actual height is reserved.
        fig.canvas.draw()
        legend_height = legend.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi
        fig.set_figheight(8.5 + legend_height + .2)
        fig.tight_layout(rect=(.02, (legend_height + .2) / fig.get_figheight(), 1, 1))
        left = axes[1, 0].get_position().x0
        right = axes[1, 2].get_position().x1
        legend_width = right - left
        legend.set_bbox_to_anchor(((1 - legend_width) / 2, .05 / fig.get_figheight(), legend_width, 0))
        capture(fig, "repeatability_panel", layout=False)
    for name, xlabel in (("repeatability_confidence", "Absolute change in native selected-answer probability"),
                         ("repeatability_vector", "Total variation of normalized option distributions")):
        rows = plot_data.get(name, [])
        if not rows:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(max(width, 10), 3.7), sharex=True, sharey=True)
        xmax = max([p["change"] for row in rows for p in row["ecdf"]]+[.01])*1.05
        for ax, pair in zip(axes, ((1, 2), (1, 3), (2, 3))):
            for row in rows:
                if (row["earlier_run"], row["later_run"]) != pair:
                    continue
                curve = row["ecdf"]
                jev = row["configuration"] == "J01"
                ax.step([0]+[p["change"] for p in curve], [0]+[100*p["fraction"] for p in curve],
                        where="post", color=dark_teal if jev else orange,
                        linestyle="-", linewidth=confidence_linewidth,
                        label=f"{model_name(row['configuration'])} (n={row['eligible_n']:,})")
            ax.set(xlim=(0, xmax), ylim=(0, 102), xlabel=f"Runs {pair[0]}–{pair[1]}")
            ax.grid(axis="y"); ax.legend(loc="lower right", fontsize=9)
        axes[0].set_ylabel("Eligible matched questions (%)")
        fig.supxlabel(xlabel)
        capture(fig, name)
    panel_fig, panel_axes = (None, None)
    if confidence_panel:
        legend_rows = (len(summary['decision_models']) + 1) // 2
        legend_height = .65 * legend_rows + .3
        panel_fig, panel_axes = plt.subplots(3, 1, figsize=(max(width, 10.), 19.5 + legend_height))
        panel_axes[2].sharey(panel_axes[1])
        for ax, label in zip(panel_axes, ("a)", "b)", "c)")):
            ax.set_box_aspect(1)
            ax.text(0, 1.04, label, transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=14, fontweight="bold")
    curves = plot_data.get("decision_risk_coverage", [])
    if curves:
        fig, ax = (panel_fig, panel_axes[1]) if confidence_panel else plt.subplots(figsize=(width, 4.))
        for i, row in enumerate(curves):
            curve = row["curve"]
            ax.plot([r["coverage"]*100 for r in curve], [r["risk"]*100 for r in curve],
                    **confidence_styles[row['configuration']],
                    label=f"{model_name(row['configuration'])} (n={row['eligible_n']:,})")
        ax.set(xlabel="Coverage within eligible valid-score answers (%)", ylabel="Error rate (%)",
               xlim=(0, 100), ylim=(0, max(r["risk"]*100 for row in curves for r in row["curve"])*1.1 or 1))
        ax.grid(axis="both")
        if not confidence_panel:
            ax.legend(loc="best", fontsize=9)
        if not confidence_panel:
            capture(fig, "decision_risk_coverage")
    curves = plot_data.get("confidence_cutoff", [])
    if curves:
        from math import ceil, isfinite
        fig, risk_ax = (panel_fig, panel_axes[2]) if confidence_panel else plt.subplots(figsize=(width, 5.))
        observed_rates = []
        for row in curves:
            x, risk, _ = _confidence_cutoff_steps(row)
            observed_rates.extend(value for value in risk if isfinite(value))
            cid = row["configuration"]
            style = confidence_styles[cid]
            label = f"{model_names.get(cid, row['model'])} (eligible n={row['eligible_n']:,})"
            # With where='pre', y[i] applies on (x[i-1], x[i]]: an answer
            # at the cutoff remains retained, and drops only above it.
            risk_ax.step(x, risk, where="pre", label=label, **style)
            if row["curve"]:
                # Mark the inclusive highest threshold, even if it is zero
                # and the defined error-rate domain is a single point.
                highest = row["curve"][0]
                from fractions import Fraction
                risk_ax.plot(float(Fraction(highest["threshold_exact"])),
                             100*highest["risk"], marker="o", markersize=3,
                             linestyle="none", color=style["color"])
        upper = min(100, max(5, 5*ceil(max(observed_rates, default=0)*1.1/5)))
        risk_ax.set(xlabel="Minimum selected-answer probability",
                    ylabel="Error rate among retained answers (%)", xlim=(0, 1), ylim=(0, upper))
        risk_ax.grid(axis="both")
        if not confidence_panel:
            risk_ax.legend(loc="best", fontsize=9)
        if not confidence_panel:
            capture(fig, "confidence_cutoff")
    rows = plot_data.get("confidence_auroc", [])
    if rows:
        fig, ax = plt.subplots(figsize=(width, max(3, .6*len(rows)+1.4)))
        ax.scatter([r["auroc"] for r in rows], range(len(rows)), s=32, color=dark_teal)
        for i, row in enumerate(rows):
            if row["interval95"] is not None:
                ax.plot(row["interval95"], [i, i], color=dark_teal, linewidth=.9,
                        marker="|", markersize=7)
            ax.annotate(f"{row['auroc']:.3f}", (row["auroc"], i),
                        xytext=(0, 12), textcoords="offset points", ha="center", fontsize=10)
        ax.axvline(.5, color=gray, linewidth=.6, linestyle="--")
        ax.set(yticks=range(len(rows)),
               yticklabels=[f"{r['model']} (n={r['eligible_n']:,})" for r in rows],
               xlabel="Correctness AUROC (95% bootstrap interval)", xlim=(0, 1),
               ylim=(-.6, len(rows)-.4))
        ax.invert_yaxis(); ax.grid(axis="x")
        capture(fig, "confidence_auroc")
    rows = plot_data.get("confidence_roc", [])
    if rows:
        fig, ax = (panel_fig, panel_axes[0]) if confidence_panel else plt.subplots(figsize=(width, 5.))
        for i, row in enumerate(rows):
            interval = row["interval95"]
            label = f"{model_names.get(row['configuration'], row['model'])} (n={row['eligible_n']:,})\nAUROC {row['auroc']:.3f}"
            if interval is not None:
                label += f" (95% CI {interval[0]:.3f}–{interval[1]:.3f})"
            ax.plot([p["false_positive_rate"] for p in row["points"]],
                    [p["true_positive_rate"] for p in row["points"]],
                    **confidence_styles[row['configuration']], label=label)
        ax.plot([0, 1], [0, 1], color=gray, linewidth=confidence_linewidth, linestyle=":")
        ax.set(xlabel="False-positive rate (incorrect answers)",
               ylabel="True-positive rate (correct answers)", xlim=(0, 1), ylim=(0, 1))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(axis="both")
        if not confidence_panel:
            ax.legend(loc="lower right", fontsize=9,
                      labelspacing=.6, handlelength=3)
        if not confidence_panel:
            capture(fig, "confidence_roc")
    if confidence_panel:
        handles, labels = panel_axes[0].get_legend_handles_labels()
        legend = panel_fig.legend(handles, labels, loc='lower center', ncol=2,
                                  fontsize=10, handlelength=3, labelspacing=.8)
        # Reserve the rendered legend height instead of an oversized per-row
        # estimate, leaving only a small gap below panel c's axis label.
        panel_fig.canvas.draw()
        legend_height = legend.get_window_extent(panel_fig.canvas.get_renderer()).height / panel_fig.dpi + .15
        panel_fig.set_figheight(19.5 + legend_height)
        panel_fig.tight_layout(rect=(0, legend_height / panel_fig.get_figheight(), 1, 1))
        capture(panel_fig, "confidence_panel", layout=False)
    return figures, {**metadata, "settings": settings, "provenance": summary["provenance"], "figures": evidence}


def save_figure_record(folder, result_sha256, expected_values, record, *, root, reason, source_before):
    """Save a separate presentation record after checking values and source stability."""
    from importlib.metadata import version
    import platform
    import re
    if folder is None or not isinstance(reason, str) or not reason.strip():
        raise ValueError("Supply a fresh JEV_RENDER_DIR and a nonempty JEV_RENDER_REASON")
    if not isinstance(result_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", result_sha256):
        raise ValueError("Presentation requires the saved numerical-result digest")
    values = {name: r["plotted_values"] for name, r in record["figures"].items()}
    if values != expected_values:
        raise ValueError("Figure quantities or coverage differ from the saved numerical results")
    if presentation_code_record(root) != source_before:
        raise ValueError("Plotting source changed during rendering; restart the kernel and render again")
    record = {"schema_version": 1, "type": "analysis_figure_record", "created_at": utc_now(),
              "reason": reason.strip(), "results_sha256": result_sha256,
              "presentation_code": source_before,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "packages": {n: version(n) for n in ("matplotlib", "numpy", "pandas", "ipython")}},
              **record}
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder/"figure_record.json", record)
    return record


def draw_figures(summary, output):
    """Retained historical CLI SVG export; notebook modes produce previews."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    output = Path(output)
    figures, manifest = build_figures(summary)
    manifest["summary_sha256"] = file_hash(output/"summary.json")
    for name, fig in figures.items():
        path = output/(name+".svg")
        if path.exists():
            raise ValueError("Figure exists: "+str(path))
        fig.savefig(path, format="svg", metadata={"Date": None, "Creator": "JEV offline analysis v2"})
        manifest["figures"][name]["sha256"] = file_hash(path)
        plt.close(fig)
    write_new(output/"figure_manifest.json", manifest)
    return manifest


def current_report_figures(summary, secondary=True, *, revision=False):
    """Shared report panels; removed diagrams cannot enter current exports."""
    if revision:
        names = ['combined_candidate_a']
        if secondary:
            names += ['probability_roc_calibration', 'probability_errors',
                      'probability_coverage', 'probability_calibration',
                      'probability_roc_supplement', 'probability_cutoff_supplement',
                      'repeatability_supplement']
        return names
    names = ['accuracy', 'resource_panel']
    if secondary:
        names += ['confidence_panel', 'repeatability_panel']
    return names


def report_plot_values(summary, saved_plot_data, *, revision=False):
    """Expected current-report quantities independently derived from saved data."""
    data = presentation_plot_data(summary, saved_plot_data, confidence_display='roc', revision=revision)
    values = {k: data[k] for k in current_report_figures(summary, revision=revision) if k in data}
    return values


def export_current_report(folder, result_path, result_sha256, figures, record, *, root, source_before,
                          formats=('svg',), dpi=300):
    """Manual render-cell export of complete, authenticated expanded results only."""
    from analysis.support.notebook import load_results, fresh_output_directory
    from analysis.methods.expanded import ROSTER
    import pandas as pd
    result = load_results(result_path, result_sha256)
    summary = result['summary']
    if (not summary.get('configuration_namespace') or set(summary['primary']) != set(ROSTER)
            or summary['coverage']['expected'] != 68742 or not summary['study_complete']):
        raise ValueError('Current export requires the complete 28-configuration expanded benchmark')
    if not formats or not set(formats) <= {'svg', 'png', 'pdf'}:
        raise ValueError('Choose reviewed SVG, PNG or PDF formats')
    folder = fresh_output_directory(folder, root)
    revision = record.get('settings', {}).get('report_layout') == 'coauthor-20261021'
    expected = report_plot_values(summary, result['figure_data'], revision=revision)
    if set(figures) != set(expected):
        raise ValueError('Render every available current-report panel before export')
    save_figure_record(folder, result_sha256, expected, record, root=root,
        reason=('Coauthor figure revision candidates (2026-10-21)' if revision else
                'Reviewed probability-only expanded report export (2026-10-12)'), source_before=source_before)
    files = {}
    for name, fig in figures.items():
        for fmt in formats:
            path = folder/(name+'.'+fmt)
            fig.savefig(path, format=fmt, dpi=dpi)
            files[path.name] = file_hash(path)
    tables = {'primary': [{'configuration': cid, **row} for cid, row in summary['primary'].items()],
        'paired': [{'family': family, **row} for family, rows in summary['paired']['families'].items() for row in rows],
        'first_attempt': [{'configuration_run': cid, **row} for cid, row in summary['first_attempt_sensitivity'].items()],
        'visual_accuracy': [{'configuration_run': cid, **row} for cid, row in summary['missing_visual_sensitivity']['accuracy'].items()],
        'complete_valid': [{'family': family, **row} for family, rows in summary['complete_valid_sensitivity']['families'].items() for row in rows],
        'visual_paired': [{'family': family, **row} for family, rows in summary['missing_visual_sensitivity']['paired']['families'].items() for row in rows],
        'confidence': [{'configuration': cid, **{k:v for k,v in model['confidence'].items() if k not in ('vector_diagnostics','risk_coverage','reliability')}} for cid,model in summary['decision_models'].items()],
        'repeatability': [{'configuration': cid, **model['stability']} for cid,model in summary['decision_models'].items()],
        'repeat_resources': [{'configuration_run': cid, **row} for cid,row in summary['repeat_operations'].items()]}
    tables['frontiers'] = []
    for basis, entry in summary['frontiers'].items():
        projections = entry if 'nondominated' not in entry else {basis: entry}
        for projection, frontier in projections.items():
            if not isinstance(frontier, dict) or 'nondominated' not in frontier:
                continue
            for cid in summary['primary']:
                tables['frontiers'].append({'basis': basis, 'projection': projection, 'configuration': cid,
                    'nondominated': cid in frontier['nondominated'],
                    'excluded': frontier['excluded'].get(cid), 'dominated_by': frontier['dominated_by'].get(cid)})
    for name in ('reliability', 'risk_coverage', 'vector_diagnostics'):
        tables[name] = [{'configuration': cid, **row} for cid, model in summary['decision_models'].items()
                        for row in model['confidence'].get(name) or []]
    for name in ('answers','statuses','questions','secondary_answers','selected_attempts'):
        tables[name] = result['review_tables'][name]
    for name, rows in tables.items():
        path = folder/(name+'.csv')
        pd.json_normalize(rows).to_csv(path, index=False)
        files[path.name] = file_hash(path)
    if presentation_code_record(root) != source_before:
        raise ValueError('Presentation source changed during export; retain failed export and review')
    files['figure_record.json'] = file_hash(folder/'figure_record.json')
    write_new(folder/'export_manifest.json', {'results_sha256': result_sha256, 'files': files,
              'formats': list(formats), 'dpi': dpi, 'presentation_code': source_before,
              'report_scope': 'probability-only-20261012',
              'presentation_layout': record.get('settings', {}).get('report_layout', 'original'),
              'saved_sap': result['statistical_release']['bindings'].get('sap')})
    return folder
