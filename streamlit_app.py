import io
import os
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="ParaGeo HDH Data Explorer", layout="wide")

FT_COLORS = ["#0F5499","#E94164","#1B7F79","#E88900","#6E358B","#0072B2","#7A7D7D","#A23A3A","#3B7EA1","#8A8C0E"]
FT_TEMPLATE = dict(layout=dict(
    font=dict(family="Inter, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif", size=15, color="#222"),
    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", colorway=FT_COLORS,
    margin=dict(l=60, r=40, t=70, b=60),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.12)", zeroline=False, linecolor="rgba(0,0,0,0.65)", linewidth=1, mirror=True),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.12)", zeroline=False, linecolor="rgba(0,0,0,0.65)", linewidth=1, mirror=True),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified"
))

def _is_number(s: str) -> bool:
    try:
        float(s); return True
    except Exception:
        return False

def normalize_newlines(raw_text: str):
    t = raw_text.replace("\r\n","\n").replace("\r","\n")
    for _ in range(3):
        t = t.replace("\\r\\n","\n").replace("\\n","\n").replace("\\r","\n")
    return t, (t != raw_text)

HEADER_PATTERN = re.compile(r"^\s*!\s*Step\s*,", re.IGNORECASE)

def find_header_and_names(lines):
    for i, line in enumerate(lines):
        if HEADER_PATTERN.search(line):
            delim = "," if ("," in line) else ("\t" if ("\t" in line) else None)
            raw = line.strip()
            if raw.startswith("!"):
                raw = raw[1:].strip()
            parts = [p.strip() for p in (raw.split(",") if delim == "," else re.split(r"\s+", raw))]
            parts = [p for p in parts if p != ""]
            cleaned, seen = [], {}
            for p in parts:
                name = re.sub(r"[^\w\.]+","_", p).strip("_")
                if not name: continue
                if name in seen:
                    seen[name]+=1; name=f"{name}_{seen[name]}"
                else:
                    seen[name]=1
                cleaned.append(name)
            return i, cleaned, ("," if delim == "," else None)
    return None, None, None

def detect_table(lines):
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s: continue
        for delim in [",",";","\t"," "]:
            toks = (re.split(r"\s+", s) if delim==" " else s.split(delim))
            if len(toks) >= 2:
                numeric_ratio = sum(_is_number(t) for t in toks) / len(toks)
                if numeric_ratio >= 0.6:
                    ok=True
                    for j in range(1, min(6, len(lines)-i)):
                        s2 = lines[i+j].strip()
                        if not s2: continue
                        toks2 = (re.split(r"\s+", s2) if delim==" " else s2.split(delim))
                        if len(toks2)!=len(toks): ok=False; break
                        if sum(_is_number(t) for t in toks2)/len(toks2) < 0.5: ok=False; break
                    if ok: return i, delim, len(toks)
    return None, None, None

def load_hdh_bytes(file_bytes: bytes):
    text = file_bytes.decode("utf-8", errors="ignore")
    text, normalized = normalize_newlines(text)
    st.session_state['_hdh_normalized_nl'] = normalized

    lines = text.split("\n")
    header_idx, header_names, header_delim = find_header_and_names(lines)

    if header_idx is not None:
        off = header_idx + 1
        sub_start, sub_delim, _ = detect_table(lines[off:])
        if sub_start is not None:
            start = off + sub_start
            delim = header_delim or sub_delim or ","
        else:
            start, delim, _ = detect_table(lines)
    else:
        start, delim, _ = detect_table(lines)
    if start is None:
        raise ValueError("Could not detect a numeric data table in this file.")

    body = "\n".join(lines[start:])
    if delim == " ":
        df = pd.read_csv(io.StringIO(body), delim_whitespace=True, header=None, engine="python")
    else:
        df = pd.read_csv(io.StringIO(body), sep=delim, header=None, engine="python")
    df = df.replace(r'^\s*$', np.nan, regex=True).dropna(axis=1, how='all')

    if header_names:
        header_names = [h for h in header_names if h]
        if len(header_names) >= df.shape[1]:
            df.columns = header_names[:df.shape[1]]
        else:
            extra = [f"col_{i+1}" for i in range(df.shape[1]-len(header_names))]
            df.columns = header_names + extra
    else:
        df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
    return df

def title_value(key: str, computed: str) -> str:
    prev_key = f"_prev_default_{key}"
    cur = st.session_state.get(key, None)
    prev = st.session_state.get(prev_key, None)
    user_overridden = (cur is not None) and (prev is not None) and (cur != prev)
    value = cur if user_overridden and cur is not None else computed
    st.session_state[prev_key] = computed
    return value

# ----- Source management -----
if "_upl_version" not in st.session_state:
    st.session_state["_upl_version"] = 1
if "_sources" not in st.session_state:
    st.session_state["_sources"] = {}  # name -> {"bytes":..., "path":..., "kind":"upload"/"path"}

with st.sidebar:
    st.title("HDH Data Explorer")

    st.header("Import")
    # Uploader (browser) sources
    upl_key = f"uploader_v{st.session_state['_upl_version']}"
    upl_files = st.file_uploader("Add data sources (.hdh)",
                                 type=["hdh","csv","txt"],
                                 accept_multiple_files=True,
                                 key=upl_key)
    # Rebuild upload-kind entries from uploader
    new_sources = {nm:meta for nm,meta in st.session_state["_sources"].items() if meta.get("kind") != "upload"}
    if upl_files is not None:
        for f in upl_files:
            new_sources[f.name] = {"bytes": f.getvalue(), "path": None, "kind": "upload"}
    st.session_state["_sources"] = new_sources

    # Directory-based .hdh loader
    with st.expander("Load directory of .hdh files (enables true refresh)"):
        dir_path = st.text_input("Directory path", value=st.session_state.get("_dir_path",""))
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("Load directory"):
                if dir_path and os.path.isdir(dir_path):
                    st.session_state["_dir_path"] = dir_path
                    # Remove previous path-kind entries, then add all .hdh in dir (non-recursive)
                    keep_uploads = {nm:meta for nm,meta in st.session_state["_sources"].items() if meta.get("kind")=="upload"}
                    for nm in sorted(os.listdir(dir_path)):
                        if nm.lower().endswith(".hdh"):
                            fp = os.path.join(dir_path, nm)
                            keep_uploads[nm] = {"bytes": None, "path": fp, "kind": "path"}
                    st.session_state["_sources"] = keep_uploads
                    # Set active to first and reset dependent state
                    names = sorted(st.session_state["_sources"].keys())
                    if names:
                        st.session_state["_active_file"] = names[0]
                        st.session_state["_active_file_prev"] = names[0]
                        for k in ["x_col","y_cols","secondary_on","x_title","y1_title","y2_title",
                                  "_prev_default_x_title","_prev_default_y1_title","_prev_default_y2_title",
                                  "xmin","xmax","y1min","y1max","y2min","y2max"]:
                            st.session_state.pop(k, None)
                    st.rerun()
                else:
                    st.warning("Directory not found.")
        with c2:
            if st.button("Reload directory"):
                dp = st.session_state.get("_dir_path","")
                if dp and os.path.isdir(dp):
                    # Same as Load directory
                    keep_uploads = {nm:meta for nm,meta in st.session_state["_sources"].items() if meta.get("kind")=="upload"}
                    for nm in sorted(os.listdir(dp)):
                        if nm.lower().endswith(".hdh"):
                            fp = os.path.join(dp, nm)
                            keep_uploads[nm] = {"bytes": None, "path": fp, "kind": "path"}
                    st.session_state["_sources"] = keep_uploads
                    names = sorted(st.session_state["_sources"].keys())
                    if names and st.session_state.get("_active_file") not in names:
                        st.session_state["_active_file"] = names[0]
                        st.session_state["_active_file_prev"] = names[0]
                    st.rerun()

    # Active selection
    names = sorted(st.session_state["_sources"].keys())
    if names:
        cur_active = st.session_state.get("_active_file")
        if cur_active not in names:
            st.session_state["_active_file"] = names[0]
            cur_active = names[0]
            for k in ["x_col","y_cols","secondary_on","x_title","y1_title","y2_title",
                      "_prev_default_x_title","_prev_default_y1_title","_prev_default_y2_title",
                      "xmin","xmax","y1min","y1max","y2min","y2max"]:
                st.session_state.pop(k, None)
            st.session_state["_active_file_prev"] = cur_active

        active_name = st.selectbox("Active data file", names, index=names.index(cur_active), key="_active_file")

        prev_active = st.session_state.get("_active_file_prev")
        if active_name != prev_active:
            for k in ["x_col","y_cols","secondary_on","x_title","y1_title","y2_title",
                      "_prev_default_x_title","_prev_default_y1_title","_prev_default_y2_title",
                      "xmin","xmax","y1min","y1max","y2min","y2max"]:
                st.session_state.pop(k, None)
            st.session_state["_active_file_prev"] = active_name
            st.rerun()

        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("Refresh selected file"):
                # For path-kind files, reread from disk; for uploads, ask to re-upload
                meta = st.session_state["_sources"].get(st.session_state["_active_file"], {})
                if meta.get("kind") == "path" and meta.get("path") and os.path.isfile(meta["path"]):
                    # We re-read at load time below; just rerun
                    st.rerun()
                else:
                    st.info("This file came from the browser uploader. Please re-upload it to refresh contents.")
        with c2:
            if st.button("Clear all"):
                st.session_state["_upl_version"] += 1
                st.session_state["_sources"] = {}
                st.session_state.pop("_active_file", None)
                st.session_state.pop("_active_file_prev", None)
                st.rerun()
    else:
        st.info("No sources loaded. Add files above or load a directory.")

    st.header("Style")
    plot_style = st.selectbox("Trace style", ["Line", "Scatter (markers)", "Line + Markers"])
    line_shape = st.selectbox("Line shape", ["linear", "spline"])
    lw = st.slider("Line width", 0.5, 6.0, 2.0, 0.5)
    ms = st.slider("Marker size", 2, 16, 6, 1)
    show_legend = st.checkbox("Show legend", value=True)
    hovermode = st.selectbox("Hover mode", ["x unified", "closest", "x", "y"], index=0)
    st.markdown("---")
    st.subheader("Colors")
    color_mode = st.radio("Series colors", ["Auto", "Custom – pick per series"], key="_color_mode")

# Early exit if no sources
chart_container = st.container()
if not st.session_state.get("_sources"):
    with chart_container: st.info("Add one or more data sources in the sidebar to begin.")
    st.stop()

# Load active source (read from disk for path-kind)
meta = st.session_state["_sources"][st.session_state["_active_file"]]
try:
    if meta.get("kind") == "path" and meta.get("path"):
        with open(meta["path"], "rb") as fh:
            file_bytes = fh.read()
        # cache bytes so export works even after dir change
        st.session_state["_sources"][st.session_state["_active_file"]]["bytes"] = file_bytes
    else:
        file_bytes = meta.get("bytes")
    df = load_hdh_bytes(file_bytes)
except Exception as e:
    with chart_container:
        st.error(f"Could not load file '{st.session_state['_active_file']}': {e}")
    st.stop()

if st.session_state.get('_hdh_normalized_nl'):
    st.caption("Note: The upload contained literal '\\n' sequences. I converted them to real newlines so rows parse correctly.")

# Left/right layout
left, right = st.columns([1,1])
with left:
    st.markdown("## Choose Axes & Series")
    all_cols = list(df.columns)
    default_x_index = 1 if len(all_cols) > 1 else 0
    x_col = st.selectbox("X axis", options=all_cols, index=default_x_index, key="x_col")

    default_y = [c for i,c in enumerate(all_cols) if i != 0 and c != x_col][:3]
    y_cols = st.multiselect("Y axis (one or many)", options=all_cols,
                            default=st.session_state.get("y_cols", default_y), key="y_cols")

    secondary_on = st.multiselect("Send these series to secondary Y axis",
                                  options=st.session_state.get("y_cols", []),
                                  default=[c for c in st.session_state.get("secondary_on", []) if c in st.session_state.get("y_cols", [])],
                                  key="secondary_on")

with right:
    st.markdown("## X & Y Range Options")
    prim_cols = [y for y in st.session_state.get("y_cols", []) if y not in st.session_state.get("secondary_on", [])]
    sec_cols  = [y for y in st.session_state.get("y_cols", []) if y in  st.session_state.get("secondary_on", [])]

    x_title_val  = title_value("x_title", st.session_state.get("x_col", ""))
    y1_title_val = title_value("y1_title", ", ".join(prim_cols) if prim_cols else "")
    y2_title_val = title_value("y2_title", ", ".join(sec_cols) if sec_cols else "")

    c1, c2 = st.columns(2)
    with c1:
        chart_title = st.text_input("Chart title", value=st.session_state.get("chart_title",""), key="chart_title")
        x_title = st.text_input("X axis title", value=x_title_val, key="x_title")
    with c2:
        y1_title = st.text_input("Primary Y axis title", value=y1_title_val, key="y1_title")
        y2_title = st.text_input("Secondary Y axis title (if used)", value=y2_title_val, key="y2_title")

    def infer_is_datetime(series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series): return True
        if pd.api.types.is_numeric_dtype(series): return False
        try:
            parsed = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
            return parsed.notna().mean() > 0.8
        except Exception:
            return False

    x_series = df[st.session_state["x_col"]]
    x_is_dt = infer_is_datetime(x_series)
    if x_is_dt and not pd.api.types.is_datetime64_any_dtype(x_series):
        df[st.session_state["x_col"]] = pd.to_datetime(x_series, errors="coerce", infer_datetime_format=True)
        x_series = df[st.session_state["x_col"]]

    r1, r2 = st.columns(2)
    with r1:
        st.subheader("X Range")
        x_auto = st.checkbox("Auto X range", value=st.session_state.get("x_auto", True), key="x_auto")
        reverse_x = st.checkbox("Reverse X axis", value=st.session_state.get("x_rev", False), key="x_rev")
        if not x_auto:
            if x_is_dt:
                xmin = st.datetime_input("X min", value=pd.to_datetime(x_series.min()), key="xmin")
                xmax = st.datetime_input("X max", value=pd.to_datetime(x_series.max()), key="xmax")
            else:
                xmin = st.number_input("X min", value=float(np.nanmin(x_series.values)), format="%.6g", key="xmin")
                xmax = st.number_input("X max", value=float(np.nanmax(x_series.values)), format="%.6g", key="xmax")
        else:
            xmin = xmax = None
    with r2:
        st.subheader("Y Ranges")
        y1_auto = st.checkbox("Auto primary Y range", value=st.session_state.get("y1_auto", True), key="y1_auto")
        reverse_y1 = st.checkbox("Reverse Primary Y axis", value=st.session_state.get("y1_rev", False), key="y1_rev")
        if not y1_auto:
            prim_df = df[prim_cols] if prim_cols else df[st.session_state.get("y_cols", [])]
            y1min = st.number_input("Y1 min", value=float(np.nanmin(prim_df.values)), format="%.6g", key="y1min")
            y1max = st.number_input("Y1 max", value=float(np.nanmax(prim_df.values)), format="%.6g", key="y1max")
        else:
            y1min = y1max = None
        if len(sec_cols) > 0:
            y2_auto = st.checkbox("Auto secondary Y range", value=st.session_state.get("y2_auto", True), key="y2_auto")
            reverse_y2 = st.checkbox("Reverse Secondary Y axis", value=st.session_state.get("y2_rev", False), key="y2_rev")
            if not y2_auto:
                y2vals = df[sec_cols].values
                y2min = st.number_input("Y2 min", value=float(np.nanmin(y2vals)), format="%.6g", key="y2min")
                y2max = st.number_input("Y2 max", value=float(np.nanmax(y2vals)), format="%.6g", key="y2max")

# Color pickers horizontally
if st.session_state.get("_color_mode") == "Custom – pick per series":
    chosen_y = st.session_state.get("y_cols", [])
    if chosen_y:
        st.sidebar.caption("Pick colors (rows wrap horizontally):")
        cols_per_row = 4
        rows = (len(chosen_y)+cols_per_row-1)//cols_per_row
        for r in range(rows):
            row_items = chosen_y[r*cols_per_row:(r+1)*cols_per_row]
            if not row_items: continue
            row_cols = st.sidebar.columns(len(row_items))
            for i, y in enumerate(row_items):
                with row_cols[i]:
                    default_color = FT_COLORS[(r*cols_per_row+i)%len(FT_COLORS)]
                    picked = st.color_picker(y, value=st.session_state.get(f"color_{y}", default_color), key=f"color_{y}")
                    st.session_state.setdefault("selected_colors", {})
                    st.session_state["selected_colors"][y] = picked
else:
    st.session_state["selected_colors"] = {}

# Build chart
fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
mode = {"Line":"lines","Scatter (markers)":"markers","Line + Markers":"lines+markers"}[plot_style]
per_series_colors = st.session_state.get("selected_colors", {})
# lw = st.session_state.get("lw", 2.0)
# ms = st.session_state.get("ms", 6)

for i, y in enumerate(st.session_state.get("y_cols", [])):
    sec = y in st.session_state.get("secondary_on", [])
    c = per_series_colors.get(y) or FT_COLORS[i % len(FT_COLORS)]
    fig.add_trace(go.Scatter(x=df[st.session_state["x_col"]], y=df[y], name=y, mode=mode,
                             line=dict(width=lw, shape=line_shape, color=c),
                             marker=dict(size=ms, color=c),
                             hovertemplate=f"<b>{y}</b><br>{st.session_state['x_col']}: %{{x}}<br>value: %{{y}}<extra></extra>"),
                  secondary_y=sec)

fig.update_layout(template=FT_TEMPLATE, showlegend=show_legend, hovermode=hovermode)

# Axes
fig.update_xaxes(title_text=st.session_state.get("x_title",""))
if not st.session_state.get("x_auto", True) and st.session_state.get("xmin") is not None and st.session_state.get("xmax") is not None:
    xrng=[st.session_state["xmin"], st.session_state["xmax"]]
    if st.session_state.get("x_rev", False): xrng=xrng[::-1]
    fig.update_xaxes(range=xrng)
elif st.session_state.get("x_rev", False):
    fig.update_xaxes(autorange="reversed")
fig.update_yaxes(title_text=st.session_state.get("y1_title",""), secondary_y=False)
if not st.session_state.get("y1_auto", True) and (st.session_state.get("y1min") is not None) and (st.session_state.get("y1max") is not None):
    yrng=[st.session_state["y1min"], st.session_state["y1max"]]
    if st.session_state.get("y1_rev", False): yrng=yrng[::-1]
    fig.update_yaxes(range=yrng, secondary_y=False)
elif st.session_state.get("y1_rev", False):
    fig.update_yaxes(autorange="reversed", secondary_y=False)
if len(st.session_state.get("secondary_on", []))>0:
    fig.update_yaxes(title_text=st.session_state.get("y2_title",""), secondary_y=True)
    if not (st.session_state.get("y2_auto", True) or (st.session_state.get("y2min") is None or st.session_state.get("y2max") is None)):
        y2rng=[st.session_state["y2min"], st.session_state["y2max"]]
        if st.session_state.get("y2_rev", False): y2rng=y2rng[::-1]
        fig.update_yaxes(range=y2rng, secondary_y=True)
    elif st.session_state.get("y2_rev", False):
        fig.update_yaxes(autorange="reversed", secondary_y=True)

fig.update_layout(title=dict(text=st.session_state.get("chart_title",""), x=0.0, xanchor="left"))

# Download
exp = df[[st.session_state["x_col"]] + st.session_state.get("y_cols", [])].copy()
if not st.session_state.get("x_auto", True) and st.session_state.get("xmin") is not None and st.session_state.get("xmax") is not None:
    try: mask = (exp[st.session_state["x_col"]] >= st.session_state["xmin"]) & (exp[st.session_state["x_col"]] <= st.session_state["xmax"])
    except Exception: mask = (exp[st.session_state["x_col"]].astype(float) >= float(st.session_state["xmin"])) & (exp[st.session_state["x_col"]].astype(float) <= float(st.session_state["xmax"]))
    exp = exp.loc[mask]
prim_cols = [y for y in st.session_state.get("y_cols", []) if y not in st.session_state.get("secondary_on", [])]
sec_cols = [y for y in st.session_state.get("y_cols", []) if y in st.session_state.get("secondary_on", [])]
if not st.session_state.get("y1_auto", True) and prim_cols:
    pvals = exp[prim_cols]; maskp = (pvals >= st.session_state.get("y1min", -np.inf)).all(axis=1) & (pvals <= st.session_state.get("y1max", np.inf)).all(axis=1)
    exp = exp.loc[maskp]
if len(sec_cols)>0 and (not (st.session_state.get("y2_auto", True) or (st.session_state.get("y2min") is None or st.session_state.get("y2max") is None))):
    svals = exp[sec_cols]; masks = (svals >= st.session_state["y2min"]).all(axis=1) & (svals <= st.session_state["y2max"]).all(axis=1)
    exp = exp.loc[masks]
csv_bytes = exp.to_csv(index=False).encode("utf-8")

with chart_container:
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.download_button("Download plotted data as CSV", data=csv_bytes, file_name="plotted_data.csv", mime="text/csv")

st.markdown("## Data Preview")
st.dataframe(df.head(20), use_container_width=True)

