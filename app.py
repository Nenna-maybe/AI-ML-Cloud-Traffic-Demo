"""
app.py - Streamlit user interface.

Run locally with:   streamlit run app.py

This file only handles the user interface. All logic lives in separate
modules (detector, traffic, fusion, incident_manager, database, ...).
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from config import BASE_DIR, load_settings, load_streamlit_secrets

st.set_page_config(
    page_title="Traffic Congestion Monitoring",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_streamlit_secrets()  # must run before load_settings()
SETTINGS = load_settings()
SETTINGS.ensure_dirs()

import analytics  # noqa: E402  (imported after settings are ready)
import database as db  # noqa: E402
import incident_manager as im  # noqa: E402
from cloud_storage import StorageManager  # noqa: E402
from detector import (  # noqa: E402
    DetectionError, ModelLoadError, ValidationError, VehicleDetector,
    detections_table, encode_png, limit_image_size, save_image, validate_image_upload,
)
from fusion import FusionInput, PRIORITY_CATEGORIES, compute_priority  # noqa: E402
from traffic import CATEGORIES as CONGESTION_CATEGORIES, estimate_congestion  # noqa: E402
from video_processor import analyze_video, validate_video_upload, write_temp_video  # noqa: E402

logging.basicConfig(
    level=getattr(logging, SETTINGS.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("traffic_app")

DB = SETTINGS.db_path
db.init_db(DB)

PAGES = [
    "Dashboard",
    "Image Detection",
    "Video Analysis",
    "Congestion Analysis",
    "Incidents",
    "Priority Engine",
    "History",
    "Analytics & Reports",
    "Cloud & Settings",
]

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .app-header {background: linear-gradient(90deg, #12263f, #1f4e79); color: #fff;
                 padding: 18px 22px; border-radius: 10px; margin-bottom: 18px;}
    .app-header h1 {font-size: 1.6rem; margin: 0; color: #fff;}
    .app-header p {margin: 4px 0 0 0; opacity: .85; font-size: .95rem;}
    .badge {display: inline-block; padding: 2px 9px; border-radius: 10px; font-size: .78rem;
            font-weight: 600; color: #fff; margin: 2px 4px 2px 0;}
    .b-ai {background:#2e7d32;} .b-rule {background:#1565c0;} .b-manual {background:#ef6c00;}
    .b-opt {background:#6d6d6d;} .b-sample {background:#8e24aa;}
    .priority-card {padding: 16px 20px; border-radius: 10px; color: #fff; margin-bottom: 10px;}
    div[data-testid="stMetric"] {background: rgba(31,78,121,.07); border: 1px solid rgba(31,78,121,.18);
            border-radius: 10px; padding: 10px 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def header(title: str, subtitle: str, badges: str = "") -> None:
    st.markdown(
        f'<div class="app-header"><h1>{title}</h1><p>{subtitle}</p>{badges}</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, kind: str) -> str:
    return f'<span class="badge b-{kind}">{text}</span>'


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO model (first run downloads ~6 MB)...")
def get_detector(model_path: str) -> VehicleDetector:
    detector = VehicleDetector(model_path)
    detector.load()
    return detector


@st.cache_resource
def get_storage(use_s3: bool, bucket: str, prefix: str, region: str) -> StorageManager:
    return StorageManager(use_s3=use_s3, bucket=bucket, prefix=prefix, region=region)


STORAGE = get_storage(SETTINGS.use_s3, SETTINGS.s3_bucket, SETTINGS.s3_prefix, SETTINGS.aws_region)


def load_detector() -> VehicleDetector | None:
    try:
        return get_detector(str(SETTINGS.model_path))
    except ModelLoadError as exc:
        st.error(str(exc))
        db.log_event(DB, "ERROR", "model_load_failed", str(exc))
        return None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def rel_path(path: Path) -> str:
    """Store paths relative to the project so the database is portable."""
    try:
        return str(Path(path).resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def abs_path(stored: str | None) -> Path | None:
    if not stored or stored.startswith("s3://"):
        return None
    path = Path(stored)
    return path if path.is_absolute() else BASE_DIR / path


def unique_name(original: str, suffix: str) -> str:
    stem = "".join(c for c in Path(original).stem if c.isalnum() or c in "-_")[:40] or "upload"
    return f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}_{stem}{suffix}"


def date_filters(key: str) -> tuple[str | None, str | None]:
    c1, c2 = st.columns(2)
    start = c1.date_input("From date", value=None, key=f"{key}_from")
    end = c2.date_input("To date", value=None, key=f"{key}_to")
    if start and end and start > end:
        st.warning("'From date' is after 'To date' - no records will match.")
    return (start.isoformat() if isinstance(start, date) else None,
            end.isoformat() if isinstance(end, date) else None)


def go_to(page: str, **state) -> None:
    st.session_state["page"] = page
    for key, value in state.items():
        st.session_state[key] = value


def show_chart(fig, empty_message: str) -> None:
    if fig is None:
        st.info(empty_message)
    else:
        st.plotly_chart(fig, width="stretch")


def session_label(row: dict) -> str:
    tag = "[SAMPLE] " if row.get("is_sample") else ""
    name = row.get("file_name") or row.get("location") or ""
    return (f"{tag}#{row['id']} · {row['created_at'][:16].replace('T', ' ')} · {row['source_type']} · "
            f"{name} · {row['total_vehicles']} vehicles · {row.get('congestion_category') or '-'}")


def incident_label(row: dict) -> str:
    tag = "[SAMPLE] " if row.get("is_sample") else ""
    return (f"{tag}#{row['id']} · {row['incident_type']} · {row['location']} · "
            f"{row['severity']} · {row['status']}")


PRIORITY_COLORS = {"Low": "#2e9e5b", "Medium": "#c99a00", "High": "#e8730c", "Critical": "#d62828"}


def show_priority(result) -> None:
    color = PRIORITY_COLORS[result.priority_category]
    st.markdown(
        f'<div class="priority-card" style="background:{color}"><h3 style="margin:0;color:#fff">'
        f"Priority: {result.priority_category} — {result.priority_score:g}/100</h3>"
        f"<p style='margin:6px 0 0 0'>Recommended action: {result.recommended_action}</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Why this priority was assigned:**")
    for reason in result.reasons:
        st.markdown(f"- {reason}")


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = PAGES[0]

with st.sidebar:
    st.markdown("## 🚦 Traffic Monitor")
    st.caption("Intelligent Traffic Congestion Monitoring & Incident Prioritization")
    st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    INCLUDE_SAMPLE = st.toggle(
        "Include SAMPLE data in views", value=False,
        help="Sample records are generated demo data, clearly marked and never mixed with real results.",
    )
    st.markdown("**What is real in this app?**")
    st.markdown(
        badge("Real AI", "ai") + " YOLO vehicle detection<br>"
        + badge("Rule-based", "rule") + " congestion & priority<br>"
        + badge("Manual", "manual") + " incident reports<br>"
        + badge("Optional", "opt") + f" AWS S3 ({'ON' if STORAGE.enabled else 'OFF'})",
        unsafe_allow_html=True,
    )

page = st.session_state["page"]


# ==========================================================================
# Page: Dashboard
# ==========================================================================
def page_dashboard() -> None:
    header(
        "Intelligent Traffic Congestion Monitoring",
        "AI vehicle detection · estimated congestion · incident prioritization · cloud-ready",
        badge("SAMPLE data included", "sample") if INCLUDE_SAMPLE else "",
    )
    stats = db.get_dashboard_stats(DB, include_sample=INCLUDE_SAMPLE)
    c = st.columns(5)
    c[0].metric("Detection sessions", stats["total_sessions"])
    c[1].metric("Vehicles detected", stats["total_vehicles"],
                help="Images: vehicles found. Videos: peak vehicles visible in one frame.")
    c[2].metric("Incidents reported", stats["total_incidents"], f"{stats['open_incidents']} not resolved",
                delta_color="off")
    c[3].metric("High/Severe congestion", stats["high_congestion"])
    c[4].metric("Priority assessments", stats["fusion_results"])

    if stats["total_sessions"] == 0 and stats["total_incidents"] == 0:
        st.info("No data yet. Start with **Image Detection**, report an **Incident**, or load SAMPLE "
                "data from **Cloud & Settings** to explore the dashboard.")

    sessions = analytics.sessions_df(DB, include_sample=INCLUDE_SAMPLE)
    fusion = analytics.fusion_df(DB, include_sample=INCLUDE_SAMPLE)
    col1, col2, col3 = st.columns(3)
    with col1:
        show_chart(analytics.congestion_distribution(sessions), "No congestion estimates yet.")
    with col2:
        show_chart(analytics.priority_distribution(fusion), "No priority assessments yet.")
    with col3:
        show_chart(analytics.vehicle_class_chart(analytics.class_totals_df(DB, INCLUDE_SAMPLE)),
                   "No vehicle counts yet.")

    st.subheader("Recent detection activity")
    if sessions.empty:
        st.caption("Nothing to show yet.")
    else:
        recent = analytics.label_sample(sessions.head(8))
        cols = [c for c in ["id", "data", "created_at", "source_type", "file_name", "location",
                            "total_vehicles", "congestion_score", "congestion_category"] if c in recent]
        st.dataframe(recent[cols], hide_index=True, width="stretch")

    st.subheader("Quick navigation")
    nav = st.columns(4)
    for i, target in enumerate(["Image Detection", "Video Analysis", "Incidents", "Priority Engine"]):
        nav[i].button(target, on_click=go_to, args=(target,), width="stretch")


# ==========================================================================
# Page: Image Detection
# ==========================================================================
def page_image_detection() -> None:
    header("AI Vehicle Detection", "Upload a traffic image. A pretrained YOLO model detects cars, "
           "motorcycles, buses and trucks.", badge("Real AI inference", "ai"))
    c1, c2 = st.columns([2, 1])
    upload = c1.file_uploader("Traffic image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    location = c2.text_input("Camera / road name (optional)", max_chars=120)
    confidence = c2.slider("Confidence threshold", 0.05, 0.95, float(SETTINGS.default_confidence), 0.05,
                           help="Detections below this confidence are ignored.")
    c2.caption(f"Model: `{SETTINGS.yolo_model}` · max {SETTINGS.max_image_mb:g} MB")

    if st.button("Run detection", type="primary", disabled=upload is None):
        try:
            data = upload.getvalue()
            image = validate_image_upload(upload.name, data, SETTINGS.max_image_mb)
            image = limit_image_size(image, SETTINGS.max_image_side)
        except ValidationError as exc:
            st.error(str(exc))
            return
        detector = load_detector()
        if detector is None:
            return
        import numpy as np

        try:
            with st.spinner("Running YOLO inference..."):
                result = detector.detect(np.array(image), confidence)
        except (DetectionError, ValidationError, ValueError) as exc:
            st.error(f"Detection failed: {exc}")
            db.log_event(DB, "ERROR", "detection_failed", str(exc))
            return

        congestion = estimate_congestion(result.counts, SETTINGS.congestion)
        # Save original and annotated images to disk (and S3 if enabled)
        original_path = SETTINGS.upload_dir / unique_name(upload.name, ".png")
        annotated_path = SETTINGS.output_dir / "annotated" / unique_name(upload.name, "_annotated.png")
        image.save(original_path)
        save_image(result.annotated_rgb, annotated_path)
        stored_original = STORAGE.save(original_path, f"uploads/{original_path.name}")
        stored_annotated = STORAGE.save(annotated_path, f"annotated/{annotated_path.name}")
        for stored in (stored_original, stored_annotated):
            if stored.warning:
                st.warning(stored.warning)

        session_id = db.insert_detection_session(
            DB, source_type="image", counts=result.counts, total_vehicles=result.total,
            congestion_score=congestion.score, congestion_category=congestion.category,
            file_name=upload.name[:120], location=im.clean_text(location) or None,
            stored_path=rel_path(original_path) if stored_original.backend == "local" else stored_original.location,
            annotated_path=rel_path(annotated_path),
            storage_backend=stored_annotated.backend, model_name=result.model_name,
            confidence_threshold=confidence,
        )
        db.log_event(DB, "INFO", "image_detection", f"session={session_id} vehicles={result.total}")
        st.session_state["last_image"] = {
            "session_id": session_id, "original": np.array(image), "result": result,
            "congestion": congestion, "file_name": upload.name,
        }

    last = st.session_state.get("last_image")
    if not last:
        st.info("Upload an image and click **Run detection**.")
        return

    result, congestion = last["result"], last["congestion"]
    st.success(f"Saved as detection session #{last['session_id']}.")
    col1, col2 = st.columns(2)
    col1.image(last["original"], caption="Uploaded image", width="stretch")
    col2.image(result.annotated_rgb, caption="YOLO detections", width="stretch")
    col2.download_button(
        "Download annotated image", data=encode_png(result.annotated_rgb),
        file_name=f"annotated_session_{last['session_id']}.png", mime="image/png",
    )

    m = st.columns(5)
    m[0].metric("Total vehicles", result.total)
    plural = {"car": "Cars", "motorcycle": "Motorcycles", "bus": "Buses", "truck": "Trucks"}
    for i, cls in enumerate(plural):
        m[i + 1].metric(plural[cls], result.counts.get(cls, 0))

    left, right = st.columns(2)
    with left:
        st.plotly_chart(analytics.congestion_gauge(congestion.score, congestion.category), width="stretch")
        st.caption(congestion.explanation)
    with right:
        rows = detections_table(result)
        if rows:
            st.markdown("**Detections and confidence scores**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=280)
        else:
            st.warning("No vehicles detected at this confidence threshold. Try lowering it.")
    st.button("Use this result in the Priority Engine →", on_click=go_to,
              args=("Priority Engine",), kwargs={"fusion_session": last["session_id"]})


# ==========================================================================
# Page: Video Analysis
# ==========================================================================
def page_video() -> None:
    header("Video-Based Traffic Analysis", "Samples frames from a short video and runs YOLO on each "
           "sampled frame.", badge("Real AI inference", "ai") + badge("Frame sampling", "rule"))
    with st.expander("Limitations of this analysis", expanded=False):
        st.markdown(
            "- Only a sample of frames is analysed, so short events between samples can be missed.\n"
            "- There is **no tracking**: the same vehicle appearing in several frames is counted in each "
            "frame. Results mean *vehicles visible per frame*, not unique vehicles that passed.\n"
            "- CPU inference is slow on free hosting; keep videos short "
            f"(≤ {SETTINGS.max_video_seconds:g} s, ≤ {SETTINGS.max_video_mb:g} MB).\n"
            "- The video file itself is not kept after analysis; only the summary and the peak "
            "annotated frame are saved."
        )
    c1, c2 = st.columns([2, 1])
    upload = c1.file_uploader("Traffic video (MP4, MOV, AVI, MKV)", type=["mp4", "mov", "avi", "mkv"])
    location = c2.text_input("Camera / road name (optional)", max_chars=120, key="video_location")
    max_frames = c2.slider("Frames to analyse", 1, max(1, SETTINGS.max_video_frames),
                           min(10, SETTINGS.max_video_frames))
    confidence = c2.slider("Confidence threshold", 0.05, 0.95, float(SETTINGS.default_confidence), 0.05,
                           key="video_conf")

    if st.button("Analyse video", type="primary", disabled=upload is None):
        try:
            validate_video_upload(upload.name, upload.size, SETTINGS.max_video_mb)
        except ValidationError as exc:
            st.error(str(exc))
            return
        detector = load_detector()
        if detector is None:
            return
        temp_path = write_temp_video(upload, Path(upload.name).suffix.lower(), SETTINGS.upload_dir)
        bar = st.progress(0.0, text="Starting...")
        try:
            analysis = analyze_video(
                temp_path, detector, confidence=confidence, max_frames=max_frames,
                max_seconds=SETTINGS.max_video_seconds,
                progress=lambda done, total: bar.progress(done / total, text=f"Processed frame {done}/{total}"),
            )
        except ValidationError as exc:
            st.error(str(exc))
            return
        except (DetectionError, ValueError) as exc:
            st.error(f"Video analysis failed: {exc}")
            db.log_event(DB, "ERROR", "video_failed", str(exc))
            return
        finally:
            temp_path.unlink(missing_ok=True)  # never leave temporary uploads behind
        bar.empty()

        congestion = estimate_congestion(analysis.average_counts, SETTINGS.congestion)
        peak_image = analysis.sample_frames[-1][1]
        peak_path = SETTINGS.output_dir / "annotated" / unique_name(upload.name, "_peak.png")
        save_image(peak_image, peak_path)
        stored = STORAGE.save(peak_path, f"annotated/{peak_path.name}")
        if stored.warning:
            st.warning(stored.warning)
        session_id = db.insert_detection_session(
            DB, source_type="video", counts=analysis.average_counts, total_vehicles=analysis.peak_total,
            congestion_score=congestion.score, congestion_category=congestion.category,
            file_name=upload.name[:120], location=im.clean_text(location) or None,
            annotated_path=rel_path(peak_path), storage_backend=stored.backend,
            model_name=detector.model_name, confidence_threshold=confidence,
            frames_processed=analysis.frames_processed, avg_vehicles=analysis.average_total,
            peak_vehicles=analysis.peak_total,
        )
        db.log_event(DB, "INFO", "video_analysis", f"session={session_id} frames={analysis.frames_processed}")
        st.session_state["last_video"] = {"session_id": session_id, "analysis": analysis, "congestion": congestion}

    last = st.session_state.get("last_video")
    if not last:
        st.info("Upload a short video and click **Analyse video**.")
        return
    analysis, congestion = last["analysis"], last["congestion"]
    st.success(f"Saved as detection session #{last['session_id']}.")
    m = st.columns(5)
    m[0].metric("Duration", f"{analysis.duration_s:g} s")
    m[1].metric("Frames analysed", analysis.frames_processed)
    m[2].metric("Average vehicles / frame", analysis.average_total)
    m[3].metric("Peak vehicles in a frame", analysis.peak_total)
    m[4].metric("Est. congestion", f"{congestion.score:g} ({congestion.category})")

    frame_df = pd.DataFrame([{"time (s)": f.timestamp_s, "vehicles": f.total, **f.counts} for f in analysis.frames])
    import plotly.express as px

    fig = px.line(frame_df, x="time (s)", y="vehicles", markers=True, title="Vehicles per sampled frame")
    st.plotly_chart(fig, width="stretch")
    st.caption("Congestion uses the *average* per-class counts across sampled frames. " + congestion.explanation)

    st.markdown("**Annotated sample frames**")
    cols = st.columns(len(analysis.sample_frames))
    for col, (label, frame) in zip(cols, analysis.sample_frames):
        col.image(frame, caption=label, width="stretch")
    with st.expander("Per-frame counts"):
        st.dataframe(frame_df, hide_index=True, width="stretch")
    st.button("Use this result in the Priority Engine →", on_click=go_to,
              args=("Priority Engine",), kwargs={"fusion_session": last["session_id"]}, key="video_to_fusion")


# ==========================================================================
# Page: Congestion Analysis
# ==========================================================================
def page_congestion() -> None:
    header("Traffic Congestion Analysis", "Estimated congestion indicator based on observed vehicle "
           "counts — not a calibrated traffic-density measurement.", badge("Rule-based", "rule"))
    cfg = SETTINGS.congestion
    with st.expander("How the score is calculated", expanded=True):
        weights = ", ".join(f"{k} = {v:g}" for k, v in cfg.vehicle_weights.items())
        st.markdown(
            f"1. Convert vehicles to **vehicle units** (bigger vehicles use more road space): {weights}.\n"
            f"2. **Score = min(100, vehicle units ÷ {cfg.reference_capacity:g} × 100)** — "
            f"{cfg.reference_capacity:g} units is treated as a full camera view.\n"
            f"3. **Category:** Low < {cfg.moderate_threshold:g} ≤ Moderate < {cfg.high_threshold:g} ≤ High < "
            f"{cfg.severe_threshold:g} ≤ Severe.\n\n"
            "Thresholds are configurable through `.env` (see `.env.example`). A single image cannot show "
            "speed, queue length or road capacity, so treat this as an *indicator*."
        )

    sessions = db.list_detection_sessions(DB, include_sample=INCLUDE_SAMPLE)
    if not sessions:
        st.info("No detection results yet. Run an image or video detection first.")
    else:
        options = {session_label(s): s["id"] for s in sessions}
        choice = st.selectbox("Choose a stored detection result", list(options))
        session = db.get_detection_session(DB, options[choice])
        current = estimate_congestion(session["counts"], cfg)
        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(analytics.congestion_gauge(current.score, current.category), width="stretch")
            if session["congestion_score"] is not None and abs(session["congestion_score"] - current.score) > 0.01:
                st.warning(f"Stored score was {session['congestion_score']:g} ({session['congestion_category']}). "
                           "It differs because the thresholds/capacity settings changed since it was saved.")
            st.caption(current.explanation)
        with right:
            rows = [{"class": k, "count": v, "weight": cfg.vehicle_weights.get(k, 1.0),
                     "vehicle units": round(v * cfg.vehicle_weights.get(k, 1.0), 2)}
                    for k, v in session["counts"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            show_chart(analytics.counts_bar(session["counts"], "Counts by vehicle class"), "No counts.")
            if session["source_type"] == "video":
                st.caption("Video counts are averages per sampled frame.")

    with st.expander("Try the formula with your own numbers (calculator — not saved)"):
        c = st.columns(4)
        labels = {"car": "Cars", "motorcycle": "Motorcycles", "bus": "Buses", "truck": "Trucks"}
        manual = {cls: c[i].number_input(labels[cls], 0, 500, 0, key=f"calc_{cls}")
                  for i, cls in enumerate(labels)}
        calc = estimate_congestion(manual, cfg)
        st.metric("Calculated score", f"{calc.score:g}", calc.category, delta_color="off")


# ==========================================================================
# Page: Incidents
# ==========================================================================
def page_incidents() -> None:
    header("Incident Management", "Manually reported incidents. The vision model does NOT detect "
           "accidents automatically.", badge("Manual reports", "manual"))
    tab_new, tab_list, tab_edit = st.tabs(["Report incident", "Incident list", "View / edit"])

    with tab_new:
        with st.form("new_incident", clear_on_submit=True):
            c1, c2 = st.columns(2)
            itype = c1.selectbox("Incident type", im.INCIDENT_TYPES)
            severity = c2.selectbox("Severity", im.SEVERITIES, index=1)
            location = c1.text_input("Location / road name *", max_chars=im.MAX_LOCATION_LEN)
            status = c2.selectbox("Status", im.STATUSES)
            description = st.text_area("Description", max_chars=im.MAX_DESCRIPTION_LEN,
                                       help="Do not enter personal information (names, phone numbers, plates).")
            if st.form_submit_button("Save incident", type="primary"):
                try:
                    new_id = im.create_incident(DB, {"incident_type": itype, "severity": severity,
                                                     "location": location, "status": status,
                                                     "description": description})
                    st.success(f"Incident #{new_id} saved.")
                except im.IncidentValidationError as exc:
                    for err in exc.errors:
                        st.error(err)

    with tab_list:
        c = st.columns(3)
        f_type = c[0].selectbox("Type", ["All"] + im.INCIDENT_TYPES)
        f_sev = c[1].selectbox("Severity", ["All"] + im.SEVERITIES)
        f_status = c[2].selectbox("Status", ["All"] + im.STATUSES)
        start, end = date_filters("inc")
        df = analytics.incidents_df(
            DB, include_sample=INCLUDE_SAMPLE,
            incident_type=None if f_type == "All" else f_type,
            severity=None if f_sev == "All" else f_sev,
            status=None if f_status == "All" else f_status,
            date_from=start, date_to=end,
        )
        if df.empty:
            st.info("No incidents match these filters.")
        else:
            df = analytics.label_sample(df)
            st.dataframe(df.drop(columns=["is_sample"]), hide_index=True, width="stretch")
            st.download_button("Export these incidents (CSV)", analytics.to_csv_bytes(df),
                               "incidents.csv", "text/csv")

    with tab_edit:
        rows = db.list_incidents(DB, include_sample=INCLUDE_SAMPLE)
        if not rows:
            st.info("No incidents yet.")
            return
        labels = {incident_label(r): r["id"] for r in rows}
        default = st.session_state.get("edit_incident")
        ids = list(labels.values())
        index = ids.index(default) if default in ids else 0
        incident = db.get_incident(DB, labels[st.selectbox("Choose incident", list(labels), index=index)])
        st.json({k: v for k, v in incident.items() if k != "is_sample"}, expanded=True)

        authorized = True
        if SETTINGS.admin_pin:
            pin = st.text_input("Admin PIN (required to edit)", type="password")
            authorized = im.is_authorized(pin, SETTINGS.admin_pin)
            if pin and not authorized:
                st.error("Incorrect PIN.")
        else:
            st.caption("Editing is open because ADMIN_PIN is not set (local demo mode).")

        with st.form(f"edit_{incident['id']}"):
            c1, c2 = st.columns(2)
            e_type = c1.selectbox("Incident type", im.INCIDENT_TYPES, im.INCIDENT_TYPES.index(incident["incident_type"]))
            e_sev = c2.selectbox("Severity", im.SEVERITIES, im.SEVERITIES.index(incident["severity"]))
            e_loc = c1.text_input("Location / road name", incident["location"], max_chars=im.MAX_LOCATION_LEN)
            e_status = c2.selectbox("Status", im.STATUSES, im.STATUSES.index(incident["status"]))
            e_desc = st.text_area("Description", incident["description"] or "", max_chars=im.MAX_DESCRIPTION_LEN)
            save = st.form_submit_button("Save changes", type="primary", disabled=not authorized)
        if save:
            try:
                im.edit_incident(DB, incident["id"], {"incident_type": e_type, "severity": e_sev,
                                                      "location": e_loc, "status": e_status,
                                                      "description": e_desc})
                st.session_state["edit_incident"] = incident["id"]
                st.success("Incident updated.")
                st.rerun()
            except (im.IncidentValidationError, LookupError) as exc:
                st.error(str(exc))
        if incident["status"] != "Resolved" and st.button("Mark as Resolved", disabled=not authorized):
            im.set_status(DB, incident["id"], "Resolved")
            st.session_state["edit_incident"] = incident["id"]
            st.rerun()


# ==========================================================================
# Page: Priority Engine
# ==========================================================================
def page_priority() -> None:
    header("Adaptive Cross-Signal Fusion Engine",
           "Combines estimated congestion with incident severity, status and recency into a priority.",
           badge("Rule-based decision support (not a trained ML model)", "rule"))
    f = SETTINGS.fusion
    with st.expander("How the priority score is calculated"):
        st.markdown(
            f"- **Congestion component C** = estimated congestion score (0–100).\n"
            f"- **Incident component I** = severity points (Low 25, Medium 50, High 75, Critical 100) × "
            f"status factor (Open 1.0, In Progress 0.7, Resolved 0) × recency factor (1.0 for the first "
            f"{f.recency_full_hours:g} h, decaying to {f.recency_floor:g} at {f.recency_decay_hours:g} h).\n"
            f"- **Priority = {f.congestion_weight:g} × C + {f.incident_weight:g} × I** (weights normalised). "
            "If only one signal exists, it is used alone.\n"
            f"- Safety rule: an active Critical incident scores at least {f.critical_incident_floor:g}.\n"
            f"- Categories: Low < {f.medium_threshold:g} ≤ Medium < {f.high_threshold:g} ≤ High < "
            f"{f.critical_threshold:g} ≤ Critical.\n"
            "- Vehicle count is **not** added separately, because the congestion score already comes from it "
            "(that would double-count the same evidence)."
        )

    sessions = db.list_detection_sessions(DB, include_sample=INCLUDE_SAMPLE)
    incidents = db.list_incidents(DB, include_sample=INCLUDE_SAMPLE)
    s_opts = {"— none —": None} | {session_label(s): s["id"] for s in sessions}
    i_opts = {"— none —": None} | {incident_label(i): i["id"] for i in incidents}
    pre = st.session_state.get("fusion_session")
    s_index = list(s_opts.values()).index(pre) if pre in s_opts.values() else 0
    c1, c2 = st.columns(2)
    s_choice = c1.selectbox("Detection result (traffic signal)", list(s_opts), index=s_index)
    i_choice = c2.selectbox("Incident (incident signal)", list(i_opts))
    session_id, incident_id = s_opts[s_choice], i_opts[i_choice]

    if st.button("Compute priority and save", type="primary"):
        if session_id is None and incident_id is None:
            st.error("Select a detection result, an incident, or both.")
            return
        session = db.get_detection_session(DB, session_id) if session_id else None
        incident = db.get_incident(DB, incident_id) if incident_id else None
        data = FusionInput(
            congestion_score=session["congestion_score"] if session else None,
            congestion_category=session["congestion_category"] if session else None,
            vehicle_count=session["total_vehicles"] if session else None,
            incident_severity=incident["severity"] if incident else None,
            incident_status=incident["status"] if incident else None,
            incident_reported_at=incident["created_at"] if incident else None,
            now=db.now_iso(),
        )
        try:
            result = compute_priority(data, f)
        except ValueError as exc:
            st.error(str(exc))
            return
        is_sample = bool((session and session["is_sample"]) or (incident and incident["is_sample"]))
        fusion_id = db.insert_fusion_result(
            DB, inputs=result.inputs, priority_score=result.priority_score,
            priority_category=result.priority_category, explanation=result.explanation,
            recommended_action=result.recommended_action, session_id=session_id,
            incident_id=incident_id, is_sample=is_sample,
        )
        db.log_event(DB, "INFO", "fusion_computed", f"id={fusion_id} score={result.priority_score}")
        st.session_state["last_fusion"] = (fusion_id, result)

    if "last_fusion" in st.session_state:
        fusion_id, result = st.session_state["last_fusion"]
        st.success(f"Saved as priority assessment #{fusion_id}.")
        show_priority(result)
        with st.expander("Stored inputs (used to reproduce this result)"):
            st.json(result.inputs)


# ==========================================================================
# Page: History
# ==========================================================================
def page_history() -> None:
    header("Detection & Incident History", "All records come from the SQLite database.",
           badge("Database = source of truth", "rule"))
    tab_d, tab_i, tab_f, tab_l = st.tabs(["Detections", "Incidents", "Priority assessments", "Processing logs"])

    with tab_d:
        c = st.columns(2)
        src = c[0].selectbox("Source", ["All", "image", "video", "sample"])
        cat = c[1].selectbox("Congestion category", ["All"] + CONGESTION_CATEGORIES)
        start, end = date_filters("hist_d")
        df = analytics.sessions_df(DB, include_sample=INCLUDE_SAMPLE, source_type=None if src == "All" else src,
                                   category=None if cat == "All" else cat, date_from=start, date_to=end)
        if df.empty:
            st.info("No detection records match.")
        else:
            df = analytics.label_sample(df)
            st.dataframe(df.drop(columns=["is_sample"]), hide_index=True, width="stretch")
            st.download_button("Export detections (CSV)", analytics.to_csv_bytes(df), "detections.csv", "text/csv")
            chosen = st.selectbox("Open record details", df["id"].tolist(), key="hist_detail")
            session = db.get_detection_session(DB, int(chosen))
            d1, d2 = st.columns(2)
            with d1:
                st.json({k: v for k, v in session.items() if k not in {"is_sample"}})
            with d2:
                path = abs_path(session.get("annotated_path"))
                if path and path.exists():
                    st.image(str(path), caption="Annotated output", width="stretch")
                elif session["source_type"] != "sample":
                    st.caption("Annotated image file is not available on this machine "
                               "(hosted storage is temporary).")

    with tab_i:
        start, end = date_filters("hist_i")
        status = st.selectbox("Status", ["All"] + im.STATUSES, key="hist_status")
        df = analytics.incidents_df(DB, include_sample=INCLUDE_SAMPLE, date_from=start, date_to=end,
                                    status=None if status == "All" else status)
        if df.empty:
            st.info("No incidents match.")
        else:
            df = analytics.label_sample(df)
            st.dataframe(df.drop(columns=["is_sample"]), hide_index=True, width="stretch")
            st.download_button("Export incidents (CSV)", analytics.to_csv_bytes(df), "incident_history.csv", "text/csv")
            chosen = st.selectbox("Open incident", df["id"].tolist(), key="hist_inc")
            st.button("Open in incident editor", on_click=go_to, args=("Incidents",),
                      kwargs={"edit_incident": int(chosen)})

    with tab_f:
        cat = st.selectbox("Priority category", ["All"] + PRIORITY_CATEGORIES)
        start, end = date_filters("hist_f")
        df = analytics.fusion_df(DB, include_sample=INCLUDE_SAMPLE, category=None if cat == "All" else cat,
                                 date_from=start, date_to=end)
        if df.empty:
            st.info("No priority assessments match.")
        else:
            df = analytics.label_sample(df)
            st.dataframe(df.drop(columns=["is_sample", "inputs_json"]), hide_index=True, width="stretch")
            st.download_button("Export priority results (CSV)", analytics.to_csv_bytes(df),
                               "priority_results.csv", "text/csv")
            chosen = st.selectbox("Open assessment", df["id"].tolist(), key="hist_fusion")
            row = db.get_fusion_result(DB, int(chosen))
            st.markdown(f"**{row['priority_category']} ({row['priority_score']:g}/100)** — {row['recommended_action']}")
            st.write(row["explanation"])
            inputs = analytics.stored_inputs(row)
            st.json(inputs, expanded=False)
            if st.button("Verify: recompute from stored inputs"):
                again = compute_priority(FusionInput.from_dict(inputs), SETTINGS.fusion)
                if abs(again.priority_score - row["priority_score"]) < 0.01:
                    st.success(f"Reproduced exactly: {again.priority_score:g} ({again.priority_category}).")
                else:
                    st.warning(f"Recomputed {again.priority_score:g} vs stored {row['priority_score']:g}. "
                               "The fusion settings have changed since this result was saved.")

    with tab_l:
        logs = db.list_logs(DB, 200)
        if logs:
            st.dataframe(pd.DataFrame(logs), hide_index=True, width="stretch")
        else:
            st.info("No log entries yet.")


# ==========================================================================
# Page: Analytics & Reports
# ==========================================================================
def page_analytics() -> None:
    header("Analytics & Reports", "Charts and CSV exports built from stored records.",
           badge("SAMPLE data included", "sample") if INCLUDE_SAMPLE else "")
    sessions = analytics.sessions_df(DB, include_sample=INCLUDE_SAMPLE)
    incidents = analytics.incidents_df(DB, include_sample=INCLUDE_SAMPLE)
    fusion = analytics.fusion_df(DB, include_sample=INCLUDE_SAMPLE)
    if sessions.empty and incidents.empty and fusion.empty:
        st.info("No records yet. Add data or load SAMPLE data from Cloud & Settings.")
        return
    c1, c2 = st.columns(2)
    with c1:
        show_chart(analytics.vehicle_class_chart(analytics.class_totals_df(DB, INCLUDE_SAMPLE)), "No vehicle counts.")
        show_chart(analytics.incidents_by_type_severity(incidents), "No incidents.")
        show_chart(analytics.priority_histogram(fusion), "No priority assessments.")
    with c2:
        show_chart(analytics.congestion_distribution(sessions), "No congestion estimates.")
        show_chart(analytics.incident_status_chart(incidents), "No incidents.")
        show_chart(analytics.daily_trend(sessions, incidents),
                   "Trend chart appears once there is data on at least two different days.")
    st.subheader("Export records")
    e = st.columns(3)
    e[0].download_button("Detections CSV", analytics.to_csv_bytes(analytics.label_sample(sessions)),
                         "detections.csv", "text/csv", disabled=sessions.empty)
    e[1].download_button("Incidents CSV", analytics.to_csv_bytes(analytics.label_sample(incidents)),
                         "incidents.csv", "text/csv", disabled=incidents.empty)
    e[2].download_button("Priority results CSV", analytics.to_csv_bytes(analytics.label_sample(fusion)),
                         "priority_results.csv", "text/csv", disabled=fusion.empty)


# ==========================================================================
# Page: Cloud & Settings
# ==========================================================================
def page_settings() -> None:
    header("Cloud & Settings", "Storage status, sample data, database backup and configuration.",
           badge("Optional AWS S3", "opt"))
    st.subheader("Cloud storage (AWS S3)")
    if st.button("Check storage status"):
        ok, message = STORAGE.status()
        (st.success if ok else st.info)(message)
    st.caption("S3 is optional. Without it, files are saved in the local `data/uploads` and `outputs` folders.")

    st.subheader("Sample data")
    st.caption("Sample records are generated demo data, marked SAMPLE, and hidden unless the sidebar toggle is on.")
    c1, c2 = st.columns(2)
    if c1.button("Load SAMPLE data"):
        counts = analytics.load_sample_data(DB, SETTINGS)
        st.success(f"Added {counts['sessions']} sample detections, {counts['incidents']} sample incidents and "
                   f"{counts['fusion']} sample priority results. Turn on the sidebar toggle to see them.")
    if c2.button("Delete all SAMPLE data"):
        st.success(f"Removed {db.delete_sample_data(DB)} sample records.")

    st.subheader("Database backup")
    if st.button("Create backup"):
        target = db.backup_database(DB, BASE_DIR / "data" / "backups")
        st.success(f"Backup created: {rel_path(target)}")
        st.download_button("Download backup file", target.read_bytes(), target.name, "application/octet-stream")

    st.subheader("Active configuration (no secrets shown)")
    cfg = {
        "database": rel_path(SETTINGS.db_path), "model": SETTINGS.yolo_model,
        "default_confidence": SETTINGS.default_confidence, "max_image_mb": SETTINGS.max_image_mb,
        "max_video_mb": SETTINGS.max_video_mb, "max_video_seconds": SETTINGS.max_video_seconds,
        "max_video_frames": SETTINGS.max_video_frames, "use_s3": SETTINGS.use_s3,
        "s3_bucket_set": bool(SETTINGS.s3_bucket), "admin_pin_set": bool(SETTINGS.admin_pin),
        "congestion": vars(SETTINGS.congestion), "fusion_weights": {
            "congestion": SETTINGS.fusion.congestion_weight, "incident": SETTINGS.fusion.incident_weight},
    }
    st.json(cfg, expanded=False)


ROUTES = {
    "Dashboard": page_dashboard,
    "Image Detection": page_image_detection,
    "Video Analysis": page_video,
    "Congestion Analysis": page_congestion,
    "Incidents": page_incidents,
    "Priority Engine": page_priority,
    "History": page_history,
    "Analytics & Reports": page_analytics,
    "Cloud & Settings": page_settings,
}

try:
    ROUTES[page]()
except Exception as exc:  # last-resort guard: show a friendly message, log details
    logger.exception("Unhandled error on page %s", page)
    db.log_event(DB, "ERROR", "unhandled_error", f"{page}: {exc.__class__.__name__}")
    st.error("Something went wrong on this page. The error was logged; please try again.")
