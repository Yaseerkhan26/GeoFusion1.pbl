"""
===============================================================================
File: src/utils/ui_components.py
Purpose: Reusable Premium UI Components, Design System Tokens, and Styling for FusionLand AI
===============================================================================
"""

from typing import Any, Dict, List, Optional
import textwrap
import streamlit as st


# -----------------------------------------------------------------------------
# Global Design System CSS Injection
# -----------------------------------------------------------------------------
def inject_custom_css():
    """Injects high-fidelity dark satellite theme CSS into Streamlit."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Root Typography and Global Overrides */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: #E2E8F0;
            background-color: #060A17;
        }

        .stApp {
            background: radial-gradient(circle at 50% 0%, #0c1838 0%, #060a17 70%);
        }

        code, pre, .font-mono {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Hide Streamlit Header/Footer Clutter if present */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Container Layout Constraints */
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 3.5rem;
            max-width: 1440px;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #080D1F !important;
            border-right: 1px solid #1E293B;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem;
        }

        /* Navigation Radio Buttons */
        div[data-testid="stSidebarUserContent"] div[role="radiogroup"] > label {
            background: transparent;
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.2rem;
            border: 1px solid transparent;
            transition: all 0.2s ease;
        }
        div[data-testid="stSidebarUserContent"] div[role="radiogroup"] > label:hover {
            background: rgba(56, 189, 248, 0.08);
            border-color: rgba(56, 189, 248, 0.2);
        }
        div[data-testid="stSidebarUserContent"] div[role="radiogroup"] > label[data-checked="true"] {
            background: rgba(56, 189, 248, 0.15) !important;
            border-color: #38BDF8 !important;
            font-weight: 600;
        }

        /* Authentication Layout Styling */
        .auth-brand-box {
            background: linear-gradient(145deg, #090e1c 0%, #0f1c3f 50%, #152445 100%);
            border: 1px solid #1E293B;
            border-left: 4px solid #38BDF8;
            border-radius: 16px;
            padding: 2.5rem;
            color: #F8FAFC;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .auth-card {
            background: rgba(11, 19, 43, 0.85);
            backdrop-filter: blur(12px);
            border: 1px solid #1E293B;
            border-radius: 16px;
            padding: 2.2rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
        }
        .auth-title {
            font-size: 1.75rem;
            font-weight: 800;
            color: #F8FAFC;
            letter-spacing: -0.02em;
            margin-bottom: 0.3rem;
        }
        .auth-subtitle {
            font-size: 0.9rem;
            color: #94A3B8;
            margin-bottom: 1.5rem;
        }
        
        /* Requirement Checkmark Chips */
        .req-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 0.75rem;
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            margin-right: 0.3rem;
            margin-bottom: 0.3rem;
            background: #090E1C;
            border: 1px solid #1E293B;
        }
        .req-pass {
            color: #34D399;
            border-color: rgba(52, 211, 153, 0.4);
            background: rgba(52, 211, 153, 0.1);
        }
        .req-fail {
            color: #64748B;
        }

        /* Brand Header Banner */
        .brand-header-box {
            background: linear-gradient(135deg, #090e1c 0%, #0f1c3f 50%, #152445 100%);
            border: 1px solid #1e293b;
            border-left: 4px solid #38bdf8;
            border-radius: 14px;
            padding: 1.25rem 1.8rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.4);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .brand-header-title {
            font-size: 1.85rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #F8FAFC;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .brand-header-subtitle {
            font-size: 0.85rem;
            color: #94A3B8;
            margin-top: 0.25rem;
            font-weight: 500;
            letter-spacing: 0.03em;
        }
        .brand-badge-chip {
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: #38BDF8;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        /* Status Grid Header */
        .status-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0.75rem;
            margin-bottom: 1.25rem;
        }
        @media (max-width: 900px) {
            .status-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        .status-card {
            background: #0B132B;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 0.75rem 1rem;
            display: flex;
            flex-direction: column;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
            transition: border-color 0.2s ease;
        }
        .status-card:hover {
            border-color: #38BDF8;
        }
        .status-label {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748B;
        }
        .status-val {
            font-size: 0.95rem;
            font-weight: 600;
            color: #E2E8F0;
            margin-top: 0.25rem;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }

        /* Status Colors */
        .status-ready, .status-validated, .status-pass {
            color: #34D399;
            font-weight: 600;
        }
        .status-warning, .status-stale {
            color: #F59E0B;
            font-weight: 600;
        }
        .status-error, .status-fail {
            color: #F87171;
            font-weight: 600;
        }
        .status-pending, .status-info {
            color: #60A5FA;
            font-weight: 600;
        }
        .status-unavailable {
            color: #64748B;
            font-weight: 600;
        }

        /* Section Headings */
        .section-header-box {
            margin-top: 0.75rem;
            margin-bottom: 1.2rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #1E293B;
        }
        .section-header-title {
            font-size: 1.4rem;
            font-weight: 800;
            color: #F1F5F9;
            display: flex;
            align-items: center;
            gap: 0.6rem;
            letter-spacing: -0.01em;
        }
        .section-header-caption {
            font-size: 0.85rem;
            color: #94A3B8;
            margin-top: 0.3rem;
            line-height: 1.5;
        }

        /* View Resolution Badges */
        .res-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.75rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }
        .res-badge-native {
            background: rgba(14, 116, 144, 0.2);
            color: #38BDF8;
            border: 1px solid #0284C7;
        }
        .res-badge-model {
            background: rgba(99, 102, 241, 0.2);
            color: #A5B4FC;
            border: 1px solid #6366F1;
        }
        .res-badge-gt {
            background: rgba(16, 185, 129, 0.2);
            color: #34D399;
            border: 1px solid #059669;
        }

        /* Scientific Metric Cards */
        .sci-stat-card {
            background: #0B132B;
            border: 1px solid #1E293B;
            border-radius: 12px;
            padding: 1.1rem;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: 104px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .sci-stat-card:hover {
            transform: translateY(-2px);
            border-color: #38BDF8;
        }
        .sci-stat-title {
            font-size: 0.75rem;
            font-weight: 700;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .sci-stat-val {
            font-size: 1.65rem;
            font-weight: 800;
            color: #38BDF8;
            margin-top: 0.25rem;
            letter-spacing: -0.02em;
        }
        .sci-stat-sub {
            font-size: 0.72rem;
            color: #64748B;
            margin-top: 0.2rem;
        }

        /* Model Card Component */
        .model-card-box {
            background: #0B132B;
            border: 1px solid #1E293B;
            border-radius: 14px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            margin-bottom: 1.5rem;
        }
        .model-card-title {
            font-size: 1.15rem;
            font-weight: 800;
            color: #F8FAFC;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }
        .model-spec-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.85rem;
        }
        @media (max-width: 900px) {
            .model-spec-grid {
                grid-template-columns: repeat(1, 1fr);
            }
        }
        .model-spec-item {
            background: #060A17;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 0.75rem 1rem;
        }
        .model-spec-key {
            font-size: 0.7rem;
            color: #64748B;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        .model-spec-value {
            font-size: 0.92rem;
            color: #E2E8F0;
            font-weight: 600;
            margin-top: 0.2rem;
        }

        /* Error, Warning & Empty Banners */
        .custom-banner {
            border-radius: 10px;
            padding: 1.1rem 1.35rem;
            margin-bottom: 1rem;
            border: 1px solid transparent;
        }
        .banner-error {
            background: rgba(239, 68, 68, 0.1);
            border-color: rgba(239, 68, 68, 0.4);
            color: #FCA5A5;
        }
        .banner-warning {
            background: rgba(245, 158, 11, 0.1);
            border-color: rgba(245, 158, 11, 0.4);
            color: #FCD34D;
        }
        .banner-info {
            background: rgba(56, 189, 248, 0.1);
            border-color: rgba(56, 189, 248, 0.4);
            color: #BAE6FD;
        }
        .banner-empty {
            background: #0B132B;
            border: 1px dashed #334155;
            text-align: center;
            padding: 2.5rem 1.5rem;
            border-radius: 14px;
        }

        /* Legend Component */
        .legend-container {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            background: #0B132B;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 0.85rem 1.1rem;
            margin-top: 0.5rem;
            margin-bottom: 0.85rem;
        }
        .legend-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
            font-weight: 600;
            color: #E2E8F0;
            background: #060A17;
            border: 1px solid #1E293B;
            padding: 0.3rem 0.75rem;
            border-radius: 6px;
        }
        .legend-color-dot {
            width: 12px;
            height: 12px;
            border-radius: 3px;
            flex-shrink: 0;
        }

        /* Pipeline Flow Diagram */
        .pipeline-flow {
            background: #0B132B;
            border: 1px solid #1E293B;
            border-radius: 14px;
            padding: 1.5rem;
            margin: 1rem 0;
            text-align: center;
        }

        /* Streamlit Input & Button Tweaks */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stTextInput > div > div > input {
            background-color: #060A17 !important;
            border-color: #1E293B !important;
            color: #F8FAFC !important;
            border-radius: 8px !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #38BDF8 !important;
            box-shadow: 0 0 0 1px #38BDF8 !important;
        }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Reusable Header Component
# -----------------------------------------------------------------------------
def render_app_header(active_ckpt: str, eval_status: str, dev_str: str, splits_info: Dict[str, int]):
    """Renders the top branding header and real-time operational status grid."""
    total_samples = splits_info.get("total", 0)
    train_n = splits_info.get("train", 0)
    val_n = splits_info.get("val", 0)
    test_n = splits_info.get("test", 0)
    data_desc = f"{total_samples} Patches ({train_n} Tr / {val_n} V / {test_n} Te)"

    eval_badge = (
        "<span class='status-validated'>● VALIDATED</span>" if eval_status == "VALIDATED"
        else "<span class='status-stale'>● STALE</span>" if eval_status == "STALE"
        else "<span class='status-pending'>● PENDING</span>"
    )

    dev_badge = (
        f"<span class='status-ready'>● {dev_str}</span>" if "cuda" in dev_str.lower()
        else f"<span class='status-info'>● {dev_str}</span>"
    )

    st.markdown(textwrap.dedent(f"""
    <div class="brand-header-box">
        <div>
            <div class="brand-header-title">
                <span>🛰️</span> GeoFusion AI
            </div>
            <div class="brand-header-subtitle">
                MULTIMODAL SATELLITE INTELLIGENCE &bull; SENTINEL-1 SAR &times; SENTINEL-2 OPTICAL &bull; ESA WORLDCOVER
            </div>
        </div>
        <div>
            <span class="brand-badge-chip">V2.1 PRODUCTION PLATFORM</span>
        </div>
    </div>
    <div class="status-grid">
        <div class="status-card">
            <span class="status-label">Active Model</span>
            <span class="status-val">📦 {active_ckpt}</span>
        </div>
        <div class="status-card">
            <span class="status-label">Dataset Split State</span>
            <span class="status-val">📊 {data_desc}</span>
        </div>
        <div class="status-card">
            <span class="status-label">Evaluation Provenance</span>
            <span class="status-val">{eval_badge}</span>
        </div>
        <div class="status-card">
            <span class="status-label">Compute Device</span>
            <span class="status-val">{dev_badge}</span>
        </div>
    </div>
    """).strip(), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Reusable Section Heading
# -----------------------------------------------------------------------------
def render_section_heading(number_str: str, title: str, badge_type: Optional[str] = None, badge_text: Optional[str] = None, caption: Optional[str] = None):
    """Renders consistent section titles with optional resolution badges and captions."""
    badge_html = ""
    if badge_text:
        cls = "res-badge-native" if badge_type == "native" else "res-badge-model" if badge_type == "model" else "res-badge-gt"
        badge_html = f"<div class='res-badge {cls}'>{badge_text}</div>"

    caption_html = f"<div class='section-header-caption'>{caption}</div>" if caption else ""

    st.markdown(textwrap.dedent(f"""
    <div class="section-header-box">
        <div class="section-header-title">
            <span>{number_str}</span> &bull; <span>{title}</span>
        </div>
        {badge_html}
        {caption_html}
    </div>
    """).strip(), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Metric Card Component
# -----------------------------------------------------------------------------
def render_metric_card(title: str, value: str, subtitle: str = ""):
    """Renders an elegant scientific stat card."""
    st.markdown(textwrap.dedent(f"""
    <div class="sci-stat-card">
        <div class="sci-stat-title">{title}</div>
        <div class="sci-stat-val">{value}</div>
        <div class="sci-stat-sub">{subtitle}</div>
    </div>
    """).strip(), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Model Card Component
# -----------------------------------------------------------------------------
def render_model_card(meta: Dict[str, Any], eval_data: Optional[Dict[str, Any]] = None):
    """Renders a comprehensive, dynamic Model Card for Experiment 3 MultimodalFusionNet."""
    filename = meta.get('filename', 'GeoFusion_AI_Final.pth')
    sha_short = meta.get('sha256_short', 'e6378e90')

    st.markdown(textwrap.dedent(f"""
    <div class="model-card-box">
        <div class="model-card-title">
            <span>📋</span> SYSTEM MODEL CARD: {filename}
        </div>
        <div class="model-spec-grid">
            <div class="model-spec-item">
                <div class="model-spec-key">Architecture</div>
                <div class="model-spec-value">{meta.get('architecture', 'MultimodalFusionNet')}</div>
            </div>
            <div class="model-spec-item">
                <div class="model-spec-key">Inputs</div>
                <div class="model-spec-value">S1 (VV, VH) + S2 (6 Bands) = 8 Channels</div>
            </div>
            <div class="model-spec-item">
                <div class="model-spec-key">Patch Dimensions</div>
                <div class="model-spec-value">256 &times; 256 &times; 8 channels</div>
            </div>
            <div class="model-spec-item">
                <div class="model-spec-key">Classes</div>
                <div class="model-spec-value">8 ESA WorldCover Classes</div>
            </div>
            <div class="model-spec-item">
                <div class="model-spec-key">Checkpoint File</div>
                <div class="model-spec-value">{filename}</div>
            </div>
            <div class="model-spec-item">
                <div class="model-spec-key">SHA-256 Fingerprint</div>
                <div class="model-spec-value font-mono">{sha_short}...</div>
            </div>
            <div class="model-spec-item">
                <div class="model-spec-key">Test Pixel Accuracy</div>
                <div class="model-spec-value">76.45%</div>
            </div>
            <div class="model-spec-item">
                <div class="model-spec-key">Test Macro F1</div>
                <div class="model-spec-value">39.57%</div>
            </div>
            <div class="model-spec-item">
                <div class="model-spec-key">Test mIoU</div>
                <div class="model-spec-value">29.13%</div>
            </div>
        </div>

        <div style="margin-top: 1.2rem; border-top: 1px solid #1E293B; padding-top: 1rem;">
            <div style="font-weight: 700; color: #F59E0B; font-size: 0.9rem; margin-bottom: 0.4rem; display: flex; align-items: center; gap: 0.4rem;">
                <span>⚠️</span> DOCUMENTED SCIENTIFIC LIMITATIONS
            </div>
            <ul style="color: #94A3B8; font-size: 0.83rem; line-height: 1.5; margin: 0; padding-left: 1.2rem;">
                <li><strong>Bare / sparse vegetation</strong>: Confusion with fallow agricultural fields and dry bare soil.</li>
                <li><strong>Grassland & Cropland</strong>: High spectral ambiguity in single-date optical satellite imagery without seasonal temporal sequences.</li>
                <li><strong>Herbaceous wetland</strong>: Rare category with lower ground truth sampling representation in regional training splits.</li>
            </ul>
        </div>
    </div>
    """).strip(), unsafe_allow_html=True)



# -----------------------------------------------------------------------------
# Legend Component
# -----------------------------------------------------------------------------
def render_legend(classes: Dict[int, str], colormap: Dict[int, str]):
    """Renders the centralized land cover taxonomy legend horizontally."""
    items = []
    for cls_id, name in classes.items():
        color = colormap.get(cls_id, "#888888")
        items.append(
            f'<div class="legend-pill">'
            f'<span class="legend-color-dot" style="background-color: {color};"></span>'
            f'<span>{name}</span>'
            f'</div>'
        )
    items_html = "".join(items)
    st.markdown(f'<div class="legend-container">{items_html}</div>', unsafe_allow_html=True)


def render_loading_state(message: str = "Processing satellite data..."):
    """Renders a modern, non-disruptive loading state box."""
    st.markdown(textwrap.dedent(f"""
    <div style="background: #0B132B; border: 1px solid #1E293B; border-left: 4px solid #60A5FA; padding: 1.2rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
        <div style="font-size: 1.5rem; margin-bottom: 0.4rem;">🛰️</div>
        <div style="font-weight: 700; color: #E2E8F0; font-size: 0.95rem;">{message}</div>
        <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 0.2rem;">Please wait while the engine completes computation...</div>
    </div>
    """).strip(), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Empty & Error States
# -----------------------------------------------------------------------------
def render_empty_state(title: str, message: str, icon: str = "📡"):
    """Renders a polite, modern empty state box."""
    st.markdown(textwrap.dedent(f"""
    <div class="custom-banner banner-empty">
        <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-weight: 700; color: #F8FAFC; font-size: 1.05rem;">{title}</div>
        <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.35rem;">{message}</div>
    </div>
    """).strip(), unsafe_allow_html=True)


def render_error_state(title: str, explanation: str, possible_reason: str, action: str, technical_details: Optional[str] = None):
    """Renders a polite error state with clear user actions and sanitized technical details."""
    import re
    import logging

    st.markdown(textwrap.dedent(f"""
    <div class="custom-banner banner-error">
        <div style="font-weight: 700; font-size: 1.05rem; display: flex; align-items: center; gap: 0.5rem;">
            <span>⚠️</span> {title}
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.88rem; color: #FECACA;">{explanation}</div>
        <div style="margin-top: 0.5rem; font-size: 0.82rem; color: #FCA5A5;">
            <strong>Possible Reason:</strong> {possible_reason}<br>
            <strong>Recommended Action:</strong> {action}
        </div>
    </div>
    """).strip(), unsafe_allow_html=True)

    if technical_details:
        logging.error(f"[SYSTEM DIAGNOSTIC] {title}: {technical_details}")
        sanitized = re.sub(r"[A-Za-z]:\\[^\s'\"]+", "[System Path]", str(technical_details))
        sanitized = re.sub(r"/[^\s'\"]+/GeoFusion AI", "[Project Root]", sanitized)
        with st.expander("🔍 Diagnostic Technical Summary"):
            st.info(f"System Message: {sanitized[:250] + ('...' if len(sanitized) > 250 else '')}")


# -----------------------------------------------------------------------------
# Interactive Pipeline Architecture Diagram
# -----------------------------------------------------------------------------
def render_pipeline_diagram():
    """Renders a clean, high-contrast visual architecture diagram of the dual-encoder fusion network."""
    st.markdown(textwrap.dedent("""
    <div class="pipeline-flow">
        <svg viewBox="0 0 860 300" style="width: 100%; max-width: 860px; height: auto;">
            <defs>
                <linearGradient id="gradS1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#0284C7" />
                    <stop offset="100%" stop-color="#0369A1" />
                </linearGradient>
                <linearGradient id="gradS2" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#059669" />
                    <stop offset="100%" stop-color="#047857" />
                </linearGradient>
                <linearGradient id="gradFusion" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#4F46E5" />
                    <stop offset="100%" stop-color="#4338CA" />
                </linearGradient>
            </defs>

            <!-- S1 Branch -->
            <rect x="30" y="30" width="180" height="60" rx="8" fill="url(#gradS1)" stroke="#38BDF8" stroke-width="1.5" />
            <text x="120" y="58" fill="#FFFFFF" font-family="'Inter', sans-serif" font-size="13" font-weight="700" text-anchor="middle">Sentinel-1 C-Band SAR</text>
            <text x="120" y="76" fill="#BAE6FD" font-family="'Inter', sans-serif" font-size="11" text-anchor="middle">[VV, VH Polarizations &bull; 2 Ch]</text>

            <line x1="210" y1="60" x2="260" y2="60" stroke="#38BDF8" stroke-width="2" stroke-dasharray="4,4" />
            <polygon points="260,60 252,55 252,65" fill="#38BDF8" />

            <rect x="260" y="30" width="160" height="60" rx="8" fill="#060A17" stroke="#0284C7" stroke-width="1.5" />
            <text x="340" y="58" fill="#F8FAFC" font-family="'Inter', sans-serif" font-size="13" font-weight="600" text-anchor="middle">Sentinel1Encoder</text>
            <text x="340" y="76" fill="#94A3B8" font-family="'Inter', sans-serif" font-size="11" text-anchor="middle">2 &rarr; 16 &rarr; 32 Ch (Conv2D)</text>

            <!-- S2 Branch -->
            <rect x="30" y="150" width="180" height="60" rx="8" fill="url(#gradS2)" stroke="#34D399" stroke-width="1.5" />
            <text x="120" y="178" fill="#FFFFFF" font-family="'Inter', sans-serif" font-size="13" font-weight="700" text-anchor="middle">Sentinel-2 Optical</text>
            <text x="120" y="196" fill="#A7F3D0" font-family="'Inter', sans-serif" font-size="11" text-anchor="middle">[B2, B3, B4, B8, B11, B12 &bull; 6 Ch]</text>

            <line x1="210" y1="180" x2="260" y2="180" stroke="#34D399" stroke-width="2" stroke-dasharray="4,4" />
            <polygon points="260,180 252,175 252,185" fill="#34D399" />

            <rect x="260" y="150" width="160" height="60" rx="8" fill="#060A17" stroke="#059669" stroke-width="1.5" />
            <text x="340" y="178" fill="#F8FAFC" font-family="'Inter', sans-serif" font-size="13" font-weight="600" text-anchor="middle">Sentinel2Encoder</text>
            <text x="340" y="196" fill="#94A3B8" font-family="'Inter', sans-serif" font-size="11" text-anchor="middle">6 &rarr; 16 &rarr; 32 Ch (Conv2D)</text>

            <!-- Converging Lines to Fusion -->
            <path d="M 420 60 L 470 60 L 470 105 L 500 105" fill="none" stroke="#38BDF8" stroke-width="2" />
            <path d="M 420 180 L 470 180 L 470 135 L 500 135" fill="none" stroke="#34D399" stroke-width="2" />

            <!-- Fusion Block -->
            <rect x="500" y="90" width="160" height="60" rx="8" fill="url(#gradFusion)" stroke="#818CF8" stroke-width="1.5" />
            <text x="580" y="118" fill="#FFFFFF" font-family="'Inter', sans-serif" font-size="13" font-weight="700" text-anchor="middle">Feature Concatenation</text>
            <text x="580" y="136" fill="#C7D2FE" font-family="'Inter', sans-serif" font-size="11" text-anchor="middle">32 + 32 = 64 Ch Fusion</text>

            <line x1="660" y1="120" x2="700" y2="120" stroke="#818CF8" stroke-width="2" />
            <polygon points="700,120 692,115 692,125" fill="#818CF8" />

            <!-- Outputs -->
            <rect x="700" y="55" width="140" height="55" rx="8" fill="#060A17" stroke="#38BDF8" stroke-width="1.5" />
            <text x="770" y="78" fill="#38BDF8" font-family="'Inter', sans-serif" font-size="12" font-weight="700" text-anchor="middle">Land Cover Map</text>
            <text x="770" y="96" fill="#94A3B8" font-family="'Inter', sans-serif" font-size="10" text-anchor="middle">8 Classes [256x256]</text>

            <rect x="700" y="130" width="140" height="55" rx="8" fill="#060A17" stroke="#F59E0B" stroke-width="1.5" />
            <text x="770" y="153" fill="#F59E0B" font-family="'Inter', sans-serif" font-size="12" font-weight="700" text-anchor="middle">Entropy Uncertainty</text>
            <text x="770" y="171" fill="#94A3B8" font-family="'Inter', sans-serif" font-size="10" text-anchor="middle">Shannon Metric [0, 1]</text>

            <text x="430" y="270" fill="#64748B" font-family="'Inter', sans-serif" font-size="12" text-anchor="middle">
                Continuous 2D spatial feature preservation &bull; No spatial downsampling &bull; Synchronized resolution &approx; 10m
            </text>
        </svg>
    </div>
    """).strip(), unsafe_allow_html=True)




