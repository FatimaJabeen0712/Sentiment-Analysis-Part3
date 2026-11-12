"""Shared look for every page: petrol blue and ice white, with high-contrast text.

Colors (contrast ratios against their background are all at least 4.5:1, most above 7:1):
  petrol    #0B5D6E   buttons, accents, headings
  deep      #073B47   sidebar and hero background, dark text accents
  ice       #F3FAFC   page background
  ice-2     #E2F0F4   cards, tables, secondary background
  text      #0A2A33   body text on ice (about 14:1)
"""
from __future__ import annotations

import html

import plotly.io as pio
import streamlit as st

PETROL, DEEP, MID = "#0B5D6E", "#073B47", "#1B7F91"
ICE, ICE2, BORDER = "#F3FAFC", "#E2F0F4", "#BFD9E0"
TEXT, MUTED = "#0A2A33", "#3D5E69"
SENT_COLORS = {"positive": "#17806D", "neutral": "#8FA7AF", "negative": "#C0502F"}
COLORWAY = [PETROL, "#2AA3B5", "#8CCBD6", "#E0A03A", "#6B7F89", "#C0502F", "#17806D", "#073B47"]
PLOT_LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=44, b=10), colorway=COLORWAY,
                   paper_bgcolor=ICE, plot_bgcolor="#FFFFFF",
                   font=dict(color=TEXT, size=13), title_font=dict(color=DEEP, size=16))

_t = pio.templates["plotly_white"]
_t.layout.colorway = COLORWAY
_t.layout.colorscale.sequential = [[0, "#E2F0F4"], [0.5, "#2AA3B5"], [1, "#073B47"]]
pio.templates.default = "plotly_white"

PAGES = [("app.py", "Part 1 - Customer Voice"), ("pages/1_Ad_Budget_Optimizer.py", "Part 2 - Ad Budget Optimizer"),
         ("pages/2_Audience_Segments.py", "Part 3 - Audience & Campaigns"), ("pages/3_Project_Hub.py", "Project Hub - Action Brief")]

CSS = f"""
<style>
:root {{ --petrol:{PETROL}; --deep:{DEEP}; --ice:{ICE}; --ice2:{ICE2}; --text:{TEXT}; --border:{BORDER}; }}
.stApp {{ background: var(--ice); color: var(--text); }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 3.6rem !important; max-width: 1300px; }}
[data-testid="stSidebarNav"] {{ display: none; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ opacity: 1 !important; }}
h1, h2, h3, h4 {{ color: var(--deep) !important; letter-spacing: -0.01em; }}
p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] {{ color: var(--text); }}
[data-testid="stCaptionContainer"] p {{ color: #2F505B !important; }}
a {{ color: {PETROL}; font-weight: 600; }}

.pj-hero {{ background: linear-gradient(115deg, {DEEP} 0%, {PETROL} 70%, {MID} 100%); border-radius: 14px;
  padding: 1.15rem 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 10px rgba(7,59,71,.18); }}
.pj-hero h1 {{ color: #FFFFFF !important; font-size: 1.65rem; margin: 0 0 .25rem 0; }}
.pj-hero p {{ color: #E6F4F8 !important; margin: 0; font-size: .98rem; }}
.pj-hero .pj-tag {{ display:inline-block; background: rgba(255,255,255,.16); color:#FFFFFF; border-radius: 999px;
  padding: .1rem .65rem; font-size: .78rem; font-weight: 600; margin-bottom: .45rem; letter-spacing:.03em; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: .25rem; border-bottom: 2px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ color: {MUTED}; font-weight: 600; padding: .55rem .9rem; background: transparent; }}
.stTabs [aria-selected="true"] {{ color: {PETROL} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {PETROL}; height: 3px; }}

/* metric cards */
div[data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid var(--border); border-left: 4px solid {PETROL};
  border-radius: 10px; padding: .65rem .85rem; }}
div[data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-weight: 600; }}
div[data-testid="stMetricValue"] {{ color: var(--deep) !important; }}
div[data-testid="stMetricValue"] > div {{ white-space: normal !important; overflow: visible !important; text-overflow: clip !important; font-size: 1.55rem; line-height: 1.25; }}

/* buttons */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{ border-radius: 8px; font-weight: 600;
  border: 1.5px solid {PETROL}; color: {PETROL}; background: #FFFFFF; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ background: {ICE2}; color: {DEEP}; border-color: {DEEP}; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{ background: {PETROL}; color: #FFFFFF; border-color: {PETROL}; }}
.stButton > button[kind="primary"]:hover {{ background: {DEEP}; color: #FFFFFF; }}
.stButton > button[kind="primary"] p {{ color: #FFFFFF !important; }}

/* inputs */
.stTextInput input, .stTextArea textarea, .stNumberInput input, [data-baseweb="select"] > div,
[data-baseweb="input"] {{ background: #FFFFFF !important; color: {TEXT} !important; border-color: {BORDER} !important; }}
[data-baseweb="select"] span, [data-baseweb="select"] div {{ color: {TEXT}; }}
[data-baseweb="popover"] li, [data-baseweb="menu"] li {{ color: {TEXT} !important; background: #FFFFFF; }}
[data-baseweb="menu"] li:hover {{ background: {ICE2} !important; }}
::placeholder {{ color: #6A858E !important; opacity: 1 !important; }}

/* tables, expanders, alerts */
[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 8px; }}
[data-testid="stExpander"] {{ background: #FFFFFF; border: 1px solid var(--border); border-radius: 10px; }}
[data-testid="stExpander"] summary p {{ color: var(--deep) !important; font-weight: 600; }}
div[data-testid="stAlert"] {{ border-radius: 10px; border: 1px solid var(--border); border-left: 5px solid {PETROL}; }}
div[data-testid="stAlert"] p, div[data-testid="stAlert"] li {{ color: {TEXT} !important; }}
div[data-testid="stAlertContainer"] {{ background: #FFFFFF; }}
code {{ background: {ICE2}; color: {DEEP}; padding: .05rem .3rem; border-radius: 4px; }}

/* sidebar (deep petrol with ice text) */
[data-testid="stSidebar"] {{ background: {DEEP}; }}
[data-testid="stSidebar"] * {{ color: #EAF6F9; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: #FFFFFF !important; }}
[data-testid="stSidebar"] a {{ color: #FFFFFF; }}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] * {{ color: {TEXT} !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] div, [data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"] {{ background-color: #FFFFFF !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] svg {{ fill: {PETROL} !important; }}
[data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] div, [data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] div {{ background-color: #FFFFFF !important; color: {TEXT} !important; }}
[data-testid="stSidebar"] :is([data-testid="stSelectbox"], [data-testid="stMultiSelect"]) div:has(input),
[data-testid="stSidebar"] :is([data-testid="stSelectbox"], [data-testid="stMultiSelect"]) div:has(input) * {{ background-color: #FFFFFF !important; color: {TEXT} !important; -webkit-text-fill-color: {TEXT} !important; }}
[data-testid="stSidebar"] :is([data-testid="stSelectbox"], [data-testid="stMultiSelect"]) svg {{ fill: {PETROL} !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div, [data-testid="stSidebar"] [data-baseweb="select"] > div > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div > div > div {{ background-color: #FFFFFF !important; color: {TEXT} !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] input, [data-testid="stSidebar"] [data-baseweb="select"] span {{ color: {TEXT} !important; -webkit-text-fill-color: {TEXT} !important; }}
[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"], [data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] * {{ background: transparent !important; color: #EAF6F9 !important; fill: #EAF6F9 !important; }}
[data-testid="stSidebar"] [data-testid="stTooltipIcon"], [data-testid="stSidebar"] [data-testid="stTooltipIcon"] * {{ background: transparent !important; color: #EAF6F9 !important; fill: #EAF6F9 !important; }}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {{ background-color: {ICE2} !important; color: {DEEP} !important; }}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg {{ fill: {DEEP} !important; }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"], [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {{ background: transparent !important; color: #EAF6F9 !important; }}
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg, [data-testid="stSidebar"] [data-testid="stMultiSelect"] svg {{ fill: {PETROL} !important; }}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button {{ background-color: {ICE2} !important; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label p, [data-testid="stSidebar"] [data-testid="stCheckbox"] label p {{ color: #EAF6F9 !important; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ color: #D5EDF2 !important; }}
[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] button, [data-testid="stSidebar"] [data-testid="stNumberInput"] [data-testid="stTooltipHoverTarget"] button {{ background-color: transparent !important; border: 0 !important; }}
[data-testid="stSidebar"] [data-testid="stTooltipHoverTarget"] button svg {{ fill: #EAF6F9 !important; }}
.pj-banner {{ background:#FFF4D6; color:#4A3500; border:1px solid #E5C56A; border-left:5px solid #E0A03A;
  border-radius:8px; padding:.55rem .9rem; font-weight:600; margin-bottom:.8rem; }}
.pj-flow {{ display:flex; gap:.6rem; flex-wrap:wrap; margin:.4rem 0 1rem 0; }}
.pj-step {{ flex:1 1 210px; background:#FFFFFF; border:1px solid {BORDER}; border-top:4px solid {PETROL}; border-radius:10px;
  padding:.7rem .9rem; color:{TEXT}; }}
.pj-step b {{ color:{DEEP}; }}
.pj-step span {{ color:{MUTED}; font-size:.86rem; }}
</style>
"""


def apply_theme(title: str, subtitle: str = "", tag: str = "", icon: str = "") -> None:
    """Inject the shared CSS and draw the page header. All text is escaped."""
    st.markdown(CSS, unsafe_allow_html=True)
    tag_html = f'<div class="pj-tag">{html.escape(tag)}</div>' if tag else ""
    st.markdown(f'<div class="pj-hero">{tag_html}<h1>{html.escape((icon + " " if icon else "") + title)}</h1>'
                f'<p>{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def banner(text: str) -> None:
    st.markdown(f'<div class="pj-banner">{html.escape(text)}</div>', unsafe_allow_html=True)


def sidebar_nav() -> None:
    """Links to every part (skipped silently if the page registry is unavailable, e.g. in tests)."""
    try:
        with st.sidebar:
            st.markdown("### Fashion & Jewelry Project")
            for path, label in PAGES:
                st.page_link(path, label=label)
            st.divider()
    except Exception:  # noqa: BLE001 - navigation is a convenience, never a reason to fail a page
        pass


def style_fig(fig):
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_xaxes(tickfont=dict(color=TEXT), title_font=dict(color=TEXT), gridcolor="#D5E6EB")
    fig.update_yaxes(tickfont=dict(color=TEXT), title_font=dict(color=TEXT), gridcolor="#D5E6EB")
    fig.update_layout(legend=dict(font=dict(color=TEXT)))
    return fig


def plot_chart(fig, **_ignored) -> None:
    """Draw a Plotly figure in the project style (theme=None stops Streamlit recolouring it)."""
    st.plotly_chart(style_fig(fig), theme=None, width="stretch")
