from __future__ import annotations
from io import BytesIO
import plotly.io as pio

STYLE_SPECS={
    "Minimal":dict(template="plotly_white",font_family="Arial",font_size=15),
    "Economics journal":dict(template="plotly_white",font_family="Times New Roman",font_size=14),
    "APA":dict(template="plotly_white",font_family="Arial",font_size=12),
    "Presentation":dict(template="plotly_white",font_family="Arial",font_size=19),
    "Dark presentation":dict(template="plotly_dark",font_family="Arial",font_size=19),
}
SIZE_SPECS={
    "Single column":(650,460),
    "Double column":(1050,600),
    "Presentation 16:9":(1280,720),
    "Square":(700,700),
}

def apply_publication_style(fig, style_name="Minimal", size_name="Double column", title=None):
    spec=STYLE_SPECS.get(style_name,STYLE_SPECS["Minimal"])
    width,height=SIZE_SPECS.get(size_name,SIZE_SPECS["Double column"])
    fig=fig
    fig.update_layout(
        template=spec["template"],font=dict(family=spec["font_family"],size=spec["font_size"]),
        width=width,height=height,title=title if title is not None else fig.layout.title.text,
        margin=dict(l=70,r=35,t=75,b=65),legend_title_text="",
    )
    fig.update_xaxes(showline=True,linewidth=1,mirror=False)
    fig.update_yaxes(showline=True,linewidth=1,mirror=False)
    return fig

def figure_bytes(fig, fmt="png", scale=2):
    return pio.to_image(fig,format=fmt,width=fig.layout.width,height=fig.layout.height,scale=scale)

def figure_html(fig):
    return pio.to_html(fig,include_plotlyjs="cdn",full_html=True).encode()
