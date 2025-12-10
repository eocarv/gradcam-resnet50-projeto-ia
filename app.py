import os
from typing import Dict, List, Tuple, Optional

import streamlit as st
from PIL import Image

from pipelines.gradcam_pipeline import infer_gradcam
from pipelines.pix2pix_pipeline import infer_pix2pix

import json
from pathlib import Path
import base64

PIX2PIX_EXPL = (
    "O pix2pix é um método de tradução imagem para imagem com GAN condicional, que aprende a mapear "
    "uma imagem de entrada para uma saída correspondente (ISOLA; ZHU; ZHOU; EFROS, 2017). "
    "As métricas comparam a reconstrução com a imagem original: Ciede2000_sum_mask soma o ΔE2000 na região mascarada, "
    "Top2pct_mean_deltaE2000 é a média do ΔE2000 nos 2% mais discrepantes, Top1pct_energy_fraction mede a fração do erro "
    "total concentrada no 1% mais discrepante e Hist_similarity_rgb mede a similaridade global por interseção de histogramas RGB. "
)

GRADCAM_EXPL = (
    "O Grad-CAM é uma técnica de explicabilidade que gera um mapa de calor a partir dos gradientes da classe de interesse "
    "nas últimas camadas convolucionais, destacando as regiões da imagem que mais influenciaram a decisão do modelo "
    "(SELVARAJU; COGSWELL; DAS; VEDANTAM; PARIKH; BATRA, 2017). "
    "Na inferência, o sistema exibe o heatmap e o overlay para indicar onde o modelo ‘olhou’. "
    "As métricas exibidas resumem a decisão do classificador, tipicamente informando a classe prevista, a confiança ou probabilidade "
    "associada e o score usado para ranquear as classes. "
    )




PORT = int(os.environ.get("PORT", "7860"))

WEIGHTS_GRADCAM = os.path.join("weights", "gradcam_model_state_dict.pth")
WEIGHTS_PIX2PIX_TS = os.path.join("weights", "pix2pix_netG_ts.pt")

def _img_to_data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    ext = p.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"



def _resize_rgb(img: Image.Image, size: int) -> Image.Image:
    return img.convert("RGB").resize((size, size), Image.BICUBIC)


def _inject_css() -> None:
    st.markdown(
        """
<style>
html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
main .block-container { max-width: 1120px; padding-top: 1.25rem; padding-bottom: 2.25rem; }

header, footer { visibility: hidden; height: 0; }

.brandtext{ display:flex; flex-direction:column; gap:2px; }
.brandtitle{ font-weight: 760; line-height: 1.05; }
.brandsub{
  font-size: 20px;          
  color: var(--muted);
  letter-spacing: .2px;    
}

.logo-wrap{
  position: relative;
  width: 100px;
  height: 100px;
  border-radius: 20px;
  overflow: hidden;
  border: 1px solid rgba(255,255,255,.16);
  background: linear-gradient(135deg, rgba(59,130,246,.18), rgba(16,185,129,.16));
  box-shadow: 0 18px 45px rgba(0,0,0,.40);
  backdrop-filter: blur(14px);
  transform: translateZ(0);
}

.brand-logo{
  position: relative;
  z-index: 2;
  width: 100%;
  height: 100%;
  object-fit: contain;
  padding: 6px;
  filter: drop-shadow(0 10px 22px rgba(0,0,0,.35)) saturate(1.05) contrast(1.03);
}

.logo-shimmer{
  position:absolute;
  inset:-70%;
  z-index: 1;
  background: conic-gradient(from 180deg at 50% 50%,
    rgba(255,255,255,0),
    rgba(255,255,255,.28),
    rgba(255,255,255,0)
  );
  animation: shimmer 5.5s linear infinite;
  opacity: .55;
  mix-blend-mode: overlay;
}

@keyframes shimmer{
  to { transform: rotate(360deg); }
}

.logo-wrap:hover{
  transform: translateY(-1px) scale(1.02);
  border-color: rgba(255,255,255,.24);
}


:root{
  --bg0: #06162B;
  --bg1: #04121F;
  --txt: rgba(255,255,255,.92);
  --muted: rgba(255,255,255,.68);
  --stroke: rgba(255,255,255,.14);
  --glass: rgba(255,255,255,.06);
  --glass2: rgba(255,255,255,.08);
  --shadow: 0 22px 60px rgba(0,0,0,.45);
  --radius: 22px;

  --blue: 12, 92, 170;
  --green: 0, 140, 72;
}

[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(900px 520px at 18% 16%, rgba(var(--blue), .34), transparent 62%),
    radial-gradient(760px 520px at 88% 22%, rgba(var(--green), .26), transparent 64%),
    radial-gradient(980px 520px at 55% 92%, rgba(var(--blue), .18), transparent 60%),
    linear-gradient(180deg, var(--bg0), var(--bg1));
}

@media (prefers-reduced-motion: reduce){
  .floaty, .shimmer { animation: none !important; }
}

.navbar{
  display:flex; align-items:center; justify-content:space-between;
  padding: 10px 2px 18px 2px;
}
.brand{
  display:flex; align-items:center; gap:10px;
  font-weight: 720; letter-spacing: .2px; color: var(--txt);
}
.dot{
  width:10px; height:10px; border-radius: 999px;
  background: linear-gradient(135deg, rgba(var(--green), 1), rgba(var(--blue), 1));
  box-shadow: 0 0 0 6px rgba(var(--blue), .12);
}
.navlinks a{
  color: var(--muted); text-decoration:none; margin-left: 14px;
  font-size: 13px;
}
.navlinks a:hover{ color: var(--txt); }

.glass{
  background: linear-gradient(180deg, var(--glass2), var(--glass));
  border: 1px solid var(--stroke);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 18px 18px 14px 18px;
  backdrop-filter: blur(14px);
}


.glass:empty{
  display: none !important;
  padding: 0 !important;
  border: 0 !important;
  margin: 0 !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
}


.h1{
  font-size: 40px; line-height: 1.05; margin: 6px 0 8px 0;
  color: var(--txt);
}
.p{
  color: var(--muted); font-size: 14px; line-height: 1.65; margin: 0 0 6px 0;
}

.pill{
  display:inline-flex; align-items:center; gap:8px;
  padding: 6px 10px; border-radius: 999px;
  border: 1px solid var(--stroke);
  color: var(--muted); font-size: 12px;
  background: rgba(255,255,255,.04);
}
.pill strong{ color: var(--txt); font-weight: 650; }

div.stButton > button{
  width: 100%;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,.16);
  background: linear-gradient(135deg, rgba(var(--blue), .92), rgba(var(--green), .92));
  color: white;
  padding: 12px 14px;
  font-weight: 650;
  box-shadow: 0 18px 45px rgba(var(--blue), .22);
}
div.stButton > button:hover{
  filter: brightness(1.05);
  transform: translateY(-1px);
}


[data-testid="stFileUploaderDropzone"]{
  border-radius: 16px;
  border: 1px dashed rgba(255,255,255,.22);
  background: rgba(255,255,255,.03);
}
[data-testid="stSelectbox"] > div{
  border-radius: 14px;
}


.metricbox{
  border-radius: 16px;
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,.03);
  padding: 12px 12px;
}
.smallcap{ color: var(--muted); font-size: 12px; margin-top: 4px; }
.hr{ height: 1px; background: rgba(255,255,255,.10); margin: 14px 0; border-radius: 999px; }


[data-testid="stTextArea"] textarea:disabled{
  -webkit-text-fill-color: rgba(255,255,255,.86) !important;
  color: rgba(255,255,255,.86) !important;
  opacity: 1 !important;
}


img{
  border-radius: 16px !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _navbar() -> None:
    logo = _img_to_data_uri("assets/unb_logo.png")

    st.markdown(
        f"""
<div class="navbar">
  <div class="brand">
    <div class="logo-wrap floaty" title="Universidade de Brasília">
      <img class="brand-logo" src="{logo}" alt="Universidade de Brasília"/>
      <span class="logo-shimmer"></span>
    </div>
    <div class="brandtext">
      <div class="brandtitle"></div>
      <div class="brandsub">Universidade de Brasília</div>
    </div>
  </div>

  <div class="navlinks">
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )



def _normalize_outputs(
    pipeline_name: str,
    img_input: Image.Image,
    out: Dict,
) -> Tuple[List[Tuple[Image.Image, str]], Dict]:
    gallery: List[Tuple[Image.Image, str]] = []
    metrics = out.get("metrics", {}) or {}

    if pipeline_name == "Grad-CAM":
        img1 = out.get("img1")
        img2 = out.get("img2")
        if img1 is not None:
            gallery.append((img1, "Grad-CAM (overlay)"))
        if img2 is not None:
            gallery.append((img2, "Grad-CAM (heatmap)"))
        return gallery, metrics

    base = _resize_rgb(img_input, 256)
    input_gray = base.convert("L").convert("RGB")

    reconstructed = out.get("reconstructed", out.get("img1"))
    deltae = out.get("deltae_heatmap", out.get("img2"))

    gallery.extend(
        [
            (input_gray, "Entrada (cinza)"),
            (reconstructed, "Imagem Reconstruída") if reconstructed is not None else None,
            (deltae, "ΔE2000") if deltae is not None else None,
        ]
    )
    gallery = [x for x in gallery if x is not None]
    return gallery, metrics


def _run_inference(pipeline_name: str, pil_img: Image.Image):
    if pipeline_name == "Grad-CAM":
        out = infer_gradcam(pil_img, weights_path=WEIGHTS_GRADCAM)
        gallery, metrics = _normalize_outputs(pipeline_name, pil_img, out)
        return "OK", gallery, metrics

    if pipeline_name == "Pix2Pix":
        base = _resize_rgb(pil_img, 256)
        out = infer_pix2pix(base, ts_path=WEIGHTS_PIX2PIX_TS)
        gallery, metrics = _normalize_outputs(pipeline_name, pil_img, out)
        return "OK", gallery, metrics

    raise ValueError(f"Pipeline inválido: {pipeline_name}")


    base = _resize_rgb(pil_img, 256)
    out = infer_pix2pix(base, ts_path=WEIGHTS_PIX2PIX_TS)
    gallery, metrics = _normalize_outputs(pipeline_name, pil_img, out)
    return "OK", gallery, metrics


def main() -> None:
    st.set_page_config(page_title="Detecção de Anomalias", layout="wide", initial_sidebar_state="collapsed")
    _inject_css()
    _navbar()

    if "last_metrics" not in st.session_state:
        st.session_state.last_metrics = None


    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="h1">Detecção de Anomalias</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="p">Faça o Upload da imagem e execute o detector <b>Grad-CAM</b> para explicabilidade ou <b>Pix2Pix</b> para reconstrução e mapa <b>ΔE2000</b>.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        pipeline = st.selectbox("Escolha o Pipeline Que Irá Utilizar", ["Grad-CAM", "Pix2Pix"], index=0)
        

        uploaded = st.file_uploader("Upload da imagem", type=["png", "jpg", "jpeg", "webp"])
        infer_btn = st.button("Inferir")

        st.markdown('</div>', unsafe_allow_html=True)

        
    with right:
      st.markdown('<div class="glass">', unsafe_allow_html=True)
      st.markdown('<div id="como-usar" class="h1" style="font-size:22px;">Métricas</div>', unsafe_allow_html=True)

      # placeholders para atualizar depois do clique
      expl_slot = st.empty()
      json_slot = st.empty()

      # renderiza o que já existir no estado (caso o usuário já tenha inferido antes)
      m = st.session_state.get("last_metrics")
      p = st.session_state.get("last_pipeline")

      if isinstance(m, dict) and m:
          if p == "Pix2Pix":
              expl_slot.markdown(
                  f"<div style='text-align:justify; text-justify:inter-word; line-height:1.7; font-size:14px; color:rgba(255,255,255,.82);'>{PIX2PIX_EXPL}</div>",
                  unsafe_allow_html=True
              )
          elif p == "Grad-CAM":
              expl_slot.markdown(
                  f"<div style='text-align:justify; text-justify:inter-word; line-height:1.7; font-size:14px; color:rgba(255,255,255,.82);'>{GRADCAM_EXPL}</div>",
                  unsafe_allow_html=True
              )
          else:
              expl_slot.info("Métricas do pipeline selecionado.")

          json_slot.code(json.dumps(m, indent=2, ensure_ascii=False), language="json")
      else:
          expl_slot.info("Faça a inferência.")


    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:18px;"></div>', unsafe_allow_html=True)
    st.markdown('<div id="resultados"></div>', unsafe_allow_html=True)

    if uploaded is not None:
        pil = Image.open(uploaded).convert("RGB")
    else:
        pil = None

    if infer_btn:
        if pil is None:
            st.error("Carregue uma imagem, para que a inferência possa ser feita.")
            return

        with st.spinner("Inferindo..."):
            try:
                status, gallery, metrics = _run_inference(pipeline, pil)
                st.session_state.last_metrics = metrics
                st.session_state.last_pipeline = pipeline

                if pipeline == "Pix2Pix":
                    expl_slot.markdown(
                        f"<div style='text-align:justify; text-justify:inter-word; line-height:1.7; font-size:14px; color:rgba(255,255,255,.82);'>{PIX2PIX_EXPL}</div>",
                        unsafe_allow_html=True
                    )
                elif pipeline == "Grad-CAM":
                    expl_slot.markdown(
                        f"<div style='text-align:justify; text-justify:inter-word; line-height:1.7; font-size:14px; color:rgba(255,255,255,.82);'>{GRADCAM_EXPL}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    expl_slot.info("Métricas do pipeline selecionado.")


                json_slot.code(json.dumps(metrics, indent=2, ensure_ascii=False), language="json")

            except Exception as e:
                st.exception(e)
                return

        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown(f'<span class="pill">Status: <strong>{status}</strong></span>', unsafe_allow_html=True)
        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

        # Galeria em 3 colunas
        cols = st.columns(3, gap="medium")
        for i, (im, cap) in enumerate(gallery):
            with cols[i % 3]:
                st.image(im, use_container_width=True)
                st.caption(cap)

        
    st.markdown('<div id="sobre"></div>', unsafe_allow_html=True)
    st.markdown(
        """
<div style="height:22px;"></div>
<div class="p" style="text-align:center; opacity:.9;">
  Diagnóstico Visual Por Meio da Detecção de Anomalias
</div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
