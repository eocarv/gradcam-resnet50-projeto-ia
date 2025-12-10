# gradcam-resnet50-projeto-ia
Segundo projeto de IIA do professor Díbio

Este projeto implementa a Pix2pix para a reconstrução de cores em pix para detectar e analisar anomalias em folhas.

## 📌 Objetivos
- Utilizar *ResNet-18 + Grad-CAM* para localizar regiões de interesse.
- Aplicar *Pix2Pix* para reconstrução de imagens.
- Avaliar a qualidade das reconstruções com métricas perceptuais *ΔE2000*.

## 📂 Estrutura
- notebooks/ → testes e experimentos em Jupyter Notebook.
- src/ → código fonte organizado.
- results/ → imagens e métricas geradas.
- relatorio.tex → arquivo LaTeX com resultados e discussão.
- requirements.txt → dependências do projeto.

## ⚙️ Como rodar
```bash
git clone https://github.com/eocarv/gradcam-resnet50-projeto-ia.git
cd gradcam-resnet50-projeto-ia
pip install -r requirements.txt
