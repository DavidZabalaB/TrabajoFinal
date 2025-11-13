import os
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
from wordcloud import WordCloud
import spacy
import base64
from io import BytesIO
from huggingface_hub import InferenceClient

# ========================
# CONFIGURACIÓN INICIAL
# ========================

# 🔒 Cargar token de entorno o usar fallback local
HF_TOKEN = os.getenv("HF_TOKEN", "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")

if not HF_TOKEN:
    raise ValueError("❌ No se encontró el token de Hugging Face. Configura HF_TOKEN en Render o en local.")

client = InferenceClient(api_key=HF_TOKEN)
MODEL = "tabularisai/multilingual-sentiment-analysis"

# ========================
# CARGA Y LIMPIEZA DE DATOS
# ========================
DATA_PATH = "opinions_classified_ordenado.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ No se encontró el archivo {DATA_PATH}. Asegúrate de subirlo al repositorio.")

df = pd.read_csv(DATA_PATH)

required_cols = ["Opinión", "Deporte", "Sentimiento IA"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Falta la columna requerida: {col}")

# ========================
# PROCESAMIENTO DE TEXTO
# ========================
print("🧠 Procesando texto para nube de palabras...")
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "es_core_news_sm"])
    nlp = spacy.load("es_core_news_sm")

def limpiar_texto(texto):
    doc = nlp(str(texto).lower())
    palabras = [
        token.lemma_ for token in doc
        if not token.is_stop and token.is_alpha and len(token.text) > 2
    ]
    return palabras

df["tokens"] = df["Opinión"].apply(limpiar_texto)
todas_palabras = [pal for sublist in df["tokens"] for pal in sublist]
freq = pd.Series(todas_palabras).value_counts().head(10)

# Nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color="#F9FAFB", colormap="viridis").generate(" ".join(todas_palabras))
buffer = BytesIO()
wordcloud.to_image().save(buffer, format="PNG")
img_wordcloud = base64.b64encode(buffer.getvalue()).decode()

# ========================
# APP DASH
# ========================
app = Dash(__name__)
server = app.server  # 👈 necesario para Render
app.title = "Análisis de Sentimientos - París 2024"

# Estilos globales
estilos = {
    "container": {'backgroundColor': '#F4F6F8', 'fontFamily': 'Arial, sans-serif', 'padding': '30px'},
    "card": {
        'backgroundColor': 'white',
        'borderRadius': '15px',
        'boxShadow': '0 4px 10px rgba(0, 0, 0, 0.1)',
        'padding': '25px',
        'marginBottom': '30px'
    },
    "titulo": {'textAlign': 'center', 'color': '#1A5276'},
    "subtitulo": {'textAlign': 'center', 'color': '#154360'},
    "boton": {'backgroundColor': '#117A65', 'color': 'white', 'borderRadius': '10px', 'padding': '10px 20px', 'border': 'none'}
}

# ========================
# LAYOUT
# ========================
app.layout = html.Div(style=estilos["container"], children=[
    html.H1("📊 Análisis de Sentimientos - París 2024 🏅", style=estilos["titulo"]),
    html.Hr(),

    html.Div(style=estilos["card"], children=[
        html.H3("Distribución de sentimientos por deporte", style=estilos["subtitulo"]),
        dcc.Dropdown(
            options=[{"label": dep, "value": dep} for dep in df["Deporte"].unique()],
            id="dropdown-deporte",
            placeholder="Selecciona un deporte o deja vacío para ver todos",
            style={'width': '50%', 'margin': 'auto'}
        ),
        html.Br(),
        dcc.Graph(id="grafico-sentimientos")
    ]),

    html.Div(style=estilos["card"], children=[
        html.H3("☁️ Nube de palabras más frecuentes", style=estilos["subtitulo"]),
        html.Img(src="data:image/png;base64," + img_wordcloud,
                 style={'width': '70%', 'display': 'block', 'margin': 'auto', 'borderRadius': '15px'}),
        html.Br(),
        dcc.Graph(
            figure=px.bar(
                x=freq.index, y=freq.values,
                title="🔟 Palabras más frecuentes en opiniones",
                labels={"x": "Palabra", "y": "Frecuencia"},
                color=freq.values, color_continuous_scale="tealgrn"
            ).update_layout(plot_bgcolor='white', paper_bgcolor='#F9FAFB')
        )
    ]),

    html.Div(style=estilos["card"], children=[
        html.H3("💬 Clasificador de nueva opinión", style=estilos["subtitulo"]),
        dcc.Textarea(
            id="nueva-opinion",
            placeholder="✍️ Escribe una nueva opinión sobre París 2024...",
            style={'width': '100%', 'height': 100, 'borderRadius': '10px', 'padding': '10px', 'border': '1px solid #B2BABB'}
        ),
        html.Br(),
        html.Button("Clasificar sentimiento", id="btn-clasificar", n_clicks=0, style=estilos["boton"]),
        html.Div(id="resultado-clasificacion", style={'marginTop': 20, 'fontSize': '18px', 'textAlign': 'center', 'color': '#0B5345'})
    ]),

    html.Div(style=estilos["card"], children=[
        html.H3("📋 Vista detallada de opiniones clasificadas", style=estilos["subtitulo"]),
        dcc.Graph(id="tabla-opiniones")
    ])
])

# ========================
# CALLBACKS
# ========================
@app.callback(
    Output("grafico-sentimientos", "figure"),
    Input("dropdown-deporte", "value")
)
def actualizar_sentimientos(deporte_sel):
    if deporte_sel:
        df_filtrado = df[df["Deporte"] == deporte_sel]
    else:
        df_filtrado = df.copy()

    conteo = (
        df_filtrado.groupby(["Deporte", "Sentimiento IA"])
        .size()
        .reset_index(name="Cantidad")
    )

    fig = px.bar(
        conteo, x="Deporte", y="Cantidad", color="Sentimiento IA",
        barmode="group",
        title="Distribución de sentimientos IA por deporte",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='#F9FAFB', title_font_color='#1A5276')
    return fig


@app.callback(
    Output("tabla-opiniones", "figure"),
    Input("dropdown-deporte", "value")
)
def actualizar_tabla(deporte_sel):
    if deporte_sel:
        df_filtrado = df[df["Deporte"] == deporte_sel]
    else:
        df_filtrado = df.copy()

    fig = px.scatter(
        df_filtrado,
        x="Nivel de confianza",
        y="Sentimiento IA",
        color="Deporte",
        hover_data=["Opinión", "Sentimiento original"],
        title="Opiniones clasificadas por IA",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='#F9FAFB', title_font_color='#1A5276')
    return fig


@app.callback(
    Output("resultado-clasificacion", "children"),
    Input("btn-clasificar", "n_clicks"),
    State("nueva-opinion", "value")
)
def clasificar_opinion(n_clicks, texto):
    if n_clicks > 0 and texto:
        try:
            result = client.text_classification(texto, model=MODEL)
            label = result[0]["label"]
            score = result[0]["score"]
            return f"🧩 Sentimiento detectado: {label} (confianza: {score:.2f})"
        except Exception as e:
            return f"❌ Error al clasificar: {e}"
    return ""

# ========================
# SERVIDOR PARA RENDER
# ========================

app = Dash(__name__)
server = app.server  # <-- Render usa esto para Gunicorn

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))

