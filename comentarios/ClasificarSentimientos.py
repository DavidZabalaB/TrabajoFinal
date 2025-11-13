import os
import pandas as pd
from huggingface_hub import InferenceClient
from tqdm import tqdm
import matplotlib.pyplot as plt

# ========================
# CONFIGURACIÓN
# ========================

HF_TOKEN = "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"  # 🔑 tu token de Hugging Face
os.environ["HF_TOKEN"] = HF_TOKEN

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

MODEL = "tabularisai/multilingual-sentiment-analysis"

# ========================
# LECTURA DEL CSV ORIGINAL
# ========================
csv_path = "opiniones_paris2024.csv"
df = pd.read_csv(csv_path)
df["sentiment_predicted"] = ""

# ========================
# CLASIFICACIÓN DE SENTIMIENTOS
# ========================
print("🔍 Analizando sentimientos...")

for i, text in tqdm(enumerate(df["opinion"]), total=len(df)):
    try:
        result = client.text_classification(text, model=MODEL)
        label = result[0]["label"]
        score = result[0]["score"]
        df.loc[i, "sentiment_predicted"] = f"{label} ({score:.2f})"
    except Exception as e:
        print(f"⚠️ Error en la fila {i}: {e}")
        df.loc[i, "sentiment_predicted"] = "error"

# ========================
# PROCESAMIENTO DE RESULTADOS
# ========================
def separar_sentimiento(valor):
    try:
        sentimiento, score = valor.split("(")
        sentimiento = sentimiento.strip()
        score = float(score.replace(")", ""))
        return pd.Series([sentimiento, score])
    except:
        return pd.Series(["Desconocido", None])

df[["sentimiento_predicho", "nivel_confianza"]] = df["sentiment_predicted"].apply(separar_sentimiento)

traduccion = {
    "Very Positive": "Muy positivo",
    "Positive": "Positivo",
    "Neutral": "Neutral",
    "Negative": "Negativo",
    "Very Negative": "Muy negativo"
}
df["sentimiento_predicho"] = df["sentimiento_predicho"].replace(traduccion)

def coincidencia(row):
    try:
        orig = row["sentiment"].lower()
        pred = row["sentimiento_predicho"].lower()
        if "positivo" in pred and "positiva" in orig:
            return "Sí"
        if "negativo" in pred and "negativa" in orig:
            return "Sí"
        if "neutral" in pred and "neutral" in orig:
            return "Sí"
        return "No"
    except:
        return "Desconocido"

df["Coincide con IA"] = df.apply(coincidencia, axis=1)

# ========================
# LIMPIEZA Y ORDEN FINAL
# ========================
df = df[[
    "id",
    "opinion",
    "deporte",
    "sentiment",
    "sentimiento_predicho",
    "nivel_confianza",
    "Coincide con IA",
    "source"
]]

df = df.rename(columns={
    "id": "ID",
    "opinion": "Opinión",
    "deporte": "Deporte",
    "sentiment": "Sentimiento original",
    "sentimiento_predicho": "Sentimiento IA",
    "nivel_confianza": "Nivel de confianza",
    "source": "Fuente"
})

df = df.sort_values(by=["Deporte", "Sentimiento IA"], ascending=[True, True])

# ========================
# GUARDAR CSV FINAL
# ========================
output_path = "opinions_classified_ordenado.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✅ Archivo limpio y ordenado guardado como: {output_path}")

# ========================
# ANÁLISIS Y VISUALIZACIÓN
# ========================
print("\n📊 Generando análisis estadístico y visualizaciones...")

resumen_final = (
    df.groupby("Deporte")["Coincide con IA"]
    .apply(lambda x: (x == "Sí").mean() * 100)
    .reset_index(name="Coincidencia (%)")
)

sentimientos_por_deporte = (
    df.groupby(["Deporte", "Sentimiento IA"]).size().unstack(fill_value=0)
)

# Mostrar resumen en consola
print("\n📈 Coincidencia IA vs Sentimiento original por deporte:\n")
print(resumen_final.to_string(index=False, formatters={"Coincidencia (%)": "{:.1f}".format}))

# Guardar CSV resumen
resumen_final.to_csv("resumen_sentimientos.csv", index=False, encoding="utf-8-sig")

# Gráfico 1
plt.figure(figsize=(8, 5))
plt.barh(resumen_final["Deporte"], resumen_final["Coincidencia (%)"], color="#4CAF50")
plt.xlabel("Coincidencia con IA (%)")
plt.ylabel("Deporte")
plt.title("Precisión de coincidencia por deporte")
plt.tight_layout()
plt.savefig("coincidencia_por_deporte.png")
plt.close()

# Gráfico 2
sentimientos_por_deporte.plot(kind="bar", figsize=(10, 6))
plt.title("Distribución de sentimientos por deporte")
plt.xlabel("Deporte")
plt.ylabel("Cantidad de opiniones")
plt.legend(title="Sentimiento IA")
plt.tight_layout()
plt.savefig("sentimientos_por_deporte.png")
plt.close()

print("\n✅ Análisis y gráficos generados con éxito:")
print("   - resumen_sentimientos.csv")
print("   - coincidencia_por_deporte.png")
print("   - sentimientos_por_deporte.png")
