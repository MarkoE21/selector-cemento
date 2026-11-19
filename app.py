# -*- coding: utf-8 -*-
"""
Selector de cemento adicionado · Hackathon Innova Sur 2026 · Reto 2 · Yura S.A.

Ejecutar:  streamlit run app.py
"""
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import ml_concreto as ml
import motor

st.set_page_config(page_title="Selector de cemento adicionado", page_icon="🏗️", layout="wide")

RUTA_BASE = Path(__file__).parent / "datos" / "base_mezclas.xlsx"
VELOCIDADES = {"Rápido": 100, "Estándar": 300, "Preciso": 500}
VERDE, ROJO, GRIS, AZUL = "#2e8b57", "#c0392b", "#9aa0a6", "#2f6fb0"


@st.cache_resource(show_spinner="Entrenando modelos… (la primera vez puede tardar hasta un minuto)")
def entrenar_cache(fuente: bytes, validacion: str, regla: str, n_arboles: int):
    return motor.entrenar(fuente, validacion, regla, n_arboles)


# ----------------------------------------------------------------------------
# BARRA LATERAL
# ----------------------------------------------------------------------------
st.sidebar.title("🏗️ Configuración")


@st.cache_data(ttl=600, show_spinner="Descargando datos de Google Sheets…")
def bajar_sheets(enlace: str):
    """Se vuelve a descargar como máximo cada 10 minutos."""
    return motor.descargar_google_sheets(enlace), datetime.now().strftime("%d/%m/%Y %H:%M")


def secreto(nombre):
    try:
        return st.secrets.get(nombre)
    except Exception:  # no hay archivo de secretos configurado
        return None


OPC_BASE, OPC_SUBIR, OPC_SHEETS = ("Base incluida (tesis UCSM 2020)", "Subir Excel actualizado",
                                   "Google Sheets (automático)")
enlace_secreto = secreto("GSHEET_ID")
origen = st.sidebar.radio("Datos", [OPC_BASE, OPC_SUBIR, OPC_SHEETS],
                          index=2 if enlace_secreto else 0)

if origen == OPC_SUBIR:
    archivo = st.sidebar.file_uploader("Excel de la base (mismo formato)", type=["xlsx"])
    if archivo is None:
        st.title("🏗️ Selector de cemento adicionado")
        st.info("Sube el Excel actualizado en la barra lateral, o elige otra fuente de datos.")
        st.stop()
    fuente = archivo.getvalue()
elif origen == OPC_SHEETS:
    enlace = enlace_secreto or st.sidebar.text_input("Enlace de la hoja de Google Sheets")
    if not enlace:
        st.title("🏗️ Selector de cemento adicionado")
        st.info("Pega el enlace de tu Google Sheet en la barra lateral (debe estar compartido como "
                "'Cualquier persona con el enlace').")
        st.stop()
    try:
        fuente, hora = bajar_sheets(enlace)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.sidebar.caption(f"Datos leídos de Google Sheets: {hora}. Se actualizan solos cada 10 minutos.")
    if st.sidebar.button("🔄 Actualizar ahora"):
        bajar_sheets.clear()
        st.rerun()
else:
    if not RUTA_BASE.exists():
        st.error(f"No se encontró {RUTA_BASE}. Copia el Excel en la carpeta 'datos'.")
        st.stop()
    fuente = RUTA_BASE.read_bytes()

with st.sidebar.expander("Opciones avanzadas"):
    validacion = st.selectbox(
        "Cómo se valida el modelo", ["mezcla", "cemento"],
        format_func=lambda v: {"mezcla": "Mezcla nueva (cementos conocidos)",
                               "cemento": "Cemento nuevo (más exigente)"}[v])
    regla = st.selectbox(
        "Criterio de cumplimiento", ["fcr", "fc"],
        format_func=lambda v: {"fcr": "f'cr de la E.060 (recomendado)",
                               "fc": "f'c de diseño"}[v])
    velocidad = st.select_slider("Precisión del modelo", options=list(VELOCIDADES), value="Estándar")

try:
    R = entrenar_cache(fuente, validacion, regla, VELOCIDADES[velocidad])
except Exception as e:  # archivo con otro formato, hojas faltantes, etc.
    st.error(f"No se pudo procesar el Excel: {e}")
    st.stop()

m28, mix = R["m28"], R["mix"]
cantera_opts = sorted(mix["cantera"].unique())
niveles = sorted(m28["fc_kg"].unique())

# ----------------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------------
st.title("🏗️ Selector de cemento adicionado")
st.caption("Predice resistencia, verifica cumplimiento (f'cr, E.060) y recomienda el cemento adicionado "
           "más conveniente frente al Tipo I, según CO₂, costo y exposición.")

fila = R["res_reg"][(R["res_reg"]["validacion"] == validacion) & (R["res_reg"]["modelo"] == R["mejor"])].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mezclas en la base", len(mix))
c2.metric("Cementos / canteras", f"{mix['ID_Cem'].nunique()} / {mix['cantera'].nunique()}")
c3.metric("Error medio a 28 d", f"{fila['MAE_MPa_28d']:.1f} MPa", help="Validación fuera de muestra")
c4.metric("Mezclas que cumplen", f"{100 * m28['cumple_real'].mean():.0f} %",
          help="Criterio de cumplimiento elegido en Opciones avanzadas")
for a in R["avisos"]:
    st.warning(a)

tab_rec, tab_mez, tab_cal, tab_dat, tab_ayu = st.tabs(
    ["🏆 Recomendar cemento", "🧪 Predecir una mezcla", "📊 Calidad del modelo",
     "📁 Datos y descargas", "ℹ️ Ayuda"])

# ----------------------------------------------------------------------------
# 1. RECOMENDAR CEMENTO
# ----------------------------------------------------------------------------
with tab_rec:
    st.subheader("¿Qué cemento adicionado conviene?")
    a, b, c = st.columns(3)
    fc_req = a.number_input("f'c requerido (kg/cm²)", min_value=100, max_value=450, value=210, step=10)
    cantera = b.selectbox("Cantera de agregados", cantera_opts)
    expo = c.selectbox("Exposición", list(motor.EXPOSICION_ETIQUETAS),
                       format_func=lambda k: motor.EXPOSICION_ETIQUETAS[k])

    with st.expander("Criterios de decisión (opcional)"):
        d1, d2 = st.columns(2)
        prob_min = d1.slider("Probabilidad mínima de cumplir f'cr", 0.50, 0.99, 0.80, 0.01)
        usar_desg = d1.checkbox("Limitar el desgaste Los Ángeles del concreto", value=False)
        max_desg = d1.number_input("Desgaste máximo (%)", 10.0, 80.0, 45.0, 1.0) if usar_desg else None
        w_co2 = d2.slider("Peso: reducción de CO₂", 0.0, 1.0, 0.5, 0.05)
        w_cos = d2.slider("Peso: ahorro de costo", 0.0, 1.0, 0.3, 0.05)
        w_res = d2.slider("Peso: margen de resistencia", 0.0, 1.0, 0.2, 0.05)
        suma = w_co2 + w_cos + w_res
        pesos = ({"co2": w_co2 / suma, "costo": w_cos / suma, "res": w_res / suma}
                 if suma > 0 else {"co2": 1 / 3, "costo": 1 / 3, "res": 1 / 3})

    nivel, _ = motor.nivel_diseno(m28, fc_req)
    if nivel != fc_req:
        st.caption(f"ℹ️ La base tiene mezclas de f'c {', '.join(str(int(n)) for n in niveles)} kg/cm². "
                   f"Se usa el nivel **{int(nivel)}** (el menor que cubre {fc_req} kg/cm²).")

    rec = motor.recomendar(R, fc_req, cantera, expo, prob_min, max_desg, pesos)
    rk = rec["ranking"]
    if rec["hay_ganador"]:
        top = rk.iloc[0]
        st.success(f"### ✅ Recomendado: {top['cemento']}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Resistencia predicha 28 d", f"{top['pred28_MPa']:.1f} MPa",
                  f"{top['margen_MPa']:+.1f} sobre el umbral")
        k2.metric("Prob. de cumplir", f"{top['prob_regresion']:.0%}")
        k3.metric("CO₂ vs Tipo I", f"{-top['red_co2_pct']:+.0f} %")
        k4.metric("Costo vs Tipo I", f"{top['dif_costo_pct']:+.0f} %")
    else:
        st.warning("### ⚠️ Ningún cemento adicionado cumple las condiciones.\n"
                   "Se recomienda mantener el Tipo I, o relajar la probabilidad mínima / la exposición.")

    esc = rec["escenario"].copy()
    esc["umbral"] = esc["umbral_MPa"]
    elegibles = set(rk.loc[rk["elegible"], "cemento"])
    esc = esc.sort_values(["tipo_cemento", "cemento"])
    colores = [GRIS if t == "Tipo I" else (VERDE if n in elegibles else ROJO)
               for t, n in zip(esc["tipo_cemento"], esc["cemento"])]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(esc["cemento"].str.replace("Cemento ", "", regex=False), esc["pred28_MPa"], color=colores)
    ax.axhline(esc["umbral"].iloc[0], color="k", ls="--", lw=1.2)
    ax.text(len(esc) - 0.5, esc["umbral"].iloc[0], " umbral", va="bottom", ha="right", fontsize=9)
    ax.set_ylabel("Resistencia predicha 28 d (MPa)")
    ax.set_title(f"f'c {int(nivel)} · {cantera}  (verde = elegible · rojo = descartado · gris = Tipo I)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    tabla = rk[["cemento", "tipo_cemento", "pred28_MPa", "prob_regresion", "red_co2_pct",
                "dif_costo_pct", "score", "ranking", "motivo"]].copy()
    tabla.columns = ["Cemento", "Tipo", "Resist. 28 d (MPa)", "P(cumple)", "Reducción CO₂ (%)",
                     "Dif. costo (%)", "Puntaje", "Ranking", "Estado"]
    st.dataframe(tabla.round(2), hide_index=True)
    st.caption("Reducción de CO₂ y diferencia de costo son frente al promedio de los 3 cementos Tipo I del "
               "mismo escenario. Costo = solo materiales. Los precios 2026 son referenciales.")

    with st.expander("Ver la recomendación para todos los escenarios"):
        G = motor.recomendaciones_globales(R, prob_min, max_desg, pesos)
        res = G["resumen"].copy()
        res["exposicion"] = res["exposicion"].map(motor.EXPOSICION_ETIQUETAS)
        st.dataframe(res.round(2), hide_index=True)
        ev = G["ev"]
        st.write(f"**Validación de la recomendación ({ev['escenarios']} escenarios):** "
                 f"{ev['recomendaciones_que_cumplen_realmente_pct']:.0f} % de los cementos recomendados "
                 f"cumple f'cr en los ensayos reales; coincide con la mejor opción según los datos medidos "
                 f"en {ev['coincidencia_top1_con_datos_reales_pct']:.0f} %.")

# ----------------------------------------------------------------------------
# 2. PREDECIR UNA MEZCLA
# ----------------------------------------------------------------------------
with tab_mez:
    st.subheader("Resistencia de una dosificación")
    st.caption("Se precarga la dosificación de la base más parecida; modifica los valores que quieras.")
    cementos, combos = motor.opciones_materiales(R)
    a, b, c = st.columns(3)
    id_cem = a.selectbox("Cemento", list(cementos), format_func=lambda k: f"{k} · {cementos[k]}")
    cant_m = b.selectbox("Cantera de agregados", list(combos), key="cant_mezcla")
    nivel_ref = c.selectbox("Precargar dosificación de f'c", [int(n) for n in niveles])
    base = motor.dosificacion_base(R, id_cem, cant_m, nivel_ref)
    kk = f"{id_cem}_{cant_m}_{nivel_ref}"  # al cambiar la selección, se recargan los valores

    d1, d2, d3, d4, d5 = st.columns(5)
    agua = d1.number_input("Agua (kg/m³)", 100.0, 300.0, base["agua"], 1.0, key=f"agua_{kk}")
    cem_t = d2.number_input("Cemento (kg/m³)", 200.0, 700.0, base["cem_total"], 5.0, key=f"cem_{kk}")
    adit = d3.number_input("Aditivo (kg/m³)", 0.0, 15.0, base["aditivo_kg"], 0.1, key=f"adit_{kk}")
    fino = d4.number_input("Agregado fino (kg/m³)", 300.0, 1200.0, base["fino_kg"], 5.0, key=f"fino_{kk}")
    grue = d5.number_input("Agregado grueso (kg/m³)", 500.0, 1400.0, base["grueso_kg"], 5.0, key=f"grue_{kk}")
    fc_obj = st.number_input("f'c requerido (kg/cm²)", 100, 450, int(nivel_ref), 10, key="fc_obj")

    P = motor.predecir_mezcla(R, id_cem, combos[cant_m]["ID_AF"], combos[cant_m]["ID_AG"],
                              agua, cem_t, adit, fino, grue, fc_obj)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Resistencia 28 d", f"{P['pred28_MPa']:.1f} MPa", f"{P['pred28_MPa'] * ml.KGCM2_POR_MPA:.0f} kg/cm²",
              delta_color="off")
    k2.metric("f'cr requerido", f"{P['fcr_MPa']:.1f} MPa", f"{P['fcr_kg']:.0f} kg/cm²", delta_color="off")
    k3.metric("Prob. de cumplir", f"{P['prob']:.0%}")
    k4.metric("CO₂ de la mezcla", f"{P['co2']:.0f} kg/m³", f"a/mc {P['a_mc']:.2f}", delta_color="off")
    (st.success if P["prob"] >= 0.8 else st.error if P["prob"] < 0.5 else st.warning)(
        "Cumple f'cr con alta probabilidad." if P["prob"] >= 0.8 else
        "No alcanza f'cr con la dosificación indicada." if P["prob"] < 0.5 else
        "Cumplimiento incierto: revisa la dosificación.")
    for av in P["avisos"]:
        st.info(av)

    t = P["tabla"]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(t["Edad (días)"], t["Resistencia predicha (MPa)"], "o-", color=AZUL, lw=2)
    ax.axhline(P["fcr_MPa"], color="k", ls="--", lw=1)
    ax.text(t["Edad (días)"].max(), P["fcr_MPa"], " f'cr", va="bottom", ha="right", fontsize=9)
    ax.set_xlabel("Edad (días)"); ax.set_ylabel("Resistencia (MPa)"); ax.set_xticks(t["Edad (días)"])
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.dataframe(t.round(1), hide_index=True)

# ----------------------------------------------------------------------------
# 3. CALIDAD DEL MODELO
# ----------------------------------------------------------------------------
with tab_cal:
    st.subheader("¿Qué tan confiable es el modelo?")
    st.markdown(
        f"Modelo elegido: **{R['mejor']}**. Las métricas son **fuera de muestra**: cada mezcla se predice "
        f"sin haber sido vista en el entrenamiento. *Cemento nuevo* es la prueba más dura (el modelo nunca vio ese cemento).")
    rr = R["res_reg"].copy()
    rr["validacion"] = rr["validacion"].map({"mezcla": "Mezcla nueva", "cemento": "Cemento nuevo"})
    st.dataframe(rr.round(3), hide_index=True)

    a, b = st.columns(2)
    with a:
        p = R["oof"][validacion][R["mejor"]]
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        sc = ax.scatter(R["tests"]["resistencia_MPa"], p, c=R["tests"]["edad"], cmap="viridis", s=24)
        lim = [0, max(R["tests"]["resistencia_MPa"].max(), p.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("Observada (MPa)"); ax.set_ylabel("Predicha (MPa)")
        ax.set_title("Observada vs predicha"); fig.colorbar(sc, label="Edad (días)")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with b:
        imp = R["importancia"]
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.barh(imp.index, imp.values, color=AZUL)
        ax.set_xlabel("Aumento del error al permutar (MPa)"); ax.set_title("Variables más influyentes")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown("#### Cumplimiento de f'cr" if regla == "fcr" else "#### Cumplimiento de f'c")
    st.dataframe(R["res_cum"].round(3), hide_index=True)
    if regla == "fc":
        st.warning("Con f'c de diseño todas las mezclas cumplen: no hay clases para entrenar un clasificador.")

    st.markdown("#### Limitaciones")
    st.markdown(
        "- Base pequeña (48 mezclas de una sola tesis): válido para estos cementos y canteras.\n"
        "- Agua constante y solo 2 niveles de a/mc: el modelo no aprendió su efecto.\n"
        "- Composición de varios cementos y precios 2026 son **estimaciones**.\n"
        "- Es una herramienta de apoyo: confirmar con ensayos de laboratorio antes de especificar.")

# ----------------------------------------------------------------------------
# 4. DATOS Y DESCARGAS
# ----------------------------------------------------------------------------
with tab_dat:
    st.subheader("Mezclas de la base")
    vista = mix[["ID_Mezcla", "codigo_fuente", "cemento", "tipo_cemento", "cantera", "fc_kg", "agua",
                 "cem_total", "a_mc", "sust_clinker_pct", "co2", "costo", "medido28_MPa", "abrasion_pct"]].copy()
    vista.columns = ["ID", "Código fuente", "Cemento", "Tipo", "Cantera", "f'c (kg/cm²)", "Agua", "Cemento (kg/m³)",
                     "a/mc", "Sust. clínker (%)", "CO₂ (kg/m³)", "Costo (S/m³)", "f'c medido 28 d (MPa)", "Desgaste (%)"]
    f_tipo = st.multiselect("Filtrar por tipo de cemento", sorted(vista["Tipo"].unique()))
    if f_tipo:
        vista = vista[vista["Tipo"].isin(f_tipo)]
    st.dataframe(vista.round(2), hide_index=True)

    st.subheader("Descargas")
    G0 = motor.recomendaciones_globales(R, 0.8, None, {"co2": 0.5, "costo": 0.3, "res": 0.2})
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Resultados (Excel)", motor.exportar_excel(R, G0), "resultados_ml.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    d2.download_button("⬇️ Modelo de resistencia (.joblib)", motor.exportar_modelo(R), "modelo_resistencia.joblib")

# ----------------------------------------------------------------------------
# 5. AYUDA
# ----------------------------------------------------------------------------
with tab_ayu:
    st.subheader("Cómo funciona")
    st.markdown(
        "1. **Datos:** la app lee las hojas *Mezclas, Ensayos_Resistencia, Cementantes, Agregados, Aditivos y "
        "Ensayos_Durabilidad* del Excel.\n"
        "2. **Modelo:** predice la resistencia a 3, 7 y 28 días con las propiedades de la mezcla y de los "
        "materiales (no usa resultados del propio concreto, para evitar trampa).\n"
        "3. **Cumplimiento:** compara la resistencia predicha con el f'cr de la E.060 "
        "(f'c + 70 / 84 / 98 kg/cm² según el rango de f'c).\n"
        "4. **Recomendación:** entre los cementos adicionados que cumplen con la probabilidad mínima y son aptos "
        "para la exposición, elige el de mejor puntaje de CO₂, costo y margen de resistencia.\n")
    st.subheader("Cómo mejorarla con ensayos nuevos")
    st.markdown(
        "- Agrega filas en el Excel (hojas *Mezclas, Ensayos_Resistencia, Ensayos_Durabilidad*), con las mismas columnas.\n"
        "- En la barra lateral elige **Subir Excel actualizado**. La app reentrena sola.\n"
        "- Mientras más variedad (agua, relación a/c, cementos, canteras), más confiable el modelo.")
    st.subheader("Supuestos a validar con el equipo")
    st.markdown(
        "- Exposición: *salitre* admite solo cemento HS; *sulfatos moderados* admite IP o HS.\n"
        "- Los pesos de CO₂ / costo / resistencia son ajustables y reflejan una prioridad, no una norma.")
