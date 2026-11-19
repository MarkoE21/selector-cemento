# -*- coding: utf-8 -*-
"""
Lógica de la app (sin Streamlit): entrenar, recomendar y predecir mezclas nuevas.
Reutiliza las funciones de ml_concreto.py.
"""
import io

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.inspection import permutation_importance

import ml_concreto as ml

EXPOSICION_ETIQUETAS = {
    "ninguna": "Sin exposición especial",
    "sulfatos": "Sulfatos moderados",
    "salitre": "Salitre / sulfatos severos",
}


# ----------------------------------------------------------------------------
# ENTRENAMIENTO (pesado: la app lo guarda en caché)
# ----------------------------------------------------------------------------
def entrenar(fuente: bytes, validacion="mezcla", regla="fcr", n_arboles=300):
    """fuente: contenido (bytes) del Excel. Devuelve un diccionario con todo lo necesario."""
    ml.N_ARBOLES = int(n_arboles)
    hojas = ml.cargar_hojas(io.BytesIO(fuente))
    tests, mix = ml.construir_tablas(hojas)
    if len(tests) < 20 or tests["ID_Cem"].nunique() < 3:
        raise ValueError("La base tiene muy pocos datos para entrenar (mínimo ~20 ensayos y 3 cementos).")

    nulos = tests[ml.FEATURES].isna().sum()
    avisos = []
    if nulos.any():
        avisos.append(f"Variables con valores vacíos (se rellenaron con la mediana): "
                      f"{', '.join(nulos[nulos > 0].index)}")
        tests[ml.FEATURES] = tests[ml.FEATURES].fillna(tests[ml.FEATURES].median())

    res_reg, oof, mejor, modelos = ml.entrenar_evaluar_regresion(tests)
    y = tests["resistencia_MPa"]
    modelo_final = clone(modelos[mejor]).fit(tests[ml.FEATURES], y)

    m28_idx = (tests["edad"] == 28).values
    sigma28 = max(float(np.std(y.values[m28_idx] - oof[validacion][mejor][m28_idx])), 0.5)
    res_cum, m28, clf_final, feats_c = ml.evaluar_cumplimiento(
        tests, oof, mejor, regla, sigma28, validacion)

    pi = permutation_importance(modelo_final, tests[ml.FEATURES], y, n_repeats=15,
                                random_state=ml.SEMILLA, scoring="neg_mean_absolute_error")
    importancia = pd.Series(pi.importances_mean, index=ml.FEATURES).sort_values()

    pred_res = tests[["ID_Mezcla", "cemento", "cantera", "fc_kg", "edad", "resistencia_MPa"]].copy()
    pred_res["pred_cv_mezcla_MPa"] = oof["mezcla"][mejor]
    pred_res["pred_cv_cemento_MPa"] = oof["cemento"][mejor]

    return {
        "hojas": hojas, "tests": tests, "mix": mix, "res_reg": res_reg, "oof": oof,
        "mejor": mejor, "modelo": modelo_final, "sigma28": sigma28, "res_cum": res_cum,
        "m28": m28, "clf": clf_final, "feats_clf": feats_c, "importancia": importancia,
        "pred_res": pred_res, "validacion": validacion, "regla": regla,
        "rango": tests[ml.FEATURES].agg(["min", "max"]).T,
        "n_unicos": tests[ml.FEATURES].nunique(), "avisos": avisos,
    }


# ----------------------------------------------------------------------------
# RECOMENDACIÓN (liviana: se recalcula al mover los controles)
# ----------------------------------------------------------------------------
def nivel_diseno(m28, fc_kg):
    """Nivel de f'c disponible en la base: el menor que sea >= al pedido."""
    niveles = sorted(m28["fc_kg"].unique())
    nivel = next((n for n in niveles if n >= fc_kg), niveles[-1])
    return nivel, niveles


def motivo(row, prob_min, max_desgaste):
    if row["elegible"]:
        return "Elegible"
    m = []
    if not row["ok_exposicion"]:
        m.append("no apto para la exposición")
    if max_desgaste is not None and not row["ok_desgaste"]:
        m.append("desgaste sobre el máximo")
    if not row["ok_resistencia"]:
        m.append(f"P(cumple) {row['prob_regresion']:.0%} < {prob_min:.0%}")
    return "; ".join(m)


def recomendar(R, fc_kg, cantera, exposicion, prob_min, max_desgaste, pesos):
    m28 = R["m28"]
    nivel, _ = nivel_diseno(m28, fc_kg)
    df = m28[(m28["fc_kg"] == nivel) & (m28["cantera"] == cantera)]
    if df.empty:
        raise ValueError("No hay mezclas para esa cantera y ese f'c.")
    rk, ganador = ml.recomendar_escenario(df, exposicion, prob_min, max_desgaste, pesos,
                                          "prob_regresion", "pred28_MPa")
    rk = rk.copy()
    rk["motivo"] = rk.apply(lambda r: motivo(r, prob_min, max_desgaste), axis=1)
    return {"nivel": nivel, "ranking": rk, "ganador": ganador, "escenario": df,
            "hay_ganador": bool(rk["elegible"].any())}


def recomendaciones_globales(R, prob_min, max_desgaste, pesos):
    resumen, detalle = ml.tabla_escenarios(R["m28"], "prob_regresion", "pred28_MPa",
                                           prob_min, max_desgaste, pesos)
    ev, comp = ml.evaluar_recomendacion(R["m28"], prob_min, max_desgaste, pesos, R["sigma28"])
    return {"resumen": resumen, "detalle": detalle, "ev": ev, "comp": comp}


# ----------------------------------------------------------------------------
# MEZCLA NUEVA
# ----------------------------------------------------------------------------
def opciones_materiales(R):
    cem = R["hojas"]["Cementantes"][["ID_Cementante", "Nombre comercial / denominación"]]
    cementos = dict(zip(cem['ID_Cementante'], cem['Nombre comercial / denominación']))
    combos = (R["mix"][["cantera", "ID_AF", "ID_AG"]].drop_duplicates()
              .set_index("cantera").to_dict("index"))
    return cementos, combos


def dosificacion_base(R, id_cem, cantera, fc_kg):
    """Dosificación de la base más parecida (para precargar el formulario)."""
    mix = R["mix"]
    d = mix[(mix["ID_Cem"] == id_cem) & (mix["cantera"] == cantera) & (mix["fc_kg"] == fc_kg)]
    if d.empty:
        d = mix[(mix["ID_Cem"] == id_cem) & (mix["cantera"] == cantera)]
    if d.empty:
        d = mix
    r = d.iloc[0]
    return {"agua": float(r["agua"]), "cem_total": float(r["cem_total"]),
            "aditivo_kg": float(r["aditivo_kg"]), "fino_kg": float(r["fino_kg"]),
            "grueso_kg": float(r["grueso_kg"])}


def predecir_mezcla(R, id_cem, id_af, id_ag, agua, cem_total, aditivo_kg, fino_kg,
                    grueso_kg, fc_req_kg, edades=(3, 7, 28)):
    h = R["hojas"]
    cem = h["Cementantes"].set_index("ID_Cementante").loc[id_cem]
    agr = h["Agregados"].set_index("ID_Agregado")
    af, ag = agr.loc[id_af], agr.loc[id_ag]
    clinker_pct = float(cem["Clínker (%)"])

    filas = []
    for e in edades:
        filas.append({
            "edad": e, "agua": agua, "cem_total": cem_total, "a_mc": agua / cem_total,
            "clinker_kg": cem_total * clinker_pct / 100, "sust_clinker_pct": 100 - clinker_pct,
            "aditivo_kg": aditivo_kg, "ratio_fino": fino_kg / (fino_kg + grueso_kg),
            "puzolana_pct": float(cem["Puzolana (%)"]), "peso_esp_cem": float(cem["Peso específico"]),
            "consistencia_normal": float(cem["Consistencia normal (%)"]),
            "mortero_edad": float(np.interp(e, [3, 7, 28], [
                cem["Resistencia mortero 3 d (MPa)"], cem["Resistencia mortero 7 d (MPa)"],
                cem["Resistencia mortero 28 d (MPa)"]])),
            "mf_fino": float(af["Módulo de finura"]), "abs_fino": float(af["Absorción (%)"]),
            "abs_grueso": float(ag["Absorción (%)"]),
            "la_grueso": float(ag["Desgaste Los Ángeles (%)"]),
        })
    X = pd.DataFrame(filas)[ml.FEATURES]
    pred = R["modelo"].predict(X)
    out = pd.DataFrame({"Edad (días)": list(edades), "Resistencia predicha (MPa)": pred,
                        "Resistencia predicha (kg/cm²)": pred * ml.KGCM2_POR_MPA})

    fcr_kg = float(ml.fcr_kgcm2(fc_req_kg))
    fcr_MPa = fcr_kg / ml.KGCM2_POR_MPA
    p28 = float(pred[list(edades).index(28)]) if 28 in edades else float(pred[-1])
    prob = float(norm.cdf((p28 - fcr_MPa) / R["sigma28"]))

    # huella de carbono (misma fórmula que la hoja Mezclas)
    fe_c = float(cem["Factor de emisión (kg CO₂e/kg)"])
    fe_ag = float(af["Factor de emisión (kg CO₂e/kg)"])
    fe_ad = float(h["Aditivos"]["Factor de emisión (kg CO₂e/kg)"].iloc[0]) if len(h["Aditivos"]) else 0.0
    co2 = cem_total * fe_c + (fino_kg + grueso_kg) * fe_ag + aditivo_kg * fe_ad

    # avisos de extrapolación
    avisos, x28 = [], X.iloc[-1] if 28 not in edades else X.iloc[list(edades).index(28)]
    nombres = {"agua": "agua", "cem_total": "cemento total", "a_mc": "relación a/mc",
               "aditivo_kg": "aditivo", "ratio_fino": "proporción de fino",
               "mf_fino": "módulo de finura", "clinker_kg": "clínker"}
    for k, nom in nombres.items():
        lo, hi = R["rango"].loc[k, "min"], R["rango"].loc[k, "max"]
        if x28[k] < lo - 1e-9 or x28[k] > hi + 1e-9:
            avisos.append(f"**{nom}** ({x28[k]:.2f}) está fuera del rango de la base ({lo:.2f} a {hi:.2f}): "
                          f"la predicción es una extrapolación poco confiable.")
    pocos = [nombres[k] for k in ["agua", "a_mc"] if R["n_unicos"][k] <= 2]
    if pocos:
        avisos.append("En la base casi no varían: " + ", ".join(pocos) +
                      ". El modelo no aprendió bien su efecto; no lo uses para optimizar esas variables.")
    return {"tabla": out, "pred28_MPa": p28, "fcr_kg": fcr_kg, "fcr_MPa": fcr_MPa,
            "prob": prob, "co2": co2, "clinker_kg": float(x28["clinker_kg"]),
            "a_mc": float(x28["a_mc"]), "avisos": avisos}


# ----------------------------------------------------------------------------
# DESCARGAS
# ----------------------------------------------------------------------------
def exportar_excel(R, G):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        R["res_reg"].to_excel(xw, sheet_name="metricas_resistencia", index=False)
        R["pred_res"].to_excel(xw, sheet_name="predicciones_resistencia", index=False)
        R["res_cum"].to_excel(xw, sheet_name="metricas_cumplimiento", index=False)
        R["m28"].drop(columns=["index"], errors="ignore").to_excel(
            xw, sheet_name="predicciones_cumplimiento", index=False)
        G["resumen"].to_excel(xw, sheet_name="recomendacion_escenarios", index=False)
        G["detalle"].drop(columns=["ID_Mezcla"], errors="ignore").to_excel(
            xw, sheet_name="ranking_detallado", index=False)
        G["comp"].to_excel(xw, sheet_name="evaluacion_recomendacion", index=False)
    return buf.getvalue()


def exportar_modelo(R):
    buf = io.BytesIO()
    joblib.dump({"modelo": R["modelo"], "features": ml.FEATURES, "nombre": R["mejor"]}, buf)
    return buf.getvalue()
