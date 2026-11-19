#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML para mezclas de concreto con cementos adicionados
=====================================================
Hackathon Innova Sur 2026 · Reto 2 · Yura S.A.

Tres objetivos, en este orden:
  1) Predecir la resistencia a compresión (MPa) a 3, 7 y 28 días.
  2) Predecir si la mezcla CUMPLE (f'cr de la E.060 por defecto, o f'c).
  3) Recomendar qué cemento adicionado conviene frente al Tipo I
     (cumplimiento + reducción de CO2 + costo + exposición).

Uso básico:
    python ml_concreto.py --excel Base_de_Datos_Mezclas_Concreto_Industrial1_VERSION_FINAL__1_.xlsx

Recomendación puntual:
    python ml_concreto.py --excel base.xlsx --fc 280 --cantera "La Poderosa" --exposicion salitre

Opciones útiles: --validacion {mezcla,cemento}  --regla {fcr,fc}  --prob-min 0.8
                 --max-desgaste 45  --w-co2 0.5 --w-costo 0.3 --w-res 0.2

Dependencias: pandas, numpy, scikit-learn, scipy, matplotlib, openpyxl, joblib
"""

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------
EXCEL_DEFECTO = "Base_de_Datos_Mezclas_Concreto_Industrial1_VERSION_FINAL__1_.xlsx"
SALIDA_DEFECTO = "resultados_ml"
SEMILLA = 42
N_ARBOLES = 500          # la app lo cambia según el modo "Rápido / Estándar / Preciso"
KGCM2_POR_MPA = 10.1972

# Variables de entrada del modelo. NO incluye resultados de ensayos del propio
# concreto (f'c medido, abrasión del concreto, asentamiento medido) para evitar
# fuga de información. Sí incluye propiedades de los materiales (ficha del cemento,
# ensayos del agregado), que se conocen antes de fabricar la mezcla.
FEATURES = [
    "edad", "agua", "cem_total", "a_mc", "clinker_kg", "sust_clinker_pct",
    "aditivo_kg", "ratio_fino", "puzolana_pct", "peso_esp_cem",
    "consistencia_normal", "mortero_edad", "mf_fino", "abs_fino",
    "abs_grueso", "la_grueso",
]

# Exposición -> tipos de cemento admisibles (SUPUESTO: validar con Civil 2, tarea 3).
EXPOSICIONES = {
    "ninguna": ["Tipo IP", "Tipo HS"],
    "sulfatos": ["Tipo IP", "Tipo HS"],   # sulfatos moderados
    "salitre": ["Tipo HS"],               # sulfatos severos / salitre
}


def fcr_kgcm2(fc):
    """f'cr (kgf/cm²) sin desviación estándar histórica, según E.060."""
    fc = np.asarray(fc, dtype=float)
    return np.where(fc < 210, fc + 70, np.where(fc <= 350, fc + 84, fc + 98))


# ----------------------------------------------------------------------------
# 1. CARGA Y CONSTRUCCIÓN DE LA TABLA
# ----------------------------------------------------------------------------
def cargar_hojas(ruta):
    xl = pd.ExcelFile(ruta)
    nombres = ["Mezclas", "Ensayos_Resistencia", "Cementantes", "Agregados",
               "Aditivos", "Ensayos_Durabilidad"]
    faltan = [n for n in nombres if n not in xl.sheet_names]
    if faltan:
        raise ValueError(f"Faltan hojas en el Excel: {faltan}")
    return {n: xl.parse(n) for n in nombres}


def construir_tablas(h):
    """Devuelve (tests, mix): tests = una fila por ensayo de compresión,
    mix = una fila por mezcla."""
    mez, ens, cem, agr, dur = (h[k] for k in
                               ["Mezclas", "Ensayos_Resistencia", "Cementantes",
                                "Agregados", "Ensayos_Durabilidad"])

    mix = mez[[
        "ID_Mezcla", "Código en la fuente", "ID_Cementante_1", "ID_Agregado_fino",
        "ID_Agregado_grueso", "Cantera", "Tipo de cemento", "f'c de diseño",
        "Agua (kg/m³)", "Material cementante total (kg/m³)", "Relación a/mc",
        "Clínker (kg/m³)", "Sustitución de clínker (%)", "Aditivo 1 (kg/m³)",
        "Agregado fino (kg/m³)", "Agregado grueso (kg/m³)",
        "Huella de carbono (kg CO₂e/m³)", "Costo de materiales (S/ por m³)",
        "f'c medido 28 d (MPa)"]].copy()
    mix.columns = [
        "ID_Mezcla", "codigo_fuente", "ID_Cem", "ID_AF", "ID_AG", "cantera",
        "tipo_cemento", "fc_kg", "agua", "cem_total", "a_mc", "clinker_kg",
        "sust_clinker_pct", "aditivo_kg", "fino_kg", "grueso_kg", "co2", "costo",
        "medido28_MPa"]
    mix["aditivo_kg"] = mix["aditivo_kg"].fillna(0.0)
    mix["ratio_fino"] = mix["fino_kg"] / (mix["fino_kg"] + mix["grueso_kg"])

    # Propiedades del cemento (ficha técnica)
    c = cem[["ID_Cementante", "Nombre comercial / denominación", "Puzolana (%)",
             "Peso específico", "Consistencia normal (%)",
             "Resistencia mortero 3 d (MPa)", "Resistencia mortero 7 d (MPa)",
             "Resistencia mortero 28 d (MPa)"]].copy()
    c.columns = ["ID_Cem", "cemento", "puzolana_pct", "peso_esp_cem",
                 "consistencia_normal", "mort3", "mort7", "mort28"]
    mix = mix.merge(c, on="ID_Cem", how="left")

    # Propiedades de los agregados (ensayos del material)
    af = agr[["ID_Agregado", "Módulo de finura", "Absorción (%)"]].copy()
    af.columns = ["ID_AF", "mf_fino", "abs_fino"]
    ag = agr[["ID_Agregado", "Absorción (%)", "Desgaste Los Ángeles (%)"]].copy()
    ag.columns = ["ID_AG", "abs_grueso", "la_grueso"]
    mix = mix.merge(af, on="ID_AF", how="left").merge(ag, on="ID_AG", how="left")

    # Abrasión observada del concreto: solo para filtrar la recomendación (no es feature)
    d = dur[dur["Tipo de ensayo"].str.contains("abrasi", case=False, na=False)]
    d = d.groupby("ID_Mezcla", as_index=False)["Valor"].mean()
    d.columns = ["ID_Mezcla", "abrasion_pct"]
    mix = mix.merge(d, on="ID_Mezcla", how="left")

    mix["medido28_kg"] = mix["medido28_MPa"] * KGCM2_POR_MPA
    mix["fcr_kg"] = fcr_kgcm2(mix["fc_kg"])
    mix["fcr_MPa"] = mix["fcr_kg"] / KGCM2_POR_MPA
    mix["fc_MPa"] = mix["fc_kg"] / KGCM2_POR_MPA

    # Ensayos de compresión por edad
    e = ens[ens["Tipo de ensayo"] == "Compresión"][
        ["ID_Mezcla", "Edad (días)", "Resistencia (MPa)"]].copy()
    e.columns = ["ID_Mezcla", "edad", "resistencia_MPa"]
    tests = e.merge(mix, on="ID_Mezcla", how="left")
    # resistencia del mortero del cemento a la misma edad (interpolada)
    tests["mortero_edad"] = [np.interp(a, [3, 7, 28], [m3, m7, m28]) for a, m3, m7, m28
                             in zip(tests["edad"], tests["mort3"], tests["mort7"],
                                    tests["mort28"])]
    return tests, mix


# ----------------------------------------------------------------------------
# 2. MODELOS Y VALIDACIÓN
# ----------------------------------------------------------------------------
def modelos_regresion():
    return {
        "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "RandomForest": RandomForestRegressor(N_ARBOLES, min_samples_leaf=2, random_state=SEMILLA, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(N_ARBOLES, min_samples_leaf=2, random_state=SEMILLA, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=SEMILLA),
    }


def oof_predict(modelo, X, y, grupos, esquema, n_splits=6):
    """Predicción fuera de muestra.
    esquema='mezcla'  : ninguna mezcla aparece en entrenamiento y prueba a la vez.
    esquema='cemento' : se deja fuera TODO un cemento (prueba de cemento nuevo)."""
    if esquema == "mezcla":
        cv = GroupKFold(n_splits=min(n_splits, grupos.nunique()))
    else:
        cv = LeaveOneGroupOut()
    return cross_val_predict(clone(modelo), X, y, groups=grupos, cv=cv)


def metricas_reg(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"R2": r2_score(y, p), "MAE_MPa": mean_absolute_error(y, p),
            "RMSE_MPa": float(np.sqrt(np.mean((y - p) ** 2)))}


def entrenar_evaluar_regresion(tests):
    X, y = tests[FEATURES], tests["resistencia_MPa"]
    grupos = {"mezcla": tests["ID_Mezcla"], "cemento": tests["ID_Cem"]}
    modelos = modelos_regresion()
    filas, oof = [], {"mezcla": {}, "cemento": {}}
    for esq in ["mezcla", "cemento"]:
        for nombre, m in modelos.items():
            p = oof_predict(m, X, y, grupos[esq], esq)
            oof[esq][nombre] = p
            r_all = metricas_reg(y, p)
            m28 = tests["edad"] == 28
            r_28 = metricas_reg(y[m28], p[m28])
            filas.append({"validacion": esq, "modelo": nombre,
                          **{f"{k}_todas": v for k, v in r_all.items()},
                          **{f"{k}_28d": v for k, v in r_28.items()}})
    res = pd.DataFrame(filas)
    sub = res[res["validacion"] == "mezcla"].sort_values("MAE_MPa_todas")
    mejor = sub.iloc[0]["modelo"]
    return res, oof, mejor, modelos


# ----------------------------------------------------------------------------
# 3. CUMPLIMIENTO
# ----------------------------------------------------------------------------
def evaluar_cumplimiento(tests, oof, mejor, regla, sigma28, esquema_clf):
    """Una fila por mezcla (a 28 d). Compara tres predictores de cumplimiento."""
    t28 = tests[tests["edad"] == 28].copy().reset_index()
    t28["pred28_MPa"] = oof[esquema_clf][mejor][t28["index"]]
    umbral_kg = t28["fcr_kg"] if regla == "fcr" else t28["fc_kg"]
    umbral_MPa = umbral_kg / KGCM2_POR_MPA
    t28["umbral_MPa"] = umbral_MPa
    t28["cumple_real"] = (t28["medido28_kg"] >= umbral_kg).astype(int)
    t28["prob_regresion"] = norm.cdf((t28["pred28_MPa"] - umbral_MPa) / sigma28)
    t28["cumple_regresion"] = (t28["pred28_MPa"] >= umbral_MPa).astype(int)

    y = t28["cumple_real"].values
    filas = []

    def reg_fila(nombre, pred, prob):
        auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else np.nan
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        filas.append({"predictor": nombre, "exactitud": accuracy_score(y, pred),
                      "exactitud_balanceada": balanced_accuracy_score(y, pred),
                      "precision": precision_score(y, pred, zero_division=0),
                      "recall": recall_score(y, pred, zero_division=0),
                      "F1": f1_score(y, pred, zero_division=0), "AUC": auc,
                      "VN": tn, "FP": fp, "FN": fn, "VP": tp})

    base = max(y.mean(), 1 - y.mean())
    filas.append({"predictor": "Línea base (clase mayoritaria)", "exactitud": base})
    reg_fila("Regresión → umbral", t28["cumple_regresion"].values, t28["prob_regresion"].values)

    if len(np.unique(y)) > 1:
        feats_c = [f for f in FEATURES if f != "edad"] + ["fcr_MPa", "fc_MPa"]
        Xc = t28[feats_c]
        grupos = t28["ID_Mezcla"] if esquema_clf == "mezcla" else t28["ID_Cem"]
        cv = (GroupKFold(n_splits=6) if esquema_clf == "mezcla" else LeaveOneGroupOut())
        clfs = {
            "Logística": make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=2000)),
            "RandomForest clasif.": RandomForestClassifier(
                N_ARBOLES, min_samples_leaf=2, class_weight="balanced_subsample",
                random_state=SEMILLA, n_jobs=-1),
        }
        for nombre, c in clfs.items():
            prob = cross_val_predict(clone(c), Xc, y, groups=grupos, cv=cv,
                                     method="predict_proba")[:, 1]
            t28[f"prob_{nombre}"] = prob
            reg_fila(nombre, (prob >= 0.5).astype(int), prob)
        # clasificador final (se guarda)
        clf_final = clone(clfs["RandomForest clasif."]).fit(Xc, y)
    else:
        print("  ! Solo hay una clase en 'cumple_real' con esta regla; se omiten los clasificadores.")
        feats_c, clf_final = None, None
    return pd.DataFrame(filas), t28, clf_final, feats_c


# ----------------------------------------------------------------------------
# 4. RECOMENDACIÓN DE CEMENTO
# ----------------------------------------------------------------------------
def _norm(x):
    x = np.asarray(x, float)
    return np.ones_like(x) if np.ptp(x) == 0 else (x - x.min()) / np.ptp(x)


def recomendar_escenario(df, exposicion, prob_min, max_desgaste, pesos,
                         col_prob="prob", col_pred="pred28_MPa"):
    """df: las 8 mezclas de un mismo escenario (f'c, cantera).
    Devuelve el ranking de cementos adicionados y la decisión."""
    ref = df[df["tipo_cemento"] == "Tipo I"]
    co2_ref, costo_ref = ref["co2"].mean(), ref["costo"].mean()
    add = df[df["tipo_cemento"] != "Tipo I"].copy()
    add["red_co2_pct"] = (co2_ref - add["co2"]) / co2_ref * 100
    add["dif_costo_pct"] = (add["costo"] - costo_ref) / costo_ref * 100
    add["margen_MPa"] = add[col_pred] - add["umbral_MPa"]

    add["ok_exposicion"] = add["tipo_cemento"].isin(EXPOSICIONES[exposicion])
    add["ok_desgaste"] = True if max_desgaste is None else (add["abrasion_pct"] <= max_desgaste)
    add["ok_resistencia"] = add[col_prob] >= prob_min
    add["elegible"] = add["ok_exposicion"] & add["ok_desgaste"] & add["ok_resistencia"]

    add["score"] = np.nan
    el = add["elegible"]
    if el.any():
        s = (pesos["co2"] * _norm(add.loc[el, "red_co2_pct"])
             + pesos["costo"] * _norm(-add.loc[el, "dif_costo_pct"])
             + pesos["res"] * _norm(add.loc[el, "margen_MPa"]))
        add.loc[el, "score"] = s
    add = add.sort_values(["elegible", "score"], ascending=[False, False])
    add["ranking"] = np.where(add["elegible"], np.arange(1, len(add) + 1), np.nan)
    ganador = add.iloc[0]["cemento"] if el.any() else "Mantener Tipo I (ningún adicionado elegible)"
    return add, ganador


def tabla_escenarios(m28, col_prob, col_pred, prob_min, max_desgaste, pesos):
    """Recomendación para todas las combinaciones (f'c, cantera, exposición)."""
    resumen, detalle = [], []
    for (fc, cant), df in m28.groupby(["fc_kg", "cantera"]):
        for expo in EXPOSICIONES:
            rk, gan = recomendar_escenario(df, expo, prob_min, max_desgaste, pesos,
                                           col_prob, col_pred)
            fila = rk.iloc[0] if rk["elegible"].any() else None
            resumen.append({
                "fc_kg_cm2": fc, "cantera": cant, "exposicion": expo, "recomendado": gan,
                "tipo": fila["tipo_cemento"] if fila is not None else "Tipo I",
                "pred_28d_MPa": fila[col_pred] if fila is not None else np.nan,
                "prob_cumple": fila[col_prob] if fila is not None else np.nan,
                "reduccion_CO2_pct": fila["red_co2_pct"] if fila is not None else 0.0,
                "dif_costo_pct": fila["dif_costo_pct"] if fila is not None else 0.0,
                "n_elegibles": int(rk["elegible"].sum()),
            })
            r = rk.copy()
            r["fc_kg_cm2"], r["cantera"], r["exposicion"] = fc, cant, expo
            detalle.append(r)
    return pd.DataFrame(resumen), pd.concat(detalle, ignore_index=True)


def evaluar_recomendacion(m28, prob_min, max_desgaste, pesos, sigma28):
    """Compara la recomendación del modelo con la que saldría con los datos medidos."""
    m = m28.copy()
    orac = m.copy()
    orac["prob_o"] = m["cumple_real"].astype(float)
    orac["pred_o"] = m["medido28_MPa"]
    res_o, _ = tabla_escenarios(orac, "prob_o", "pred_o", 0.5, max_desgaste, pesos)
    res_m, _ = tabla_escenarios(m, "prob_regresion", "pred28_MPa", prob_min, max_desgaste, pesos)
    comp = res_m.merge(res_o, on=["fc_kg_cm2", "cantera", "exposicion"], suffixes=("_modelo", "_real"))
    coincide = (comp["recomendado_modelo"] == comp["recomendado_real"])

    # ¿El cemento recomendado por el modelo realmente cumple?
    def cumple_real(row):
        if row["tipo_modelo"] == "Tipo I":
            return True
        d = m[(m["fc_kg"] == row["fc_kg_cm2"]) & (m["cantera"] == row["cantera"])
              & (m["cemento"] == row["recomendado_modelo"])]
        return bool(d["cumple_real"].iloc[0])
    comp["recomendacion_cumple_realmente"] = comp.apply(cumple_real, axis=1)
    return {
        "escenarios": len(comp),
        "coincidencia_top1_con_datos_reales_pct": 100 * coincide.mean(),
        "recomendaciones_que_cumplen_realmente_pct": 100 * comp["recomendacion_cumple_realmente"].mean(),
        "veces_recomienda_Tipo_I_pct": 100 * (comp["tipo_modelo"] == "Tipo I").mean(),
    }, comp


# ----------------------------------------------------------------------------
# 5. GRÁFICOS
# ----------------------------------------------------------------------------
def graficos(salida, tests, oof, mejor, res_reg, modelo_final, m28):
    # a) observado vs predicho
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, esq in zip(ax, ["mezcla", "cemento"]):
        p = oof[esq][mejor]
        sc = a.scatter(tests["resistencia_MPa"], p, c=tests["edad"], cmap="viridis", s=26)
        lim = [0, max(tests["resistencia_MPa"].max(), p.max()) * 1.05]
        a.plot(lim, lim, "k--", lw=1)
        r = metricas_reg(tests["resistencia_MPa"], p)
        a.set_title(f"{mejor} · validación por {esq}\nR²={r['R2']:.2f} · MAE={r['MAE_MPa']:.2f} MPa")
        a.set_xlabel("Observada (MPa)"); a.set_ylabel("Predicha fuera de muestra (MPa)")
    fig.colorbar(sc, ax=ax, label="Edad (días)")
    fig.savefig(salida / "fig1_observado_vs_predicho.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    # b) MAE por modelo
    piv = res_reg.pivot(index="modelo", columns="validacion", values="MAE_MPa_todas")
    ax = piv.plot.bar(figsize=(7, 4), rot=0)
    ax.set_ylabel("MAE (MPa)"); ax.set_title("Error por modelo y tipo de validación")
    plt.tight_layout(); plt.savefig(salida / "fig2_mae_modelos.png", dpi=150); plt.close()

    # c) importancia por permutación
    pi = permutation_importance(modelo_final, tests[FEATURES], tests["resistencia_MPa"],
                                n_repeats=20, random_state=SEMILLA,
                                scoring="neg_mean_absolute_error")
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    ax = imp.plot.barh(figsize=(7, 5))
    ax.set_xlabel("Aumento del MAE al permutar (MPa)")
    ax.set_title("Importancia de variables (modelo final, datos de entrenamiento)")
    plt.tight_layout(); plt.savefig(salida / "fig3_importancia.png", dpi=150); plt.close()

    # d) mapa de probabilidad de cumplimiento
    m = m28.copy()
    m["escenario"] = m["fc_kg"].astype(int).astype(str) + " · " + m["cantera"]
    hm = m.pivot_table(index="cemento", columns="escenario", values="prob_regresion")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(hm.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns, rotation=40, ha="right")
    ax.set_yticks(range(hm.shape[0])); ax.set_yticklabels(hm.index)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            ax.text(j, i, f"{hm.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="P(cumple)"); ax.set_title("Probabilidad predicha de cumplimiento a 28 d")
    fig.savefig(salida / "fig4_prob_cumplimiento.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------
# 6. USO DEL MODELO CON MEZCLAS NUEVAS
# ----------------------------------------------------------------------------
def predecir_resistencia(paquete, df_nuevo):
    """paquete = joblib.load('modelo_resistencia.joblib'); df_nuevo debe traer las
    columnas de paquete['features'] (una fila por mezcla y edad). Devuelve MPa."""
    return paquete["modelo"].predict(df_nuevo[paquete["features"]])


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--excel", default=EXCEL_DEFECTO)
    ap.add_argument("--salida", default=SALIDA_DEFECTO)
    ap.add_argument("--validacion", choices=["mezcla", "cemento"], default="mezcla",
                    help="esquema usado para cumplimiento y recomendación")
    ap.add_argument("--regla", choices=["fcr", "fc"], default="fcr",
                    help="criterio de cumplimiento (fcr = E.060; fc = f'c de diseño)")
    ap.add_argument("--prob-min", type=float, default=0.80,
                    help="probabilidad mínima de cumplir para ser elegible")
    ap.add_argument("--max-desgaste", type=float, default=None,
                    help="desgaste Los Ángeles máximo del concreto (%%), opcional")
    ap.add_argument("--w-co2", type=float, default=0.5)
    ap.add_argument("--w-costo", type=float, default=0.3)
    ap.add_argument("--w-res", type=float, default=0.2)
    ap.add_argument("--fc", type=float, help="f'c requerido (kg/cm²) para recomendación puntual")
    ap.add_argument("--cantera", choices=["La Poderosa", "Socabaya", "Cono Norte"])
    ap.add_argument("--exposicion", choices=list(EXPOSICIONES), default="ninguna")
    a = ap.parse_args()

    salida = Path(a.salida); salida.mkdir(parents=True, exist_ok=True)
    pesos = {"co2": a.w_co2, "costo": a.w_costo, "res": a.w_res}
    log = []

    def p(txt=""):
        print(txt); log.append(txt)

    # ---- datos
    try:
        tests, mix = construir_tablas(cargar_hojas(a.excel))
    except ValueError as e:
        raise SystemExit(str(e))
    p(f"Datos: {mix.shape[0]} mezclas · {tests.shape[0]} ensayos de compresión · "
      f"{tests['ID_Cem'].nunique()} cementos · {tests['cantera'].nunique()} canteras")
    nulos = tests[FEATURES].isna().sum()
    if nulos.any():
        p(f"  ! Variables con nulos (se imputan con la mediana): {nulos[nulos > 0].to_dict()}")
        tests[FEATURES] = tests[FEATURES].fillna(tests[FEATURES].median())

    # ---- 1) resistencia
    p("\n=== 1) PREDICCIÓN DE RESISTENCIA ===")
    res_reg, oof, mejor, modelos = entrenar_evaluar_regresion(tests)
    p(res_reg.round(3).to_string(index=False))
    p(f"\nMejor modelo (menor MAE con validación por mezcla): {mejor}")
    p("Nota: 'cemento' = predice un cemento que el modelo nunca vio; es la prueba más dura.")
    modelo_final = clone(modelos[mejor]).fit(tests[FEATURES], tests["resistencia_MPa"])
    joblib.dump({"modelo": modelo_final, "features": FEATURES, "nombre": mejor},
                salida / "modelo_resistencia.joblib")
    pred_res = tests[["ID_Mezcla", "cemento", "cantera", "fc_kg", "edad", "resistencia_MPa"]].copy()
    pred_res["pred_cv_mezcla_MPa"] = oof["mezcla"][mejor]
    pred_res["pred_cv_cemento_MPa"] = oof["cemento"][mejor]

    # ---- 2) cumplimiento
    p(f"\n=== 2) CUMPLIMIENTO ({'f' + chr(39) + 'cr E.060' if a.regla == 'fcr' else 'f' + chr(39) + 'c de diseño'}) ===")
    m28_idx = tests["edad"] == 28
    sigma28 = max(float(np.std(tests.loc[m28_idx, "resistencia_MPa"].values
                               - oof[a.validacion][mejor][m28_idx.values])), 0.5)
    p(f"Validación usada: {a.validacion} · σ del error a 28 d = {sigma28:.2f} MPa")
    res_cum, m28, clf_final, feats_c = evaluar_cumplimiento(
        tests, oof, mejor, a.regla, sigma28, a.validacion)
    p(f"Mezclas que cumplen realmente: {m28['cumple_real'].sum()} de {len(m28)} "
      f"({100 * m28['cumple_real'].mean():.0f} %)")
    p(res_cum.round(3).to_string(index=False))
    if clf_final is not None:
        joblib.dump({"modelo": clf_final, "features": feats_c, "regla": a.regla},
                    salida / "modelo_cumplimiento.joblib")

    # ---- 3) recomendación
    p("\n=== 3) RECOMENDACIÓN DE CEMENTO ADICIONADO vs TIPO I ===")
    p(f"Criterios: P(cumple) ≥ {a.prob_min:.2f} · pesos CO2/costo/resistencia = "
      f"{pesos['co2']}/{pesos['costo']}/{pesos['res']}"
      + (f" · desgaste ≤ {a.max_desgaste} %" if a.max_desgaste else ""))
    resumen, detalle = tabla_escenarios(m28, "prob_regresion", "pred28_MPa", a.prob_min,
                                        a.max_desgaste, pesos)
    ev, comp = evaluar_recomendacion(m28, a.prob_min, a.max_desgaste, pesos, sigma28)
    p("\nEvaluación de la recomendación (modelo fuera de muestra vs datos medidos):")
    for k, v in ev.items():
        p(f"  {k}: {v:.1f}" if isinstance(v, float) else f"  {k}: {v}")
    p("\nResumen por exposición (qué cemento se recomienda más veces):")
    p(resumen.groupby("exposicion")["recomendado"].value_counts().to_string())

    # recomendación puntual
    if a.fc is not None and a.cantera is not None:
        niveles = sorted(m28["fc_kg"].unique())
        nivel = next((n for n in niveles if n >= a.fc), niveles[-1])
        if nivel != a.fc:
            p(f"\n  ! No hay mezclas de f'c={a.fc:g}; se usa el nivel de diseño {nivel:g} kg/cm².")
        df = m28[(m28["fc_kg"] == nivel) & (m28["cantera"] == a.cantera)]
        rk, gan = recomendar_escenario(df, a.exposicion, a.prob_min, a.max_desgaste, pesos,
                                       "prob_regresion", "pred28_MPa")
        p(f"\n>>> RECOMENDACIÓN: f'c {nivel:g} · {a.cantera} · exposición {a.exposicion}  →  {gan}")
        cols = ["cemento", "tipo_cemento", "pred28_MPa", "umbral_MPa", "prob_regresion",
                "red_co2_pct", "dif_costo_pct", "ok_exposicion", "elegible", "score", "ranking"]
        p(rk[cols].round(2).to_string(index=False))

    # ---- guardar
    with pd.ExcelWriter(salida / "resultados_ml.xlsx") as xw:
        res_reg.to_excel(xw, sheet_name="metricas_resistencia", index=False)
        pred_res.to_excel(xw, sheet_name="predicciones_resistencia", index=False)
        res_cum.to_excel(xw, sheet_name="metricas_cumplimiento", index=False)
        m28.drop(columns=["index"]).to_excel(xw, sheet_name="predicciones_cumplimiento", index=False)
        resumen.to_excel(xw, sheet_name="recomendacion_escenarios", index=False)
        detalle.drop(columns=["ID_Mezcla"], errors="ignore").to_excel(xw, sheet_name="ranking_detallado", index=False)
        comp.to_excel(xw, sheet_name="evaluacion_recomendacion", index=False)
    graficos(salida, tests, oof, mejor, res_reg, modelo_final, m28)
    (salida / "resumen.txt").write_text("\n".join(log), encoding="utf-8")
    p(f"\nListo. Resultados en: {salida.resolve()}")


if __name__ == "__main__":
    main()
