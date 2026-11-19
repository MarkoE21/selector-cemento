# Selector de cemento adicionado (Streamlit)

Hackathon Innova Sur 2026 · Reto 2 · Yura S.A.

App web que, a partir de la base de mezclas de concreto:

1. **Predice la resistencia** a 3, 7 y 28 días (modelo de ML).
2. **Verifica el cumplimiento** de f'cr (E.060) con una probabilidad.
3. **Recomienda el cemento adicionado** más conveniente frente al Tipo I, según CO₂, costo y exposición.
4. Permite **predecir una dosificación nueva** y **reentrenar subiendo el Excel actualizado**.

## Estructura

```
app_concreto/
├── app.py              # interfaz (Streamlit)
├── motor.py            # lógica: entrenar, recomendar, predecir
├── ml_concreto.py      # funciones de ML (también sirve como script de consola)
├── datos/base_mezclas.xlsx   # base incluida (tesis UCSM 2020)
├── requirements.txt
└── .streamlit/config.toml
```

## Probar en tu computadora

Requiere Python 3.10 o superior.

```bash
cd app_concreto
python -m venv .venv
# Windows:  .venv\Scripts\activate      Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Se abre en http://localhost:8501. La primera carga entrena los modelos (de 10 s a 1 min).

## Publicarla en internet (gratis)

1. Crea una cuenta en https://github.com y un repositorio nuevo.
2. Sube **todo el contenido** de la carpeta `app_concreto` (incluida `datos/`).
3. Entra a https://share.streamlit.io, inicia sesión con GitHub y pulsa **Create app**.
4. Elige el repositorio, la rama `main` y como archivo principal `app.py`.
5. En *Advanced settings* elige Python 3.11 o 3.12 y pulsa **Deploy**.

Te entrega un link público para compartir con el jurado.

> **Sobre los datos:** la base proviene de una tesis (uso académico con cita de la fuente) e incluye
> precios referenciales. Si el repositorio es público, el Excel también lo será. Para evitarlo, usa un
> repositorio **privado** (Streamlit Cloud lo permite) o borra `datos/base_mezclas.xlsx` y haz que se suba
> siempre desde la barra lateral.

## Actualizar el modelo con ensayos nuevos

1. Agrega las filas nuevas al Excel (hojas *Mezclas*, *Ensayos_Resistencia*, *Ensayos_Durabilidad*) con las mismas columnas.
2. En la barra lateral de la app elige **Subir Excel actualizado** y sube el archivo.
3. La app reentrena automáticamente. Para que sea permanente, reemplaza `datos/base_mezclas.xlsx` y vuelve a subirlo a GitHub (la app se actualiza sola).

## Cómo leer los resultados

- **Prob. de cumplir:** probabilidad de que la resistencia a 28 d supere el f'cr, estimada con el error del modelo fuera de muestra.
- **Elegible:** apto para la exposición y con probabilidad ≥ el mínimo elegido.
- **Puntaje:** combinación ponderada (ajustable) de reducción de CO₂, ahorro de costo y margen de resistencia.
- **Cemento nuevo vs mezcla nueva:** la primera validación es la más exigente (el cemento nunca fue visto).

## Limitaciones y supuestos

- 48 mezclas de una sola fuente: válido para estos 8 cementos y 3 canteras.
- Agua constante (205 kg/m³) y solo 2 niveles de a/mc: el modelo no aprendió su efecto; la app avisa cuando se sale del rango.
- Composición de varios cementos y precios 2026 son estimaciones.
- Reglas de exposición (salitre → HS; sulfatos → IP o HS) son un supuesto: validarlas con el equipo de Civil.
- Herramienta de apoyo; no reemplaza ensayos de laboratorio ni el criterio del ingeniero.

## Solución de problemas

| Problema | Causa probable |
|---|---|
| "Faltan hojas en el Excel" | El archivo subido no conserva los nombres de hojas de la base original. |
| Tarda mucho | Usa *Precisión: Rápido* en Opciones avanzadas. |
| Error al desplegar | Revisa que `requirements.txt` esté en la raíz del repositorio. |
