"""Regenera figuras documentales a partir de métricas agregadas ya guardadas.

Uso desde la raíz: python scripts/generar_figuras_experimentos.py
No carga datos de pacientes, pesos ni módulos de entrenamiento.
"""
from pathlib import Path
import json

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


RAIZ = Path(__file__).resolve().parents[1] / "experiments"
TINTA = '#233849'
CONTROL = '#697d8a'
ACENTO = '#147d82'
ENSAYO = '#bd6849'
FONDO = '#ffffff'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'text.color': TINTA, 'axes.labelcolor': TINTA,
    'xtick.color': TINTA, 'ytick.color': TINTA,
    'axes.edgecolor': '#d0d9dd', 'figure.facecolor': FONDO,
    'axes.facecolor': FONDO, 'savefig.facecolor': FONDO,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.axisbelow': True,
})


def cabecera(fig, titulo, subtitulo, pie):
    fig.text(.04, .95, titulo, fontsize=18, weight='bold', va='top')
    fig.text(.04, .875, subtitulo, fontsize=11, va='top', color=CONTROL)
    fig.text(.04, .045, pie, fontsize=10, color=CONTROL, va='bottom')


def terminar(fig, carpeta, nombre):
    destino = carpeta / 'figuras'
    destino.mkdir(exist_ok=True)
    fig.savefig(destino / f'{nombre}.png', dpi=170, metadata={'Software': 'Matplotlib'})
    plt.close(fig)


def puntos(carpeta, datos):
    modelos = datos['modelos']
    fig, ax = plt.subplots(figsize=(13, max(4.8, len(modelos) * .47 + 2.4)))
    fig.subplots_adjust(left=.38, right=.79, top=.76, bottom=.21)
    cabecera(fig, datos['titulo'], 'Validación exterior · media de AUC y desviación estándar entre cinco folds',
             'Fuente: resultados.json y los informes de fuentes/. La DE no es un intervalo de confianza.')
    minimo = min(m['auc_media_folds'] - m['auc_de_folds'] for m in modelos)
    maximo = max(m['auc_media_folds'] + m['auc_de_folds'] for m in modelos)
    ax.set_xlim(min(.5, minimo - .015), maximo + .02)
    for i, m in enumerate(modelos):
        color = ACENTO if m.get('adoptado') else CONTROL
        ax.errorbar(m['auc_media_folds'], i, xerr=m['auc_de_folds'], fmt='o',
                    color=color, capsize=4, markersize=7, lw=2)
        ax.text(1.04, i, f"{m['auc_media_folds']:.4f} ± {m['auc_de_folds']:.4f}",
                transform=ax.get_yaxis_transform(), va='center', color=color)
    ax.set_yticks(range(len(modelos)), [m['nombre'] for m in modelos])
    ax.set_ylim(len(modelos) - .5, -.5)
    ax.set_xlabel('AUC media por fold')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.grid(axis='x', alpha=.25)
    ax.tick_params(axis='y', length=0, pad=12)
    ax.text(1.04, 1.08, 'Media ± DE', transform=ax.transAxes, weight='bold')
    terminar(fig, carpeta, 'resultados')


def diferencias(carpeta, datos):
    filas = datos['comparaciones']
    fig, ax = plt.subplots(figsize=(13, 4.9))
    fig.subplots_adjust(left=.35, right=.73, top=.73, bottom=.26)
    subtitulo = ('Δ AUC OOF agrupada cruda' if datos['metrica_principal'] == 'auc_agrupada'
                 else 'Δ AUC media por fold')
    cabecera(fig, datos['titulo'], subtitulo + ' · contraste pareado e IC95 % bootstrap',
             'Un intervalo que incluye cero no confirma mejora. Desarrollo; no evaluación de test.')
    inferior = min(f['limite_inferior'] for f in filas)
    superior = max(f['limite_superior'] for f in filas)
    margen = datos.get('margen_adopcion')
    if margen is not None:
        superior = max(superior, margen)
        ax.axvline(margen, ls='--', color=ACENTO, lw=1.4)
        ax.text(margen, 1.05, 'Margen +0,010', ha='center', color=ACENTO,
                transform=ax.get_xaxis_transform(), fontsize=10)
    amplitud = superior - inferior
    ax.set_xlim(inferior - amplitud * .13, superior + amplitud * .13)
    ax.axvline(0, color=CONTROL, lw=1)
    for i, f in enumerate(filas):
        ax.errorbar(f['delta'], i, xerr=[[f['delta'] - f['limite_inferior']],
                                       [f['limite_superior'] - f['delta']]],
                    fmt='o', markersize=7, lw=2, capsize=4, color=ENSAYO)
        texto = f"{f['delta']:+.4f}\n[{f['limite_inferior']:+.4f}; {f['limite_superior']:+.4f}]"
        ax.text(1.05, i, texto, transform=ax.get_yaxis_transform(), va='center', fontsize=11)
    ax.set_yticks(range(len(filas)), [f['nombre'] for f in filas])
    ax.set_ylim(len(filas) - .45, -.45)
    ax.set_xlabel('Diferencia de AUC · valores positivos favorecen la variante')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%+.3f'))
    ax.grid(axis='x', alpha=.2)
    ax.tick_params(axis='y', length=0, pad=10)
    terminar(fig, carpeta, 'diferencias')


def realce(carpeta, datos):
    modelos = datos['modelos']
    fig = plt.figure(figsize=(13, 5.2))
    auc = fig.add_axes([.26, .25, .29, .47])
    errores = fig.add_axes([.66, .25, .29, .47])
    cabecera(fig, 'Realce temporal sobre el multimodal',
             'Predicciones OOF crudas · 1.097 pacientes de desarrollo · selección histórica adaptativa',
             'Los recuentos usan umbral 0,5 sin calibrar. Positivo = pCR. Las agregaciones se detallan en la ficha.')
    for i, m in enumerate(modelos):
        color = CONTROL if i == 0 else ENSAYO
        auc.plot(m['auc_agrupada'], i, 'o', color=color, ms=8)
        auc.annotate(f"{m['auc_agrupada']:.4f}", (m['auc_agrupada'], i),
                     xytext=(0, 12), textcoords='offset points', ha='center', color=color)
        for campo, desplazamiento, c in [('fp', -.16, ENSAYO), ('fn', .16, CONTROL)]:
            errores.barh(i + desplazamiento, m[campo], .25, color=c,
                         label=campo.upper() if i == 0 else None)
            errores.text(m[campo] + 6, i + desplazamiento, str(m[campo]), va='center', fontsize=10)
    auc.set_yticks(range(3), [m['nombre'] for m in modelos])
    auc.set_ylim(2.5, -.5)
    auc.set_xlim(.705, .74)
    auc.set_xlabel('AUC OOF agrupada')
    auc.xaxis.set_major_locator(MaxNLocator(4))
    auc.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    auc.tick_params(axis='y', length=0, pad=12)
    errores.set_ylim(2.5, -.5)
    errores.set_yticks([])
    errores.set_xlim(0, 335)
    errores.set_xlabel('Pacientes · errores a umbral 0,5')
    errores.legend(loc='upper center', bbox_to_anchor=(.5, 1.22), ncol=2, frameon=False)
    for ax in (auc, errores):
        ax.grid(axis='x', alpha=.2)
    terminar(fig, carpeta, 'resultados')


def ablacion(carpeta, datos):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(left=.24, right=.92, top=.72, bottom=.27)
    cabecera(fig, 'Contribución visual en el ensayo SAM',
             'Ablación sin reentrenamiento · mismos pesos clínicos y sesgo · BCE ponderada',
             'En SAM, ambas AUC coinciden; cambio absoluto medio de probabilidad = 4,21 × 10⁻⁷.')
    for i, clave in enumerate(('control_weighted', 'sam_weighted')):
        m = datos['ablacion'][clave]
        ax.plot([m['full_raw_auc'], m['no_visual_raw_auc']], [i, i], color=CONTROL, lw=1.5)
        ax.scatter(m['full_raw_auc'], i, s=140, facecolors='none', edgecolors=ACENTO,
                   lw=2.2, label='Modelo completo' if i == 0 else None, zorder=3)
        ax.scatter(m['no_visual_raw_auc'], i, s=65, marker='x', c=ENSAYO,
                   lw=2, label='Sin término visual' if i == 0 else None, zorder=4)
        ax.text(m['full_raw_auc'], i - .18, f"{m['full_raw_auc']:.6f}", ha='center', fontsize=10, color=ACENTO)
        if m['full_raw_auc'] != m['no_visual_raw_auc']:
            ax.text(m['no_visual_raw_auc'], i + .23, f"{m['no_visual_raw_auc']:.6f}", ha='center', fontsize=10, color=ENSAYO)
    ax.set_yticks([0, 1], ['Control ponderado', 'SAM ponderado'])
    ax.set_ylim(1.6, -.6)
    ax.set_xlim(.698, .712)
    ax.set_xlabel('AUC OOF agrupada cruda')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.tick_params(axis='y', length=0, pad=10)
    ax.grid(axis='x', alpha=.2)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.27), ncol=2, frameon=False)
    terminar(fig, carpeta, 'ablacion')


def folds(carpeta, datos):
    fig, ax = plt.subplots(figsize=(12, 5.6))
    fig.subplots_adjust(left=.10, right=.95, top=.76, bottom=.23)
    cabecera(fig, datos['titulo'], 'Evaluación exterior por paciente · selección interna de cada variante',
             'Media AUC: referencia 0,5810; ajustada 0,5766. No confundir con la AUC agrupada cruda.')
    for m, color, marcador in zip(datos['modelos'], (CONTROL, ENSAYO), ('o', 'D')):
        ax.plot(range(1, 6), m['auc_por_fold'], marker=marcador, lw=2, color=color,
                ms=7, label=m['nombre'])
    ax.axhline(.60, color=ACENTO, lw=1, ls='--', label='Mínimo de referencia: 0,60')
    ax.set_xticks(range(1, 6), [f'Fold {i}' for i in range(1, 6)])
    ax.set_ylabel('AUC del fold')
    ax.set_ylim(.50, .65)
    ax.grid(axis='y', alpha=.2)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=3, frameon=False, fontsize=10)
    terminar(fig, carpeta, 'folds')


def main():
    carpetas = sorted(RAIZ.glob('[0-9][0-9]_*/resultados.json'))
    for archivo in carpetas:
        datos = json.loads(archivo.read_text(encoding='utf-8'))
        carpeta = archivo.parent
        if carpeta.name.startswith('01_'):
            realce(carpeta, datos)
        elif carpeta.name.startswith('02_'):
            ablacion(carpeta, datos)
        elif carpeta.name.startswith('03_'):
            folds(carpeta, datos)
        else:
            puntos(carpeta, datos)
        if datos['comparaciones'] and not carpeta.name.startswith('05_'):
            diferencias(carpeta, datos)
        print(f'Figuras actualizadas: {carpeta.name}')


if __name__ == '__main__':
    main()
