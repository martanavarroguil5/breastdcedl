'use strict';
// Sustituye al servidor en la web estática: recompone cada página con bloques ya calculados.
(() => {
  const pagina = window.PAGINA_ESTATICA;
  const caché = new Map();
  const leer = url => {
    if (!caché.has(url)) caché.set(url, fetch(url).then(r => {
      if (!r.ok) throw new Error('No se han podido cargar los datos de la página.');
      return r.json();
    }).catch(error => {caché.delete(url); throw error;}));
    return caché.get(url);
  };
  const variante = nombre => leer(`datos/v/${nombre}.json`);
  const resolver = (campo, datos) => campo.replace(/\{(.+?)\}/g, (_, n) => datos.get(n) ?? '');

  let manifiesto = null, ramaActual = null, bloques = null;

  async function aplicarGrupo(grupo, datos) {
    const partes = grupo.campos.map(c => datos.get(resolver(c, datos)) ?? '');
    const clave = partes.join('|');
    const alternativa = [...partes.slice(0, -1), ''].join('|');
    const nombre = grupo.claves[clave] ?? grupo.claves[alternativa];
    if (!nombre) return;
    for (const [i, html] of Object.entries(await variante(nombre))) bloques[Number(i)] = html;
  }

  async function actualizarVisores(rama, datos) {
    const {campos, indices, red, bloques: alto} = rama.dinamico;
    const paciente = datos.get(campos[0]);
    if (!paciente) return;
    const corte = datos.get(campos[1]);
    const visores = await window.recorridoEstatico(paciente, corte === null ? null : Number(corte));
    const marco = (html, codigo) => {
      const caja = document.createElement('template');
      caja.innerHTML = html;
      caja.content.querySelector('iframe').srcdoc = codigo;
      return caja.innerHTML;
    };
    bloques[indices[0]] = marco(bloques[indices[0]], visores.red(red));
    bloques[indices[1]] = marco(bloques[indices[1]], visores.bloques(alto));
  }

  async function atender(datos, control) {
    if (pagina === 'prediccion') return window.predecirEstatico(datos, control);
    manifiesto ??= await leer(`datos/${pagina}.json`);
    const valor = manifiesto.campo ? datos.get(manifiesto.campo) : '';
    const rama = manifiesto.ramas[valor] ?? Object.values(manifiesto.ramas)[0];
    const nombre = control?.closest('.rango')?.querySelector('input[type=hidden]')?.name ?? control?.name;
    if (rama !== ramaActual) {
      ramaActual = rama;
      bloques = [...await variante(rama.bloques)];
      await Promise.all(rama.grupos.map(g => aplicarGrupo(g, datos)));
    } else {
      const grupos = rama.grupos.filter(g => g.campos.some(c => resolver(c, datos) === nombre));
      await Promise.all(grupos.map(g => aplicarGrupo(g, datos)));
    }
    if (rama.dinamico?.campos.includes(nombre)) await actualizarVisores(rama, datos);
    return bloques.join('').replace(/href="(prediccion|red_imagen|combinado|metodologia)(#[^"]*)?"/g, 'href="$1.html$2"');
  }

  // Las peticiones se atienden de una en una; en Predecir solo importa el último estado del formulario.
  let cola = Promise.resolve(), turno = 0;
  window.panelEstatico = (datos, control) => {
    const mio = ++turno;
    const tarea = cola.catch(() => {}).then(() => (pagina === 'prediccion' && mio !== turno ? '' : atender(datos, control))).catch(error => {
      if (error instanceof TypeError || /wasm|abort|fetch/i.test(error.message)) {
        throw new Error('Se ha perdido la conexión al descargar el modelo. Revisa la red y recarga la página.');
      }
      throw error;
    });
    cola = tarea;
    return tarea;
  };
})();
