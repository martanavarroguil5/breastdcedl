'use strict';

let graficas = [];
let archivosElegidos = [];
let peticion = 0;
let pendiente = null;
const estado = document.getElementById('estado-peticion');

function mensaje(texto, error = false) {
  estado.textContent = texto;
  estado.classList.toggle('error-peticion', error);
  estado.hidden = !texto;
}

function montarGraficas(raiz) {
  const observador = new IntersectionObserver(async entradas => {
    for (const entrada of entradas) {
      if (!entrada.isIntersecting) continue;
      const el = entrada.target;
      observador.unobserve(el);
      const datos = el.querySelector('.grafica-datos');
      if (!datos || el.dataset.montada) continue;
      el.dataset.montada = '1';
      try {
        const especificacion = JSON.parse(datos.textContent);
        const resultado = await vegaEmbed(el, especificacion, {actions: false, renderer: 'svg', tooltip: true});
        if (!el.isConnected) { resultado.finalize(); continue; }
        const resize = new ResizeObserver(() => {
          if (el.clientWidth > 0) resultado.view.width(el.clientWidth).resize().runAsync();
        });
        resize.observe(el);
        graficas.push({resultado, resize});
      } catch (error) {
        el.textContent = 'No se ha podido dibujar esta gráfica.';
        el.dataset.error = '1';
        console.error(error);
      }
    }
  }, {rootMargin: '250px'});
  raiz.querySelectorAll('.grafica').forEach(el => observador.observe(el));
  return observador;
}

let observador = montarGraficas(document);

function actualizarRango(input) {
  const [valor, etiqueta] = JSON.parse(input.dataset.opciones)[Number(input.value)];
  const grupo = input.closest('.rango');
  grupo.querySelector('input[type=hidden]').value = valor;
  grupo.querySelector('output').textContent = etiqueta;
  input.setAttribute('aria-valuetext', etiqueta);
}

async function actualizar(formulario, control) {
  const cambiaPaciente = control && (['paciente_de_ejemplo_train', 'origen'].includes(control.name) || control.type === 'file');
  if (!cambiaPaciente && !formulario.reportValidity()) return;
  const numero = ++peticion;
  if (pendiente) pendiente.abort();
  pendiente = new AbortController();
  const datos = new FormData(formulario);
  datos.delete('archivos');
  if (datos.get('origen') === 'Subir PNG') {
    archivosElegidos.forEach(archivo => datos.append('archivos', archivo, archivo.name));
  } else archivosElegidos = [];
  // No trasladar la clínica de una paciente al cambiar de ejemplo, origen o PNG.
  if (cambiaPaciente) {
    for (const clave of [...datos.keys()]) {
      if (['edad_conocida', 'edad_anos', 'volumen_conocido', 'volumen_tumoral_cm3'].includes(clave) || clave.startsWith('hr_') || clave.startsWith('her2_')) datos.delete(clave);
    }
  }
  const abiertos = [...formulario.querySelectorAll('details[open]')].map(el => el.id);
  const foco = control && control.id;
  const desplazamiento = window.scrollY;
  document.body.classList.add('cargando');
  formulario.setAttribute('aria-busy', 'true');
  mensaje(location.pathname === '/prediccion' ? 'Calculando con los modelos guardados…' : 'Actualizando…');
  try {
    const respuesta = await fetch(formulario.action, {method: 'POST', body: datos, headers: {'X-Panel-Fragmento': '1'}, signal: pendiente.signal});
    if (!respuesta.ok) {
      const error = await respuesta.json().catch(() => ({}));
      throw new Error(error.error || 'No se ha podido actualizar el panel.');
    }
    const contenido = await respuesta.text();
    if (numero !== peticion) return;
    observador.disconnect();
    graficas.forEach(({resultado, resize}) => {resize.disconnect(); resultado.finalize();});
    graficas = [];
    formulario.innerHTML = contenido;
    abiertos.forEach(id => {const el = document.getElementById(id); if (el) el.open = true;});
    observador = montarGraficas(formulario);
    if (foco) document.getElementById(foco)?.focus({preventScroll: true});
    window.scrollTo({top: desplazamiento, behavior: 'instant'});
    mensaje('');
    document.dispatchEvent(new CustomEvent('panel:actualizado'));
  } catch (error) {
    if (error.name !== 'AbortError' && numero === peticion) mensaje(error.message, true);
  } finally {
    if (numero === peticion) {
      document.body.classList.remove('cargando');
      formulario.removeAttribute('aria-busy');
    }
  }
}

document.addEventListener('input', evento => {
  if (evento.target.matches('input[type=range][data-opciones]')) actualizarRango(evento.target);
});
document.addEventListener('change', evento => {
  const control = evento.target;
  const formulario = control.closest('#panel-form');
  if (!formulario) return;
  if (control.matches('[data-interruptor]')) {
    control.closest('.control').querySelector('input[type=hidden]').value = control.checked ? '1' : '0';
    const numero = control.closest('.columna')?.querySelector('input[type=number]');
    if (numero) numero.disabled = !control.checked;
  }
  if (control.matches('input[type=file]')) archivosElegidos = [...control.files];
  if (control.matches('input[type=range]')) actualizarRango(control);
  actualizar(formulario, control);
});
document.addEventListener('submit', evento => {
  if (evento.target.id === 'panel-form') {evento.preventDefault(); actualizar(evento.target);}
});
document.addEventListener('dragover', evento => {if (evento.target.closest('.subida')) evento.preventDefault();});
document.addEventListener('drop', evento => {
  const zona = evento.target.closest('.subida');
  if (!zona) return;
  evento.preventDefault();
  archivosElegidos = [...evento.dataTransfer.files];
  actualizar(zona.closest('form'), zona.querySelector('input[type=file]'));
});
