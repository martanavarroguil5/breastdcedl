'use strict';
// Predicción en el navegador: misma lectura de PNG, normalización, red, calibración y fusión que Python.
(() => {
  const FASES = ['PRE', 'EARLY', 'LATE'];
  const LADO = 256;
  const EPS = 1e-6;
  const MAXIMO_ARCHIVOS = 96;
  const MAXIMO_BYTES = 48 * 1024 * 1024;
  const NOMBRE_FASE = /^(.+?)[_\-\s]?(PRE|EARLY|LATE)\.png$/i;
  const VERSION_ENLACE = 'static/vendor/ort/';

  let info = null, sesion = null, sesionRecorrido = null, cargaOrt = null;

  // Un corte breve de la red no debe echar a perder una predicción.
  async function descargar(url, mensaje, intentos = 3) {
    for (let i = 1; ; i++) {
      try {
        const respuesta = await fetch(url);
        if (respuesta.ok) return respuesta;
        if (respuesta.status < 500 || i === intentos) throw new Error(mensaje);
      } catch (error) {
        if (i === intentos) throw new Error(mensaje);
      }
      await new Promise(r => setTimeout(r, 400 * i));
    }
  }
  const datosInfo = () => info ??= descargar('datos/prediccion.json', 'No se han podido cargar los datos del modelo.').then(r => r.json())
    .catch(error => {info = null; throw error;});

  // ---------------------------------------------------------------- formato
  const fijo = (x, d) => x.toFixed(d).replace('.', ',');
  const pct = (x, d = 0) => `${fijo(100 * x, d)} %`;
  const escapar = t => String(t).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const media = a => a.reduce((s, v) => s + v, 0) / a.length;
  const logit = p => {const c = Math.min(Math.max(p, EPS), 1 - EPS); return Math.log(c / (1 - c));};
  const sigmoide = z => 1 / (1 + Math.exp(-z));

  function percentil(valores, q) {
    const orden = Float64Array.from(valores).sort();
    const pos = (orden.length - 1) * q / 100;
    const i = Math.floor(pos), j = Math.min(i + 1, orden.length - 1);
    return orden[i] + (orden[j] - orden[i]) * (pos - i);
  }

  // ---------------------------------------------------------------- PNG exacto, sin pasar por canvas
  const TABLA_CRC = Int32Array.from({length: 256}, (_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c;
  });
  function crc(bytes) {
    let c = -1;
    for (const b of bytes) c = TABLA_CRC[(c ^ b) & 255] ^ (c >>> 8);
    return (c ^ -1) >>> 0;
  }

  async function leerPNG(buffer, nombre) {
    const bytes = new Uint8Array(buffer), vista = new DataView(buffer);
    const malo = texto => new Error(`${nombre}: ${texto}`);
    const firma = [137, 80, 78, 71, 13, 10, 26, 10];
    if (bytes.length < 33 || firma.some((b, i) => bytes[i] !== b)) throw malo('no es un PNG válido');
    let pos = 8, cabecera = null, final = false;
    const idat = [];
    while (pos + 12 <= bytes.length && !final) {
      const largo = vista.getUint32(pos);
      if (pos + 12 + largo > bytes.length) throw malo('archivo PNG truncado');
      const tipo = String.fromCharCode(...bytes.subarray(pos + 4, pos + 8));
      if (crc(bytes.subarray(pos + 4, pos + 8 + largo)) !== vista.getUint32(pos + 8 + largo)) throw malo('archivo PNG corrupto');
      if (tipo === 'IHDR') cabecera = {ancho: vista.getUint32(pos + 8), alto: vista.getUint32(pos + 12), bits: bytes[pos + 16], color: bytes[pos + 17], entrelazado: bytes[pos + 20]};
      else if (tipo === 'IDAT') idat.push(bytes.subarray(pos + 8, pos + 8 + largo));
      else if (tipo === 'IEND') final = true;
      pos += 12 + largo;
    }
    if (!cabecera || !idat.length || !final) throw malo('archivo PNG incompleto');
    const {ancho, alto, bits, color, entrelazado} = cabecera;
    if (ancho !== LADO || alto !== LADO || bits !== 8 || color !== 0 || entrelazado !== 0) {
      throw malo(`se espera PNG gris L ${LADO}×${LADO}, recibido ${ancho}×${alto}, ${bits} bits, tipo de color ${color}`);
    }
    let crudo;
    try {
      const flujo = new Blob(idat).stream().pipeThrough(new DecompressionStream('deflate'));
      crudo = new Uint8Array(await new Response(flujo).arrayBuffer());
    } catch {throw malo('archivo PNG corrupto');}
    if (crudo.length !== (LADO + 1) * LADO) throw malo('archivo PNG incompleto');
    const salida = new Uint8Array(LADO * LADO);
    for (let y = 0; y < LADO; y++) {
      const filtro = crudo[y * (LADO + 1)], o = y * (LADO + 1) + 1, d = y * LADO;
      if (filtro > 4) throw malo('archivo PNG corrupto');
      for (let x = 0; x < LADO; x++) {
        const a = x ? salida[d + x - 1] : 0, b = y ? salida[d + x - LADO] : 0, c = x && y ? salida[d + x - LADO - 1] : 0;
        let previo = 0;
        if (filtro === 1) previo = a;
        else if (filtro === 2) previo = b;
        else if (filtro === 3) previo = (a + b) >> 1;
        else if (filtro === 4) {
          const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
          previo = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
        }
        salida[d + x] = (crudo[o + x] + previo) & 255;
      }
    }
    return salida;
  }

  // ---------------------------------------------------------------- imágenes para los visores
  function aURL(rgba, ancho, alto) {
    const lienzo = document.createElement('canvas');
    lienzo.width = ancho; lienzo.height = alto;
    lienzo.getContext('2d').putImageData(new ImageData(rgba, ancho, alto), 0, 0);
    return Math.min(ancho, alto) >= 48 ? lienzo.toDataURL('image/webp', 0.86) : lienzo.toDataURL('image/png');
  }
  function urlGris(valores, ancho, alto) {
    const rgba = new Uint8ClampedArray(ancho * alto * 4);
    for (let i = 0; i < ancho * alto; i++) {
      rgba[4 * i] = rgba[4 * i + 1] = rgba[4 * i + 2] = valores[i];
      rgba[4 * i + 3] = 255;
    }
    return aURL(rgba, ancho, alto);
  }
  function urlRGB(pintar, ancho, alto) {
    const rgba = new Uint8ClampedArray(ancho * alto * 4);
    for (let i = 0; i < ancho * alto; i++) {
      const [r, g, b] = pintar(i);
      rgba[4 * i] = r; rgba[4 * i + 1] = g; rgba[4 * i + 2] = b; rgba[4 * i + 3] = 255;
    }
    return aURL(rgba, ancho, alto);
  }

  function escalaRealce(cortes) {
    const positivas = [];
    for (const c of cortes) for (let i = 0; i < LADO * LADO; i++) {
      const d = c[1][i] - c[0][i];
      if (d > 0) positivas.push(d);
    }
    return Math.max(positivas.length ? percentil(positivas, 99.5) : 1, 1);
  }
  function urlRealce(corte, escala) {
    const rampa = [[110, 45, 28], [201, 119, 83], [232, 195, 154]];
    const rgba = new Uint8ClampedArray(LADO * LADO * 4);
    const divisor = Math.max(escala, 1);
    for (let i = 0; i < LADO * LADO; i++) {
      const t = Math.min(Math.max((corte[1][i] - corte[0][i]) / divisor, 0), 1);
      const u = Math.min(t * 2, 2), [a, b, f] = u < 1 ? [rampa[0], rampa[1], u] : [rampa[1], rampa[2], u - 1];
      for (let k = 0; k < 3; k++) rgba[4 * i + k] = Math.trunc(a[k] + (b[k] - a[k]) * f);
      rgba[4 * i + 3] = Math.trunc(Math.min(t * 1.6, 1) ** 0.8 * 255);
    }
    return aURL(rgba, LADO, LADO);
  }

  // ---------------------------------------------------------------- modelos
  function cargarOrt() {
    cargaOrt ??= new Promise((resolver, rechazar) => {
      const guion = document.createElement('script');
      guion.src = `${VERSION_ENLACE}ort.wasm.min.js`;
      guion.onload = () => {
        ort.env.wasm.wasmPaths = new URL(VERSION_ENLACE, document.baseURI).href;
        ort.env.wasm.proxy = false;
        resolver(ort);
      };
      guion.onerror = () => {cargaOrt = null; rechazar(new Error('No se ha podido cargar el motor de inferencia.'));};
      document.head.append(guion);
    });
    return cargaOrt;
  }
  const crear = async (nombre, huella) => {
    await cargarOrt();
    const contenido = await (await descargar(`modelo/${nombre}`, 'No se ha podido descargar el modelo.')).arrayBuffer();
    const resumen = await crypto.subtle?.digest('SHA-256', contenido);
    if (resumen) {
      const texto = [...new Uint8Array(resumen)].map(b => b.toString(16).padStart(2, '0')).join('');
      if (texto !== huella) throw new Error('La huella del modelo descargado no coincide.');
    }
    return ort.InferenceSession.create(contenido, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
  };
  const modelo = async () => sesion ??= crear('reddce.onnx', (await datosInfo()).modelo.sha256).catch(error => {sesion = null; throw error;});
  const modeloRecorrido = async () => sesionRecorrido ??= crear('recorrido.onnx', (await datosInfo()).modelo.sha256_recorrido).catch(error => {sesionRecorrido = null; throw error;});

  const tensorCorte = (corte, n = 1) => {
    const x = new Float32Array(3 * LADO * LADO);
    for (let k = 0; k < 3; k++) x.set(corte[k], k * LADO * LADO);
    return new ort.Tensor('float32', x, [n, 3, LADO, LADO]);
  };

  async function predecir(cortes, avisar) {
    const red = await modelo();
    const inicio = performance.now();
    const pCortes = [];
    for (let i = 0; i < cortes.length; i++) {
      avisar?.(`Calculando corte ${i + 1} de ${cortes.length}…`);
      await new Promise(r => setTimeout(r));
      const {p} = await red.run({pixeles: tensorCorte(cortes[i])});
      pCortes.push(Array.from(p.data));
    }
    const redes = pCortes[0].length;
    const porCorte = pCortes.map(media);
    const porRed = Array.from({length: redes}, (_, r) => media(pCortes.map(f => f[r])));
    return {pCortes, porCorte, porRed, pCruda: media(porCorte), segundos: (performance.now() - inicio) / 1000};
  }

  // ---------------------------------------------------------------- calibración y fusión
  const calibrar = (d, p) => sigmoide(d.platt.a * logit(p) + d.platt.b);

  function aplicarLogistica(params, fila) {
    const Z = params.columnas.map((c, i) => (fila[c] - params.media[i]) / params.escala[i]);
    let z = params.intercepto + params.columnas.reduce((s, c, i) => s + Z[i] * params.coeficientes[c], 0);
    if (params.interaccion) {
      const i = params.columnas.indexOf('imagen');
      z += params.coeficientes['imagen×HER2'] * Z[i] * fila.HER2 + params.coeficientes['imagen×HR−'] * Z[i] * (1 - fila.HR);
    }
    return sigmoide(z);
  }

  function decidir(d, pCruda, clinica) {
    const fila = {edad: clinica.edad, HR: clinica.HR, HER2: clinica.HER2,
                  log_volumen: clinica.volumen === null ? null : Math.log1p(clinica.volumen), imagen: logit(pCruda)};
    const grupos = [['subtipo', ['HR', 'HER2']], ['edad', ['edad']], ['volumen', ['log_volumen']]]
      .filter(([, vs]) => vs.every(v => fila[v] !== null)).map(([g]) => g);
    if (!grupos.length) return {p: calibrar(d, pCruda), umbral: d.umbral, submodelo: 'imagen sola (paquete de imagen)'};
    const nombre = [...grupos, 'imagen'].join('+');
    const params = d.submodelos[nombre];
    return {p: aplicarLogistica(params, fila), umbral: params.umbral, submodelo: nombre};
  }

  // ---------------------------------------------------------------- lectura
  const buscarDerecha = (lista, v) => {let i = 0; while (i < lista.length && lista[i] <= v) i++; return i;};
  const buscarIzquierda = (lista, v) => {let i = 0; while (i < lista.length && lista[i] < v) i++; return i;};

  function lectura(d, titulo, p, perfiles, umbral, pie, extra = '') {
    const nombres = Object.keys(d.colores.perfiles);
    const filas = perfiles.perfiles;
    const hi = Math.max(0.6, Math.min(1, filas.at(-1).p_max + 0.05));
    const x = v => 100 * Math.min(Math.max(v, 0), hi) / hi;
    const limites = [0, ...perfiles.cortes, hi];
    const perfil = nombres[buscarDerecha(perfiles.cortes, p)];
    const fila = filas.find(f => f.perfil === perfil);
    const chipColor = ['alta', 'muy_alta'].includes(perfil) ? '#0b0e12' : d.colores.tinta;
    const chip = `<span class="chip" style="background:${d.colores.perfiles[perfil]};color:${chipColor}">P${nombres.indexOf(perfil) + 1} · ${d.colores.nombres[perfil]}</span>`;
    const segmentos = filas.map((f, i) => {
      const a = limites[i], b = limites[i + 1], ic = f.ic95;
      const tip = `<b>${d.colores.nombres[f.perfil]}</b> · probabilidad ${fijo(a, 2)}–${fijo(b, 2)}<br>En la validación cruzada: ${f.pcr} pCR de ${f.n} pacientes (${pct(f.tasa_pcr)}; IC95 % ${pct(ic[0])}–${pct(ic[1])}), ${fijo(f.razon, 1)} × la media`;
      return `<div class="seg" style="left:${x(a).toFixed(2)}%;width:${(x(b) - x(a)).toFixed(2)}%;background:${d.colores.perfiles[f.perfil]}"><div class="tip">${tip}</div></div>`;
    }).join('');
    const escalera = filas.map(f => `<div class="${f.perfil === perfil ? 'on' : ''}" title="${f.n} pacientes · ${f.pcr} con pCR"><i style="background:${d.colores.perfiles[f.perfil]}"></i><span>${d.colores.nombres[f.perfil]}</span><b>${pct(f.tasa_pcr)}</b></div>`).join('');
    let marcas = '';
    for (let k = 0; k * 0.05 < hi + 1e-9; k++) {
      const v = k * 0.05, mayor = Math.round(v * 100) % 10 === 0;
      marcas += `<div class="tick${mayor ? ' m' : ''}" style="left:${x(v).toFixed(2)}%">${mayor ? `<span>${fijo(v, 1)}</span>` : ''}</div>`;
    }
    const percentilP = Math.min(Math.max(buscarIzquierda(perfiles.percentiles, p) - 1, 0), 100);
    const clase = p >= umbral ? 'pCR' : 'no pCR';
    return `<div class="lectura"><div class="cab"><span class="t">${titulo}</span><span>umbral ${fijo(umbral, 3)} → ${clase}</span></div>`
      + `<div class="cifra">${fijo(100 * p, 1)}<small>%</small></div>`
      + `<div class="que">${chip} En la validación cruzada, <b>${pct(fila.tasa_pcr)}</b> de las pacientes de este perfil alcanzó pCR (IC95 % ${pct(fila.ic95[0])}–${pct(fila.ic95[1])}): <b>${fijo(fila.razon, 1)} ×</b> la media del ${pct(perfiles.prevalencia)}. Su probabilidad supera a la de <b>${percentilP} %</b> de las pacientes de desarrollo.${extra}</div>`
      + `<div class="regla-p">${segmentos}${marcas}<div class="umbral ${p >= umbral ? 'izq' : 'der'}" style="left:${x(umbral).toFixed(2)}%"><span>UMBRAL</span></div><div class="aguja" style="left:${x(p).toFixed(2)}%"></div></div>`
      + `<div class="escalera" aria-label="pCR observada en cada perfil">${escalera}</div><div class="pie-l">${pie}</div></div>`;
  }
  const lecturaVacia = (titulo, texto) => `<div class="lectura vacia"><div class="cab"><span class="t">${titulo}</span></div><div class="cifra gris">—</div><div class="que">${texto}</div></div>`;

  function graficaDistribucion(d, clave, p, titulo, texto) {
    const spec = structuredClone(d.distribucion[clave]);
    spec.title = titulo;
    const [bajo, alto] = spec.layer[0].encoding.x.scale.domain;
    for (const filas of Object.values(spec.datasets)) for (const f of filas) if (f.t === d.sentinela) {
      f.x = Math.min(Math.max(p, bajo), alto);
      f.p = p; f.t = texto;
    }
    const minimo = typeof spec.height === 'number' ? spec.height : 300;
    const json = JSON.stringify(spec).replaceAll('<', '\\u003c');
    return `<div class="grafica" role="figure" style="min-height:${minimo}px"><script type="application/json" class="grafica-datos">${json}<\/script></div>`;
  }

  // ---------------------------------------------------------------- recorrido por la red
  const descodificar = ({forma, b64}) => {
    const crudo = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    return {forma, datos: new Float32Array(crudo.buffer)};
  };
  const ordenDesc = (valores, n = valores.length) => [...valores.keys()].sort((a, b) => valores[b] - valores[a]).slice(0, n);

  function aristas(pesos, porDestino = 4, signo = false) {
    const maximo = Math.max(...pesos.flat().map(Math.abs)) || 1;
    const salida = [];
    pesos.forEach((fila, j) => {
      for (const i of ordenDesc(fila.map(Math.abs), porDestino)) {
        const w = Math.abs(fila[i]) / maximo;
        if (w > 0.02) salida.push([i, j, Math.round(w * 1000) / 1000, signo ? Math.sign(fila[i]) : 0]);
      }
    });
    return salida;
  }

  async function recorrido(corte) {
    const d = await datosInfo(), red = await modeloRecorrido();
    const s = await red.run({pixeles: tensorCorte(corte)});
    const pesos = (d.pesosDecodificados ??= {
      conectividad: d.pesos.conectividad.map(descodificar), lineal: descodificar(d.pesos.lineal), cabeza: descodificar(d.pesos.cabeza)});
    const mapas = ['canales', 'b1', 'b2', 'b3', 'b4'].map(n => s[n]);
    const vistas = s.vistas, z = s.z.data, oculta = s.oculta.data;
    const logitRed = s.logit.data[0];
    const [nVistas, nCanales4] = mapas[4].dims;
    const [nDensa, nRasgos] = pesos.lineal.forma;
    const aporte = Array.from(oculta, (v, i) => pesos.cabeza.datos[i] * v);
    const densa = ordenDesc(aporte.map(Math.abs), 16);
    const haciaDensa = densa.map(n => Array.from({length: nRasgos}, (_, f) => pesos.lineal.datos[n * nRasgos + f] * z[f]));
    const importancia = Array.from({length: nRasgos}, (_, f) => haciaDensa.reduce((t, fila) => t + Math.abs(fila[f]), 0));

    const canales4 = [], pool = [];
    for (const f of ordenDesc(importancia)) {
      const c = f % nCanales4;
      if (!canales4.includes(c) && canales4.length < 8) canales4.push(c);
      if (canales4.includes(c)) pool.push(f);
      if (pool.length === 12) break;
    }
    const actividad = mapas.map(m => {
      const [n, c, a, b] = m.dims, suma = new Float64Array(c);
      m.data.forEach((v, i) => {suma[Math.floor(i / (a * b)) % c] += Math.abs(v);});
      return Array.from(suma, v => v / (n * a * b));
    });
    const elegidos = [canales4], pares = [];
    for (let b = 3; b >= 0; b--) {
      const destino = elegidos[0], {forma, datos} = pesos.conectividad[b], entradas = forma[1];
      const m = destino.map(dst => Array.from({length: entradas}, (_, e) => datos[dst * entradas + e] * actividad[b][e]));
      const sumas = Array.from({length: entradas}, (_, e) => m.reduce((t, f) => t + f[e], 0));
      const origen = b === 0 ? [...Array(entradas).keys()] : ordenDesc(sumas, 8);
      pares.unshift(m.map(f => origen.map(e => f[e])));
      elegidos.unshift(origen);
    }

    const canalDe = (m, v, c) => {
      const [, canales, a, b] = m.dims, inicio = (v * canales + c) * a * b;
      return m.data.subarray(inicio, inicio + a * b);
    };
    const imagenes = (m, c, diverge) => {
      const [n, , a, b] = m.dims;
      const todas = Array.from({length: n}, (_, v) => canalDe(m, v, c));
      const planos = todas.flatMap(t => Array.from(t));
      if (diverge) {
        const escala = percentil(planos.map(Math.abs), 99.5) || 1;
        const fondo = [11, 14, 18], realce = [201, 119, 83], frio = [90, 158, 196];
        return todas.map(t => urlRGB(i => {
          const f = Math.min(Math.max(t[i] / Math.max(escala, 1e-6), -1), 1), color = f > 0 ? realce : frio;
          return fondo.map((b0, k) => Math.trunc(b0 + (color[k] - b0) * Math.abs(f)));
        }, b, a));
      }
      const maximo = Math.max(...planos) || 1;
      return todas.map(t => urlRGB(i => d.magma[Math.min(Math.floor(Math.min(Math.max(t[i] / maximo, 0), 1) * 256), 255)], b, a));
    };
    const gris = t => urlGris(Array.from(t, v => Math.trunc(Math.min(Math.max(v, 0), 1) * 255)), d.config.lado, d.config.lado);
    const vistaCanal = (v, k) => {
      const [, canales, a, b] = vistas.dims, inicio = (v * canales + k) * a * b;
      return vistas.data.subarray(inicio, inicio + a * b);
    };
    const nombresCanal = ['PRE', 'EARLY', 'LATE', 'EARLY − PRE', 'LATE − PRE', 'EARLY − LATE'];
    const textoCanal = ['Antes del contraste', 'Captación temprana', 'Fase tardía', 'Realce temprano: dónde entra el contraste', 'Realce tardío', 'Lavado: positivo si el contraste ya sale'];
    const {lado, escalas} = d.config;
    const mitad = k => {
      const salida = new Uint8Array(128 * 128);
      for (let y = 0; y < 128; y++) for (let x = 0; x < 128; x++) salida[y * 128 + x] = corte[k][y * 2 * LADO + x * 2];
      return salida;
    };
    const capas = [
      {id: 'fases', titulo: 'Fases', forma: '3 × 256²', nodos: FASES.map((f, k) => ({img: [urlGris(mitad(k), 128, 128)], tit: f, txt: textoCanal[k]}))},
      {id: 'vistas', titulo: 'Vistas', forma: `3 × ${lado}²`, nodos: escalas.map((e, v) => ({img: [gris(vistaCanal(v, 1))], vista: v, tit: `Vista central de ${e} px`, txt: `Recorte de ${e} × ${e} reescalado a ${lado}. Las tres vistas pasan por la misma red.`}))},
      {id: 'canales', titulo: 'Canales', forma: `6 × ${lado}²`, nodos: nombresCanal.map((n, k) => ({img: k >= 3 ? imagenes(mapas[0], k, true) : Array.from({length: nVistas}, (_, v) => gris(vistaCanal(v, k))), tit: n, txt: textoCanal[k]}))},
    ];
    elegidos.slice(1).forEach((canales, indice) => {
      const b = indice + 1, m = mapas[b], ladoM = m.dims[3], total = m.dims[1];
      capas.push({id: `b${b}`, titulo: `Bloque ${b}`, forma: `${total} × ${ladoM}²`, mostrados: `${canales.length} de ${total}`, pixelado: ladoM <= 32,
        nodos: canales.map(c => ({img: imagenes(m, c, false), tit: `Bloque ${b} · filtro ${c + 1} de ${total}`,
          txt: `Mapa de ${ladoM} × ${ladoM} tras 2 conv 3×3 + ReLU y MaxPool. Activación media ${fijo(actividad[b][c], 2)}`}))});
    });
    const impPool = pool.map(f => importancia[f]), maxPool = Math.max(...impPool);
    const redondo = v => Math.round(v * 1000) / 1000;
    capas.push({id: 'pool', titulo: 'Pooling', forma: `${z.length} rasgos`, mostrados: `${pool.length} de ${z.length}`,
      nodos: pool.map((f, k) => ({v: redondo(impPool[k] / maxPool),
        tit: `Vista ${escalas[Math.floor(f / (2 * nCanales4))]} px · ${(f % (2 * nCanales4)) < nCanales4 ? 'media' : 'máximo'} · filtro ${f % nCanales4 + 1}`,
        txt: `Valor ${fijo(z[f], 2)}: resume todo el mapa del filtro en un número`}))});
    const ap = densa.map(n => aporte[n]), maxAp = Math.max(...ap.map(Math.abs)) || 1;
    capas.push({id: 'densa', titulo: 'Densa', forma: `${nDensa} neuronas`, mostrados: `${densa.length} de ${nDensa}`,
      nodos: densa.map((n, k) => ({v: redondo(Math.abs(ap[k]) / maxAp), signo: Math.sign(ap[k]), tit: `Neurona ${n + 1} de ${nDensa}`,
        txt: `Activación ${fijo(oculta[n], 2)} · aporta ${ap[k] >= 0 ? '+' : ''}${fijo(ap[k], 2)} al logit`}))});
    const p = sigmoide(logitRed);
    capas.push({id: 'salida', titulo: 'Salida', forma: '1 logit', nodos: [{p: Math.round(p * 1e4) / 1e4, tit: 'Probabilidad de pCR de este corte',
      txt: `logit ${logitRed >= 0 ? '+' : ''}${fijo(logitRed, 2)} → σ = ${fijo(p, 3)}. Red 1 de 5, sin reflexiones ni calibrar; la predicción final promedia cortes y redes.`}]});
    const posicion = new Map(canales4.map((c, k) => [c, k]));
    const lista = [
      Array.from({length: 3}, (_, i) => Array.from({length: nVistas}, (_, j) => [i, j, 0.5, 0, j])).flat(),
      Array.from({length: nVistas}, (_, v) => Array.from({length: 6}, (_, k) => [v, k, 0.5, 0, v])).flat(),
      ...pares.map(par => aristas(par)),
      pool.map((f, k) => [posicion.get(f % nCanales4), k, redondo(impPool[k] / maxPool), 0]),
      aristas(haciaDensa.map(fila => pool.map(f => fila[f])), 4, true),
      ap.map((a, k) => [k, 0, redondo(Math.abs(a) / maxAp), Math.sign(a)]),
    ];
    return {capas, aristas: lista, p, logit: logitRed};
  }

  function bloquesRed(d, red) {
    const capas = Object.fromEntries(red.capas.map(c => [c.id, c]));
    const {lado, anchos} = d.config;
    const mapas = [['fases', 256, 3], ['canales', lado, 6], ...anchos.map((c, i) => [`b${i + 1}`, lado >> (i + 1), c])];
    const salida = mapas.map(([id, s, canales], i) => ({tipo: 'mapas', lado: s, canales, pixelado: s <= 32,
      imgs: capas[id].nodos.map(n => n.img[0]), mostrados: capas[id].nodos.length, ...d.capas[i]}));
    const final = capas.salida.nodos[0];
    salida.push({tipo: 'barra', valores: capas.pool.nodos.map(n => n.v), ...d.capas[6]});
    salida.push({tipo: 'neuronas', valores: capas.densa.nodos.map(n => n.v), signos: capas.densa.nodos.map(n => n.signo), ...d.capas[7]});
    salida.push({tipo: 'salida', p: final.p, ...d.capas[8]});
    return salida;
  }

  const json = v => JSON.stringify(v).replaceAll('<', '\\u003c');
  const codigoRed = (d, datos, alto) => d.plantillas.red.replace('__ALTO__', alto).replace('__DATOS__', () => json(datos));
  const codigoBloques = (d, datos, alto) => d.plantillas.bloques.replace('__ALTO__', alto).replace('__DATOS__', () => json(datos));
  const codigoVisor = (d, cortes, titulo, alto, inicial) => d.plantillas.visor.replaceAll('__ALTO__', alto).replaceAll('__LADO__', alto)
    .replace('__CORTES__', () => json(cortes)).replace('__META__', () => json({titulo})).replace('__INICIAL__', inicial);
  const conMarco = (html, codigo) => {
    const caja = document.createElement('template');
    caja.innerHTML = html;
    caja.content.querySelector('iframe').srcdoc = codigo;
    return caja.innerHTML;
  };

  async function cortesDeEjemplo(e, avisar) {
    avisar?.('Cargando las imágenes de la paciente…');
    return Promise.all(e.cortes.map(c => Promise.all(c.rutas.map(async r => {
      const respuesta = await descargar(`ejemplos/${r}`, 'No se ha podido cargar una imagen de ejemplo.');
      return leerPNG(await respuesta.arrayBuffer(), r);
    }))));
  }

  window.recorridoEstatico = async (paciente, corte) => {
    const d = await datosInfo();
    const e = d.ejemplos.find(x => x.id === paciente);
    const cortes = await cortesDeEjemplo(e);
    await cargarOrt();
    const k = corte !== null && corte < cortes.length ? corte : Math.floor(cortes.length / 2);
    const red = await recorrido(cortes[k]);
    return {red: alto => codigoRed(d, red, alto), bloques: alto => codigoBloques(d, bloquesRed(d, red), alto)};
  };

  // ---------------------------------------------------------------- Predecir
  let actual = null;      // {clave, cortes, nombres, pred, etiqueta, titulo, clinica}

  function clinicaDeFormulario(formulario) {
    const valorDe = etiqueta => {
      const nombre = {HR: 'hr_', HER2: 'her2_'}[etiqueta];
      const marcado = formulario.querySelector(`input[type=radio][name^="${nombre}"]:checked`);
      return {Positivo: 1, Negativo: 0}[marcado?.value] ?? null;
    };
    const numero = (nombre, interruptor) => {
      const activo = formulario.querySelector(`[data-interruptor="${interruptor}"]`)?.checked;
      const campo = formulario.querySelector(`input[name="${nombre}"]`);
      if (!activo) return null;
      const v = Number(campo.value);
      if (!Number.isFinite(v) || v < Number(campo.min) || v > Number(campo.max)) {
        throw new Error(`${campo.labels[0]?.textContent ?? nombre}: introduce un valor entre ${campo.min} y ${campo.max}.`);
      }
      return v;
    };
    return {edad: numero('edad_anos', 'edad_conocida'), volumen: numero('volumen_tumoral_cm3', 'volumen_conocido'), HR: valorDe('HR'), HER2: valorDe('HER2')};
  }

  function ponerClinica(formulario, c) {
    const interruptor = (nombre, campo, valor, defecto) => {
      const caja = formulario.querySelector(`[data-interruptor="${nombre}"]`);
      const entrada = formulario.querySelector(`input[name="${campo}"]`);
      caja.checked = valor !== null;
      caja.closest('.control').querySelector('input[type=hidden]').value = valor !== null ? '1' : '0';
      entrada.disabled = valor === null;
      entrada.value = valor ?? defecto;
    };
    interruptor('edad_conocida', 'edad_anos', c.edad, 50);
    interruptor('volumen_conocido', 'volumen_tumoral_cm3', c.volumen, 10);
    const radio = (prefijo, v) => {
      const texto = v === null ? 'Desconocido' : v === 1 ? 'Positivo' : 'Negativo';
      formulario.querySelectorAll(`input[type=radio][name^="${prefijo}"]`).forEach(r => {r.checked = r.value === texto;});
    };
    radio('hr_', c.HR); radio('her2_', c.HER2);
  }

  function agrupar(archivos) {
    const grupos = new Map(), errores = [];
    for (const a of archivos) {
      const m = NOMBRE_FASE.exec(a.name);
      if (!m) {errores.push(`${a.name}: el nombre debe terminar en _PRE.png, _EARLY.png o _LATE.png`); continue;}
      if (!grupos.has(m[1])) grupos.set(m[1], {});
      grupos.get(m[1])[m[2].toUpperCase()] = a;
    }
    for (const [base, fases] of grupos) if (Object.keys(fases).length !== 3) {
      errores.push(`${base}: faltan fases (${FASES.filter(f => !fases[f]).sort().join(', ')})`);
      grupos.delete(base);
    }
    return {grupos: [...grupos].sort(([a], [b]) => (a < b ? -1 : 1)), errores};
  }

  async function leerSubidos(archivos, avisar) {
    if (archivos.length > MAXIMO_ARCHIVOS) throw new Error('El máximo es 32 cortes (96 archivos PNG) por predicción.');
    if (archivos.reduce((s, a) => s + a.size, 0) > MAXIMO_BYTES) throw new Error('Los archivos superan el límite de 48 MB por petición.');
    const vistos = new Set();
    for (const a of archivos) {
      const n = a.name.toLowerCase();
      if (vistos.has(n)) throw new Error('Hay nombres de archivo duplicados. Revisa las fases de cada corte.');
      vistos.add(n);
    }
    const {grupos, errores} = agrupar(archivos);
    if (!grupos.length) return {errores, cortes: [], nombres: []};
    avisar?.('Leyendo los PNG…');
    try {
      const cortes = await Promise.all(grupos.map(([, f]) => Promise.all(FASES.map(async fase => leerPNG(await f[fase].arrayBuffer(), f[fase].name)))));
      return {errores, cortes, nombres: grupos.map(([b]) => b)};
    } catch (error) {
      return {errores: [...errores, `Formato no válido: ${error.message}`], cortes: [], nombres: []};
    }
  }

  function textoInferencia(pred) {
    const hilos = ort.env.wasm.numThreads ?? 1;
    return `Inferencia en este navegador · WebAssembly (CPU, ${hilos} hilo${hilos === 1 ? '' : 's'}) · ${fijo(pred.segundos, 1)} s · `
      + `sin gradientes, modo evaluación · modelo ${info_nombre} · SHA-256 ${info_huella.slice(0, 12)}`;
  }
  let info_nombre = '', info_huella = '';

  async function fragmento(d, clinicaEntrada, corteRecorrido, errores = []) {
    const t = d.plantillas;
    const avisos = errores.map(e => `<div class="nota atencion" role="status">${escapar(e)}</div>`).join('');
    const ejemplo = actual?.etiqueta !== null && actual?.etiqueta !== undefined;
    const subida = origenActual === 'Subir PNG';
    const partes = [t.cabecera, subida ? t.origen_subida : t.origen_ejemplo];
    partes.push(subida ? t.fuente_subida.replace('</small>', `${archivosNumero ? ` · ${archivosNumero} archivos seleccionados` : ''}</small>`) : marcarSelector(t.fuente_ejemplo, actual?.clave), avisos);
    if (!actual) {
      partes.push(t.seccion_clinica, t.clinica, t.seccion_lectura, t.vacio, t.pie);
      return partes.join('');
    }
    if (ejemplo) partes.push(t.aviso_ejemplo);
    const {pred, nombres, cortes} = actual;
    const pCortes = pred.porCorte.map(p => calibrar(d, p));
    const orden = [...pCortes.keys()].sort((a, b) => pCortes[a] - pCortes[b]);
    const inicial = orden[Math.floor(pCortes.length / 2)];
    actual.visor ??= (() => {
      const escala = escalaRealce(cortes);
      return nombres.map((nombre, i) => ({nombre, p: pCortes[i], PRE: urlGris(cortes[i][0], LADO, LADO), EARLY: urlGris(cortes[i][1], LADO, LADO), LATE: urlGris(cortes[i][2], LADO, LADO), REALCE: urlRealce(cortes[i], escala)}));
    })();
    partes.push(conMarco(t.marco_visor, codigoVisor(d, actual.visor, actual.titulo, 500, inicial)));
    partes.push(`<p class="texto-auxiliar">${nombres.length} corte(s) válidos. Las ventanas discontinuas son las tres vistas centrales que recibe la red (256, 128 y 64 px). La probabilidad de cada corte ya está calibrada.</p>`);
    partes.push(t.seccion_clinica, t.clinica, t.seccion_lectura);

    const c = clinicaEntrada;
    const pImg = calibrar(d, pred.pCruda);
    const calibradas = pred.porRed.map(p => calibrar(d, p));
    const m = media(calibradas), dispersion = Math.sqrt(media(calibradas.map(v => (v - m) ** 2)));
    const hayClinica = Object.values(c).some(v => v !== null);
    let referencia = {p: pImg, clave: 'imagen', nombre: 'solo imagen'};
    const colImagen = lectura(d, 'Solo imagen · CNN', pImg, d.perfiles.imagen, d.umbral, `${nombres.length} corte(s) × 5 redes × 4 reflexiones · desacuerdo entre redes ± ${fijo(dispersion, 3)}`);
    let colFusion = lecturaVacia('Imagen + clínica', 'Añade al menos un dato clínico para ver la estimación combinada.');
    if (hayClinica) {
      const r = decidir(d, pred.pCruda, c);
      const perfiles = d.perfiles[r.submodelo] ?? d.perfiles.imagen;
      if (d.perfiles[r.submodelo]) referencia = {p: r.p, clave: r.submodelo, nombre: 'imagen + clínica'};
      const cambio = 100 * (r.p - pImg);
      colFusion = lectura(d, 'Imagen + clínica', r.p, perfiles, r.umbral, `submodelo ${escapar(r.submodelo)} · según los datos disponibles`,
        ` Con la clínica, la estimación cambia <b>${(cambio >= 0 ? '+' : '') + fijo(cambio, 1)} puntos</b>.`);
    }
    partes.push(`${t.columnas_inicio}<div class="columna">${colImagen}</div><div class="columna">${colFusion}</div></div>`);
    partes.push(`<p class="texto-auxiliar" id="info-inferencia">${textoInferencia(pred)}</p>`);
    if (ejemplo) partes.push(`<p class="texto-auxiliar">Resultado real de esta paciente de entrenamiento: ${actual.etiqueta ? 'pCR' : 'no pCR'}.</p>`);
    const ref = d.perfiles[referencia.clave];
    partes.push(graficaDistribucion(d, referencia.clave, referencia.p, `Dónde cae entre las pacientes de desarrollo · ${referencia.nombre}`, `esta paciente · ${fijo(100 * referencia.p, 1)} %`));
    partes.push(`<p class="texto-auxiliar">Probabilidades fuera de fold de las ${ref.n.toLocaleString('de-DE')} pacientes de desarrollo con el mismo modelo y los mismos datos disponibles. La línea blanca es esta estimación; las zonas de color, los cinco perfiles. Pasa el ratón por las barras para ver cuántas pacientes de cada tramo alcanzaron pCR.</p>`);
    partes.push(t.seccion_red);
    const k = corteRecorrido !== null && corteRecorrido < nombres.length ? corteRecorrido : inicial;
    if (nombres.length > 1) {
      const opciones = nombres.map((n, i) => [String(i), `${n} · p ${pct(pCortes[i], 1)}`]);
      partes.push(t.rango.replace(/min="0" max="0"/, `min="0" max="${nombres.length - 1}"`).replace(/ value="0"/, ` value="${k}"`)
        .replace(/data-opciones='[^']*'/, `data-opciones='${escapar(JSON.stringify(opciones))}'`)
        .replace(/name="corte_que_se_propaga" value="[^"]*"/, `name="corte_que_se_propaga" value="${k}"`)
        .replaceAll('__TEXTO__', opciones[k][1]).replace(/<div class="extremos">.*?<\/div>/, `<div class="extremos"><span>${opciones[0][1]}</span><span>${opciones.at(-1)[1]}</span></div>`));
    }
    actual.redes ??= new Map();
    if (!actual.redes.has(k)) actual.redes.set(k, await recorrido(cortes[k]));
    partes.push(conMarco(t.marco_red, codigoRed(d, actual.redes.get(k), 600)), t.final);
    return partes.join('');
  }

  const marcarSelector = (html, id) => {
    const caja = document.createElement('template');
    caja.innerHTML = html;
    const selector = caja.content.querySelector('select');
    if (selector && id) selector.value = id;
    selector?.querySelectorAll('option').forEach(o => o.toggleAttribute('selected', o.value === id));
    return caja.innerHTML;
  };

  let origenActual = 'Paciente de ejemplo', archivosNumero = 0, clinicaPendiente = null, firmaArchivos = '', clinicaVigente = null, modeloListo = false;
  // La página se recompone en cada cambio: los controles clínicos recuperan aquí su valor.
  document.addEventListener('panel:actualizado', () => {
    if (clinicaVigente) ponerClinica(document.getElementById('panel-form'), clinicaVigente);
  });
  const avisar = texto => {
    const estado = document.getElementById('estado-peticion');
    estado.textContent = texto; estado.hidden = false;
  };

  async function cargarEjemplo(d, id) {
    const e = d.ejemplos.find(x => x.id === id) ?? d.ejemplos[0];
    const cortes = await cortesDeEjemplo(e, avisar);
    const pred = await predecir(cortes, avisar);
    actual = {clave: e.id, cortes, nombres: e.cortes.map(c => c.nombre), pred, etiqueta: e.pcr, titulo: `${e.cohorte} · ${escapar(e.id)}`};
    clinicaPendiente = {edad: e.edad, volumen: e.volumen, HR: e.HR, HER2: e.HER2};
  }

  window.predecirEstatico = async (datos, control) => {
    const d = await datosInfo();
    info_nombre = d.modelo.nombre; info_huella = d.modelo.sha256;
    await cargarOrt();
    if (!modeloListo) avisar('Preparando el modelo; la primera vez puede tardar un minuto…');
    await modelo();
    modeloListo = true;
    const formulario = document.getElementById('panel-form');
    const origen = datos.get('origen') ?? 'Paciente de ejemplo';
    const cambiaOrigen = origen !== origenActual;
    const casoAnterior = actual;
    origenActual = origen;
    let errores = [];
    if (origen === 'Subir PNG') {
      const archivos = datos.getAll('archivos').filter(a => a.name);
      const firma = archivos.map(a => `${a.name}:${a.size}:${a.lastModified}`).join('|');
      if (cambiaOrigen || firma !== firmaArchivos) {
        firmaArchivos = firma;
        archivosNumero = archivos.length;
        actual = null;
        if (archivos.length) {
          const leido = await leerSubidos(archivos, avisar);
          errores = leido.errores;
          if (leido.cortes.length) {
            const pred = await predecir(leido.cortes, avisar);
            actual = {clave: 'subida', cortes: leido.cortes, nombres: leido.nombres, pred, etiqueta: null, titulo: 'subida'};
            clinicaPendiente = {edad: null, volumen: null, HR: null, HER2: null};
          }
        }
      }
    } else {
      const id = datos.get('paciente_de_ejemplo_train') ?? d.ejemplos[0].id;
      const cambiaPaciente = cambiaOrigen || (actual ? actual.clave !== id : control?.name === 'paciente_de_ejemplo_train');
      if (cambiaPaciente || !actual) await cargarEjemplo(d, id);
      // En el caso inicial, la clínica del formulario puede llevar ya cambios de quien usa la página.
      if (!cambiaPaciente) clinicaPendiente = null;
    }
    const clinica = clinicaPendiente ?? clinicaDeFormulario(formulario);
    clinicaPendiente = null;
    avisar('Componiendo la lectura…');
    const corte = actual === casoAnterior && datos.has('corte_que_se_propaga') ? Number(datos.get('corte_que_se_propaga')) : null;
    const html = await fragmento(d, clinica, corte, errores);
    clinicaVigente = clinica;
    return html;
  };

  // Predecir descarga los modelos al abrirse para que la primera predicción no tenga que esperarlos.
  if (window.PAGINA_ESTATICA === 'prediccion') {
    window.addEventListener('load', () => modelo().then(modeloRecorrido).catch(() => {}));
  }

  // API de contraste con Python; la usa la prueba de equivalencia.
  window.motorEstatico = {datosInfo, leerPNG, cortesDeEjemplo, predecir, calibrar, decidir, lectura, recorrido, cargarOrt, modelo};
})();
