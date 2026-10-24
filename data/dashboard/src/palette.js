/** Colours: one fixed colour per engine version (Okabe-Ito, colour-blind safe) and quiet single-hue scales. */

// Okabe & Ito (2008), reordered so neighbouring versions contrast on a light background;
// yellow and black come last because they are the weakest against paper and ink.
export const OKABE_ITO = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442', '#000000'];

// Warm ochre to oxblood, light to dark.
export const SEQUENTIAL = ['#F2E6C9', '#E5C27F', '#D39A52', '#B86B35', '#93402A', '#5F2317'];
// Slate blue to paper to oxblood, for signed values such as correlations.
export const DIVERGING = ['#2F5D7C', '#8FB1C6', '#F3EEE4', '#D89A83', '#9C3D26'];
export const RESULT_COLORS = { win: '#4E7D5B', draw: '#BFB4A2', loss: '#A34E36', inconclusive: '#8C8274' };

/** "Tomahawk 1.2.0"; falls back to the id when name and version are missing. */
export function engineLabel(engine) {
  const text = [engine?.name, engine?.version].filter(v => v != null && String(v).trim() !== '').join(' ').trim();
  return text || `Engine ${engine?.id}`;
}

/** Engines ordered by id, each with a unique label and a fixed colour assigned in that order. */
export function assignColors(engines, { palette = OKABE_ITO, dark = false } = {}) {
  const sorted = [...engines].sort((a, b) => Number(a.id) - Number(b.id));
  const seen = new Map();
  return sorted.map((engine, index) => {
    let label = engineLabel(engine);
    if (seen.has(label)) label = `${label} (#${engine.id})`;
    seen.set(label, true);
    let color = palette[index % palette.length];
    if (dark && color === '#000000') color = '#E8E1D3';
    return { ...engine, id: Number(engine.id), label, color, order: index };
  });
}

function hexToRgb(hex) {
  const h = hex.replace('#', '');
  return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16));
}

function rgbToHex(rgb) {
  return '#' + rgb.map(v => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, '0')).join('');
}

/** Piecewise-linear interpolation through a list of hex colours, t in [0, 1]. */
export function interpolate(stops, t) {
  if (!Number.isFinite(t)) return null;
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const a = hexToRgb(stops[i]), b = hexToRgb(stops[i + 1]);
  const f = x - i;
  return rgbToHex(a.map((v, k) => v + (b[k] - v) * f));
}

export const sequentialColor = t => interpolate(SEQUENTIAL, t);
/** Colour for a value in [-limit, limit], centred on paper at zero. */
export const divergingColor = (v, limit = 1) => interpolate(DIVERGING, (v / limit + 1) / 2);

/** Ink or paper text, whichever reads better on the given background. */
export function textOn(background) {
  const [r, g, b] = hexToRgb(background).map(v => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.36 ? '#2B2620' : '#FBF9F4';
}
