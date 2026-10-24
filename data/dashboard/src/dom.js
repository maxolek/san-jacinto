/** DOM element helper and number formatting (no chart code, so the entry chunk stays small). */

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, val] of Object.entries(attrs ?? {})) {
    if (val == null || val === false) continue;
    if (key === 'class') node.className = val;
    else if (key === 'style' && typeof val === 'object') Object.assign(node.style, val);
    else if (key.startsWith('on') && typeof val === 'function') node.addEventListener(key.slice(2).toLowerCase(), val);
    else if (key === 'html') node.innerHTML = val;
    else node.setAttribute(key, val === true ? '' : val);
  }
  append(node, children);
  return node;
}

function append(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    node.appendChild(typeof child === 'string' || typeof child === 'number' ? document.createTextNode(String(child)) : child);
  }
}

/* ------------------------------------------------------------------ formatting */
const MINUS = '\u2212';
const finite = v => typeof v === 'number' && Number.isFinite(v);
const withMinus = s => s.replace(/^-/, MINUS);

export const fmt = {
  int: v => (finite(v) ? withMinus(Math.round(v).toLocaleString('en-US')) : '\u2014'),
  num: (v, d = 1) => (finite(v) ? withMinus(v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d })) : '\u2014'),
  pct: (v, d = 1) => (finite(v) ? withMinus((v * 100).toFixed(d)) + '%' : '\u2014'),
  signedPct: (v, d = 1) => (finite(v) ? (v > 0 ? '+' : v < 0 ? MINUS : '\u00b1') + Math.abs(v * 100).toFixed(d) + '%' : '\u2014'),
  signed: (v, d = 1) => (finite(v) ? (v > 0 ? '+' : v < 0 ? MINUS : '\u00b1') + Math.abs(v).toFixed(d) : '\u2014'),
  compact(v, d = 1) {
    if (!finite(v)) return '\u2014';
    const a = Math.abs(v);
    const s = a >= 1e9 ? (v / 1e9).toFixed(d) + 'G' : a >= 1e6 ? (v / 1e6).toFixed(d) + 'M'
      : a >= 1e4 ? (v / 1e3).toFixed(d) + 'k' : a >= 100 ? Math.round(v).toLocaleString('en-US') : v.toFixed(a >= 10 ? 1 : 2);
    return withMinus(s);
  },
  ms(v) {
    if (!finite(v)) return '\u2014';
    return v >= 10000 ? (v / 1000).toFixed(1) + ' s' : v >= 1000 ? (v / 1000).toFixed(2) + ' s' : v >= 10 ? Math.round(v) + ' ms' : v.toFixed(1) + ' ms';
  },
  nps: v => (finite(v) ? (v >= 1e6 ? (v / 1e6).toFixed(2) + ' Mnps' : Math.round(v / 1000) + ' knps') : '\u2014'),
  date: ms => (finite(ms) ? new Date(ms).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }) : '\u2014'),
  cp: v => (finite(v) ? withMinus((v > 0 ? '+' : '') + Math.round(v)) + ' cp' : '\u2014'),
};
