'use strict';
/* Shared by the figures of the Transformers page: formatting, colours, numbered cells, seeded random numbers,
   softmax. Everything hangs off one global, TF. */
window.TF = (() => {
  const fmt = (v, d = 1) => { const r = (Math.round(v * 10 ** d) / 10 ** d).toFixed(d); return (+r === 0 ? (0).toFixed(d) : r).replace('-', '−'); };
  function hex(c) { c = c.replace('#', ''); if (c.length === 3) c = c.split('').map(x => x + x).join(''); const v = parseInt(c, 16); return [v >> 16 & 255, v >> 8 & 255, v & 255]; }
  function mix(c0, c1, t) { const a = hex(c0), b = hex(c1); t = Fig.clamp(t, 0, 1); return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(',')})`; }
  /* Signed value → colour: surface at 0, accent for positive, hole for negative. */
  function diverge(v, max) { const s = Fig.token('--surface'); return v >= 0 ? mix(s, Fig.token('--accent'), 0.8 * v / (max || 1)) : mix(s, Fig.token('--hole'), 0.8 * -v / (max || 1)); }

  function cell(ctx, x, y, cs, o) {
    const pad = Math.max(1, cs * 0.05);
    ctx.beginPath(); ctx.roundRect(x + pad, y + pad, cs - 2 * pad, cs - 2 * pad, Math.min(5, cs * 0.14));
    if (o.fill) { ctx.fillStyle = o.fill; ctx.fill(); }
    if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.width || 1.1; ctx.stroke(); }
    if (o.text !== undefined && o.text !== '' && cs >= 16) {
      ctx.fillStyle = o.color || Fig.token('--fg'); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.font = `500 ${Math.round(Math.min(14, cs * (o.size || 0.32)))}px ${Fig.MONO}`;
      ctx.fillText(o.text, x + cs / 2, y + cs / 2 + 1);
    }
  }
  function outline(ctx, x, y, w, h, color, width) { ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); ctx.roundRect(x, y, w, h, 5); ctx.stroke(); }
  function label(ctx, text, x, y, align = 'left', color) {
    ctx.fillStyle = color || Fig.token('--muted'); ctx.font = `600 13px ${Fig.DISPLAY}`; ctx.textAlign = align; ctx.textBaseline = 'bottom'; ctx.fillText(text, x, y);
  }
  function rng(seed) {
    let s = seed;
    const u = () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
    const g = () => { let a = 0; while (a === 0) a = u(); return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * u()); };
    return { u, g };
  }
  function softmax(s) { const m = Math.max(...s.filter(isFinite)); const e = s.map(v => isFinite(v) ? Math.exp(v - m) : 0); const z = e.reduce((a, b) => a + b, 0); return e.map(v => v / z); }

  return { fmt, mix, diverge, cell, outline, label, rng, softmax };
})();
