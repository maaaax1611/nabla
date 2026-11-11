'use strict';
/* Shared by the figures of the Convolutional networks page: number formatting, colours and grids of numbered cells.
   Everything hangs off one global, CNN. */
window.CNN = (() => {
  const fmt = (v, d = 0) => { const r = (Math.round(v * 10 ** d) / 10 ** d).toFixed(d); return (+r === 0 ? (0).toFixed(d) : r).replace('-', '−'); };
  /* Shortest readable form: integers without decimals, others with up to two. */
  const num = v => Number.isInteger(Math.round(v * 100) / 100) ? fmt(v, 0) : fmt(v, Math.abs(v) < 10 ? 2 : 1);

  function hex(c) { c = c.replace('#', ''); if (c.length === 3) c = c.split('').map(x => x + x).join(''); const v = parseInt(c, 16); return [v >> 16 & 255, v >> 8 & 255, v & 255]; }
  function mix(c0, c1, t) { const a = hex(c0), b = hex(c1); t = Fig.clamp(t, 0, 1); return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(',')})`; }

  /* Colour for a signed value: surface at 0, towards accent for positive and hole for negative values. */
  function diverge(v, max) { const s = Fig.token('--surface'); return v >= 0 ? mix(s, Fig.token('--accent'), 0.75 * v / (max || 1)) : mix(s, Fig.token('--hole'), 0.75 * -v / (max || 1)); }
  /* One of four distinct colours, for windows and patches. */
  const tint = k => Fig.token('--bin' + (k % 4));

  /* One cell with a number. o: { fill, stroke, width, text, color, dash, size } */
  function cell(ctx, x, y, cs, o) {
    const pad = Math.max(1, cs * 0.05);
    ctx.beginPath(); ctx.roundRect(x + pad, y + pad, cs - 2 * pad, cs - 2 * pad, Math.min(5, cs * 0.14));
    if (o.fill) { ctx.fillStyle = o.fill; ctx.fill(); }
    if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.width || 1.1; ctx.setLineDash(o.dash || []); ctx.stroke(); ctx.setLineDash([]); }
    if (o.text !== undefined && o.text !== '' && cs >= 15) {
      ctx.fillStyle = o.color || Fig.token('--fg'); ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.font = `500 ${Math.round(Math.min(15, cs * (o.size || 0.36)))}px ${Fig.MONO}`;
      ctx.fillText(o.text, x + cs / 2, y + cs / 2 + 1);
    }
  }

  /* Outline around a block of cells. */
  function outline(ctx, x, y, w, h, color, width) {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); ctx.roundRect(x, y, w, h, 5); ctx.stroke();
  }

  function label(ctx, text, x, y, align = 'left') {
    ctx.fillStyle = Fig.token('--muted'); ctx.font = `600 13px ${Fig.DISPLAY}`; ctx.textAlign = align; ctx.textBaseline = 'bottom'; ctx.fillText(text, x, y);
  }

  /* Seeded random integers in [lo, hi]. */
  function ints(seed) { let s = seed; return (lo, hi) => { s = (s * 16807) % 2147483647; return lo + Math.floor((s - 1) / 2147483646 * (hi - lo + 1)); }; }

  return { fmt, num, mix, diverge, tint, cell, outline, label, ints };
})();
