'use strict';
/* Shared by the figures of the Neural networks page: activation functions, a seeded random generator
   and a small plot frame. Everything hangs off one global, NN. */
window.NN = (() => {
  const C = Math.sqrt(2 / Math.PI), A = 0.044715;
  const sig = x => 1 / (1 + Math.exp(-x));
  const ACT = {
    relu:    { label: 'ReLU',    f: x => Math.max(0, x), d: x => (x > 0 ? 1 : 0) },
    sigmoid: { label: 'Sigmoid', f: sig, d: x => sig(x) * (1 - sig(x)) },
    tanh:    { label: 'Tanh',    f: Math.tanh, d: x => 1 - Math.tanh(x) ** 2 },
    gelu:    { label: 'GELU',    f: x => 0.5 * x * (1 + Math.tanh(C * (x + A * x ** 3))),
               d: x => { const t = Math.tanh(C * (x + A * x ** 3)); return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * C * (1 + 3 * A * x * x); } },
  };

  /* Reproducible random numbers: u() uniform in (0, 1), g() standard normal. */
  function rng(seed) {
    let s = seed;
    const u = () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
    const g = () => { let a = 0; while (a === 0) a = u(); return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * u()); };
    return { u, g };
  }

  const fmt = (v, d = 2) => { const r = v.toFixed(d); return (+r === 0 ? (0).toFixed(d) : r).replace('-', '−'); };
  const sup = n => String(n).replace('-', '⁻').replace(/\d/g, c => '⁰¹²³⁴⁵⁶⁷⁸⁹'[c]);

  function hex(c) { c = c.replace('#', ''); if (c.length === 3) c = c.split('').map(x => x + x).join(''); const v = parseInt(c, 16); return [v >> 16 & 255, v >> 8 & 255, v & 255]; }
  /* Colour between two hex colours, t in [0, 1]. */
  function mix(c0, c1, t) { const a = hex(c0), b = hex(c1); return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(',')})`; }

  /* Draws grid, ticks and an optional title into the box { x, y, w, h } and returns the mappings px, py.
     o: { xmin, xmax, ymin, ymax, xticks, yticks, xfmt, yfmt, title } */
  function frame(ctx, box, o) {
    const px = v => box.x + (v - o.xmin) / (o.xmax - o.xmin) * box.w, py = v => box.y + box.h - (v - o.ymin) / (o.ymax - o.ymin) * box.h;
    const grid = Fig.token('--grid'), muted = Fig.token('--muted');
    ctx.lineWidth = 1; ctx.font = `500 11px ${Fig.MONO}`;
    for (const v of o.xticks || []) {
      ctx.strokeStyle = grid; ctx.beginPath(); ctx.moveTo(px(v), box.y); ctx.lineTo(px(v), box.y + box.h); ctx.stroke();
      ctx.fillStyle = muted; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText((o.xfmt || String)(v), px(v), box.y + box.h + 5);
    }
    for (const v of o.yticks || []) {
      ctx.strokeStyle = grid; ctx.beginPath(); ctx.moveTo(box.x, py(v)); ctx.lineTo(box.x + box.w, py(v)); ctx.stroke();
      ctx.fillStyle = muted; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText((o.yfmt || String)(v), box.x - 6, py(v));
    }
    if (o.title) { ctx.font = `600 13px ${Fig.DISPLAY}`; ctx.fillStyle = muted; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText(o.title, box.x, box.y - 6); }
    return { px, py };
  }

  return { ACT, sig, rng, fmt, sup, mix, frame };
})();
