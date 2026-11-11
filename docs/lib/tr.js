'use strict';
/* Shared by the figures of the Training page: formatting, seeded random numbers, labels and arrows.
   Everything hangs off one global, TR. */
window.TR = (() => {
  const fmt = (v, d = 2) => { if (!isFinite(v)) return v > 0 ? '∞' : '−∞'; const r = (Math.round(v * 10 ** d) / 10 ** d).toFixed(d); return (+r === 0 ? (0).toFixed(d) : r).replace('-', '−'); };
  const sci = v => { if (v === 0) return '0'; const e = Math.floor(Math.log10(Math.abs(v))), m = v / 10 ** e; return `${fmt(m, 2)}·10${String(e).replace('-', '⁻').replace(/\d/g, c => '⁰¹²³⁴⁵⁶⁷⁸⁹'[c])}`; };
  function rng(seed) {
    let s = seed;
    const u = () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; };
    const g = () => { let a = 0; while (a === 0) a = u(); return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * u()); };
    return { u, g };
  }
  function label(ctx, text, x, y, align = 'left', color) {
    ctx.fillStyle = color || Fig.token('--muted'); ctx.font = `600 13px ${Fig.DISPLAY}`; ctx.textAlign = align; ctx.textBaseline = 'bottom'; ctx.fillText(text, x, y);
  }
  function arrow(ctx, x0, y0, x1, y1, color, width = 2, head = 9) {
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const a = Math.atan2(y1 - y0, x1 - x0); if (Math.hypot(x1 - x0, y1 - y0) < 2) return;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - head * Math.cos(a - 0.4), y1 - head * Math.sin(a - 0.4)); ctx.lineTo(x1 - head * Math.cos(a + 0.4), y1 - head * Math.sin(a + 0.4)); ctx.closePath(); ctx.fill();
  }
  /* Grid, ticks and title in box { x, y, w, h }; returns px, py. o: { xmin, xmax, ymin, ymax, xticks, yticks, xfmt, yfmt, title } */
  function frame(ctx, box, o) {
    const px = v => box.x + (v - o.xmin) / (o.xmax - o.xmin) * box.w, py = v => box.y + box.h - (v - o.ymin) / (o.ymax - o.ymin) * box.h;
    const grid = Fig.token('--grid'), muted = Fig.token('--muted');
    ctx.lineWidth = 1; ctx.font = `500 11px ${Fig.MONO}`;
    for (const v of o.xticks || []) { ctx.strokeStyle = grid; ctx.beginPath(); ctx.moveTo(px(v), box.y); ctx.lineTo(px(v), box.y + box.h); ctx.stroke(); ctx.fillStyle = muted; ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText((o.xfmt || String)(v), px(v), box.y + box.h + 5); }
    for (const v of o.yticks || []) { ctx.strokeStyle = grid; ctx.beginPath(); ctx.moveTo(box.x, py(v)); ctx.lineTo(box.x + box.w, py(v)); ctx.stroke(); ctx.fillStyle = muted; ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText((o.yfmt || String)(v), box.x - 6, py(v)); }
    if (o.title) label(ctx, o.title, box.x, box.y - 6);
    return { px, py };
  }
  return { fmt, sci, rng, label, arrow, frame };
})();
