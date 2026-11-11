'use strict';
/* Shared by the figures of the Autograd page: the example graph  d = a * b,  e = d + a,  L = e ** 2
   and the code that draws it. Everything hangs off one global, AG. */
window.AG = (() => {
  // Positions in fractions of the canvas; `from` lists the inputs of a non-leaf tensor.
  const NODES = {
    a: { x: 0.12, y: 0.24, leaf: true },
    b: { x: 0.12, y: 0.78, leaf: true },
    d: { x: 0.40, y: 0.66, op: '×', from: ['a', 'b'], fn: 'Multiply' },
    e: { x: 0.66, y: 0.36, op: '+', from: ['d', 'a'], fn: 'Add' },
    L: { x: 0.90, y: 0.36, op: 'x²', from: ['e'], fn: 'Pow' },
  };
  const ORDER = ['d', 'e', 'L'];                       // order in which the forward pass creates them

  const fmt = v => { const r = Math.round(v * 100) / 100; return (Object.is(r, -0) ? 0 : r).toString().replace('-', '−'); };

  function forward(a, b) { const d = a * b, e = d + a, L = e * e; return { a, b, d, e, L }; }

  /* The backward pass, one processed tensor per step. Each step lists the contributions it sends to its inputs. */
  function backwardSteps(v) {
    return [
      { node: 'L', sends: [{ to: 'e', local: 2 * v.e, text: `2·e = ${fmt(2 * v.e)}` }] },
      { node: 'e', sends: [{ to: 'd', local: 1, text: '1' }, { to: 'a', local: 1, text: '1' }] },
      { node: 'd', sends: [{ to: 'a', local: v.b, text: `b = ${fmt(v.b)}` }, { to: 'b', local: v.a, text: `a = ${fmt(v.a)}` }] },
    ];
  }

  function box(w) { const bw = Fig.clamp(w * 0.17, 54, 96); return { bw, bh: bw * 0.78 }; }
  function pos(n, w, h) { return { x: NODES[n].x * w, y: NODES[n].y * h }; }

  /* Point on the border of the box of node n, towards the point (tx, ty). */
  function edgePoint(n, tx, ty, w, h) {
    const p = pos(n, w, h), { bw, bh } = box(w), dx = tx - p.x, dy = ty - p.y;
    const s = Math.min((bw / 2) / Math.max(1e-6, Math.abs(dx)), (bh / 2) / Math.max(1e-6, Math.abs(dy)));
    return { x: p.x + dx * s, y: p.y + dy * s };
  }
  function edge(from, to, w, h) {
    const p = pos(from, w, h), q = pos(to, w, h);
    return { p0: edgePoint(from, q.x, q.y, w, h), p1: edgePoint(to, p.x, p.y, w, h) };
  }

  function arrow(ctx, p0, p1, t, color, width) {
    const x = Fig.lerp(p0.x, p1.x, t), y = Fig.lerp(p0.y, p1.y, t);
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(x, y); ctx.stroke();
    if (t < 1) return;
    const ang = Math.atan2(p1.y - p0.y, p1.x - p0.x), s = 7;
    ctx.beginPath(); ctx.moveTo(x, y);
    ctx.lineTo(x - s * Math.cos(ang - 0.4), y - s * Math.sin(ang - 0.4));
    ctx.lineTo(x - s * Math.cos(ang + 0.4), y - s * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();
  }

  /* Draws the graph.
       values   { a, b, d, e, L }
       shown    which tensors exist already, e.g. { a: 1, b: 1, d: 1 }
       grow     { node, t }  the edges into this tensor are drawn up to fraction t (forward animation)
       grads    { a: 64, ... } gradients to print; missing ones are not shown
       active   tensor to highlight
       pulses   [{ from, to, t }]  dots travelling against the edge direction (backward animation)  */
  function draw(cv, s) {
    const { ctx, w, h } = Fig.fit(cv), { bw, bh } = box(w);
    const fg = Fig.token('--fg'), muted = Fig.token('--muted'), rule = Fig.token('--rule'), accent = Fig.token('--accent');
    const surface = Fig.token('--surface'), soft = Fig.token('--accent-soft'), hole = Fig.token('--hole');
    const small = w < 480, fName = small ? 13 : 15, fVal = small ? 11.5 : 13;

    for (const n of ORDER) {
      if (!s.shown[n]) continue;
      const t = s.grow && s.grow.node === n ? s.grow.t : 1;
      for (const src of NODES[n].from) { const { p0, p1 } = edge(src, n, w, h); arrow(ctx, p0, p1, t, muted, 1.4); }
    }
    for (const pu of s.pulses || []) {
      const { p0, p1 } = edge(pu.to, pu.from, w, h);       // edge runs input -> output, the pulse runs back
      const t = Fig.ease(Fig.clamp(pu.t, 0, 1)), x = Fig.lerp(p1.x, p0.x, t), y = Fig.lerp(p1.y, p0.y, t);
      ctx.strokeStyle = hole; ctx.lineWidth = 2.4;
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(x, y); ctx.stroke();
      ctx.fillStyle = hole; ctx.beginPath(); ctx.arc(x, y, 4.5, 0, 2 * Math.PI); ctx.fill();
    }
    for (const n of Object.keys(NODES)) {
      if (!s.shown[n]) continue;
      const p = pos(n, w, h), node = NODES[n], on = s.active === n, x = p.x - bw / 2, y = p.y - bh / 2;
      ctx.fillStyle = on ? soft : surface; ctx.strokeStyle = on ? accent : (node.leaf ? fg : rule); ctx.lineWidth = on ? 2 : 1.3;
      ctx.beginPath(); ctx.roundRect(x, y, bw, bh, 6); ctx.fill(); ctx.stroke();
      ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      ctx.fillStyle = fg; ctx.font = `600 ${fName}px ${Fig.DISPLAY}`;
      ctx.fillText(n, x + 8, y + bh * 0.24);
      if (node.op) { ctx.textAlign = 'right'; ctx.fillStyle = muted; ctx.font = `500 ${fVal}px ${Fig.MONO}`; ctx.fillText(node.op, x + bw - 8, y + bh * 0.24); }
      ctx.textAlign = 'left'; ctx.font = `500 ${fVal}px ${Fig.MONO}`;
      ctx.fillStyle = fg; ctx.fillText(fmt(s.values[n]), x + 8, y + bh * 0.54);
      if (s.grads && s.grads[n] !== undefined) { ctx.fillStyle = hole; ctx.fillText('∇ ' + fmt(s.grads[n]), x + 8, y + bh * 0.82); }
    }
  }

  return { NODES, ORDER, fmt, forward, backwardSteps, draw };
})();
