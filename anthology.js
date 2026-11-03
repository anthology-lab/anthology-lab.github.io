// anthology: theme, archive filters, random quote, search by word and by meaning (same model as soul)
(function () {
  const root = document.documentElement;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

  // ---------- theme ----------
  const tt = document.querySelector('.theme-toggle');
  const cur = () => root.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  const paint = () => { if (tt) tt.textContent = cur() === 'dark' ? '☀' : '☾'; };
  paint();
  if (tt) tt.onclick = () => { const n = cur() === 'dark' ? 'light' : 'dark'; root.dataset.theme = n; paint(); try { localStorage.setItem('anth-theme', n); } catch (e) {} };

  // ---------- archive filters ----------
  const bar = document.querySelector('.filters');
  const boxes = [...document.querySelectorAll('.items')];
  if (bar && boxes.length) {
    const text = bar.querySelector('.f-text'), sort = bar.querySelector('.f-sort');
    const sels = [...bar.querySelectorAll('select[data-key]')];
    const empty = document.querySelector('.empty');
    const run = () => {
      const w = text.value.trim().toLowerCase();
      let shown = 0;
      boxes.forEach(box => {
        const items = [...box.querySelectorAll('.item')];
        let here = 0;
        items.forEach(it => {
          const group = it.closest('.agroup');
          const hay = (it.textContent + ' ' + (group ? group.querySelector('.aname').textContent : '')).toLowerCase();
          const ok = (!w || hay.includes(w)) && sels.every(s => !s.value || (s.dataset.key === 'tag' ? it.dataset.tag.includes('|' + s.value + '|') : it.dataset[s.dataset.key] === s.value));
          it.hidden = !ok; if (ok) { shown++; here++; }
        });
        // commonplace: hide an author with no quote left, and a part with nothing in it
        box.querySelectorAll('.agroup').forEach(g => { g.hidden = !g.querySelector('.item:not([hidden])'); });
        const part = box.previousElementSibling;
        if (part && part.classList.contains('part')) { part.hidden = !here; box.hidden = !here; }
        if (sort) {
          const k = sort.value;
          items.slice().sort((a, b) => k === 'date' ? b.dataset.date.localeCompare(a.dataset.date) : a.dataset[k].localeCompare(b.dataset[k]))
            .forEach(it => box.appendChild(it));
        }
      });
      if (empty) empty.hidden = shown > 0;
    };
    text.addEventListener('input', run);
    [...sels, sort].filter(Boolean).forEach(s => s.addEventListener('change', run));
  }

  // ---------- commonplace: one quote at random ----------
  const feat = document.getElementById('featured');
  if (feat && window.ANTH_QUOTES && ANTH_QUOTES.length) {
    let last = -1;
    const show = () => {
      let i; do { i = Math.floor(Math.random() * ANTH_QUOTES.length); } while (ANTH_QUOTES.length > 1 && i === last);
      last = i; const q = ANTH_QUOTES[i];
      feat.querySelector('.fq').textContent = q.q; feat.querySelector('.fa').textContent = q.a;
      feat.querySelector('.fopen').href = q.h;
    };
    document.getElementById('fnext').onclick = show; show();
  }

  // ---------- spoilers: one gate over the whole note ----------
  document.querySelectorAll('[data-gate]').forEach(b => b.addEventListener('click', () => b.closest('.gatebox').classList.add('open')));

  // ---------- home search ----------
  const IDX = window.ANTH_INDEX;
  if (!IDX) return;
  let scope = '';
  const inScope = d => !scope || d.k === scope;
  const one = { books: 'book', films: 'film', podcasts: 'podcast', commonplace: 'quote' };
  const hit = (d, snip) => `<a class="hit" href="${d.h}">${esc(d.t.length > 90 ? d.t.slice(0, 90) + '…' : d.t)} <span class="k">// ${one[d.k]}${d.b ? ' · ' + esc(d.b) : ''}</span>${snip ? `<span class="snip">${esc(snip)}</span>` : ''}</a>`;

  const qw = document.getElementById('q-word'), sw = document.getElementById('s-word'), rw = document.getElementById('r-word');
  const qm = document.getElementById('q-mean'), sm = document.getElementById('s-mean'), rm = document.getElementById('r-mean');

  function wordSearch() {
    const w = qw.value.trim().toLowerCase();
    if (w.length < 2) { sw.textContent = ''; rw.innerHTML = ''; return; }
    const hits = IDX.filter(d => inScope(d) && (d.t + ' ' + d.b + ' ' + d.x).toLowerCase().includes(w));
    sw.textContent = hits.length ? hits.length + (hits.length === 1 ? ' result' : ' results') : 'nothing found';
    rw.innerHTML = hits.slice(0, 12).map(d => {
      const i = d.x.toLowerCase().indexOf(w);
      return hit(d, i < 0 ? '' : '…' + d.x.slice(Math.max(0, i - 50), i + 80) + '…');
    }).join('');
  }
  qw.addEventListener('input', wordSearch);

  document.querySelectorAll('[data-scope]').forEach(b => b.addEventListener('click', () => {
    scope = b.dataset.scope;
    document.querySelectorAll('[data-scope]').forEach(x => x.classList.toggle('on', x === b));
    wordSearch(); if (qm.value.trim().length >= 3) meaningSearch();
  }));

  // search by meaning: all-MiniLM-L6-v2 in a worker, vectors cached in localStorage (as in soul)
  const MODEL = 'Xenova/all-MiniLM-L6-v2', KEY = 'anth-semantic-v1';
  let worker = null, msg = 0; const pending = new Map();
  function getWorker() {
    if (worker) return worker;
    const src = `import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";
let ex = null;
self.onmessage = async (e) => {
  const { id, model, texts } = e.data;
  try {
    if (!ex) ex = pipeline("feature-extraction", model, { dtype: "q8", progress_callback: p => { if (p.status === "progress") self.postMessage({ id: "load", progress: p.progress || 0 }); } });
    const f = await ex; const out = [];
    for (let i = 0; i < texts.length; i += 8) {
      const r = await f(texts.slice(i, i + 8), { pooling: "mean", normalize: true });
      const w = r.dims[r.dims.length - 1];
      for (let j = 0; j < Math.min(8, texts.length - i); j++) out.push(Array.from(r.data.slice(j * w, (j + 1) * w)));
      self.postMessage({ id, progress: Math.min(i + 8, texts.length), total: texts.length });
    }
    self.postMessage({ id, done: true, vecs: out });
  } catch (err) { self.postMessage({ id, error: String(err) }); }
};`;
    worker = new Worker(URL.createObjectURL(new Blob([src], { type: 'application/javascript' })), { type: 'module' });
    worker.onmessage = e => {
      const d = e.data;
      if (d.id === 'load') { sm.textContent = `downloading model… ${Math.round(d.progress)}%`; return; }
      const p = pending.get(d.id); if (!p) return;
      if (d.error) { p.reject(new Error(d.error)); pending.delete(d.id); return; }
      if (d.done) { p.resolve(d.vecs); pending.delete(d.id); return; }
      sm.textContent = `indexing… ${d.progress}/${d.total}`;
    };
    return worker;
  }
  const embed = texts => new Promise((resolve, reject) => { const id = ++msg; pending.set(id, { resolve, reject }); getWorker().postMessage({ id, model: MODEL, texts }); });
  async function vectors() {
    const fp = IDX.length + ':' + IDX.reduce((a, d) => a + d.x.length, 0);
    try { const c = JSON.parse(localStorage.getItem(KEY) || 'null'); if (c && c.fp === fp) return c.vecs; } catch (e) {}
    sm.textContent = 'loading model… (one time, ~23MB)';
    const vecs = await embed(IDX.map(d => (d.t + '. ' + d.x).slice(0, 800)));
    try { localStorage.setItem(KEY, JSON.stringify({ fp, vecs })); } catch (e) {}
    return vecs;
  }
  let timer = null;
  async function meaningSearch() {
    const q = qm.value.trim();
    if (q.length < 3) { sm.textContent = ''; rm.innerHTML = ''; return; }
    const vecs = await vectors();
    sm.textContent = 'searching…';
    const [qv] = await embed([q]);
    const scored = IDX.map((d, i) => ({ d, s: vecs[i].reduce((a, v, j) => a + v * qv[j], 0) }))
      .filter(x => inScope(x.d) && x.s > 0.25).sort((a, b) => b.s - a.s).slice(0, 6);
    sm.textContent = scored.length ? scored.length + (scored.length === 1 ? ' result' : ' results') : 'no results';
    rm.innerHTML = scored.map(x => hit(x.d, x.d.x.slice(0, 140) + '…')).join('');
  }
  qm.addEventListener('input', () => {
    clearTimeout(timer); sm.textContent = qm.value.trim().length >= 3 ? '…' : '';
    timer = setTimeout(() => meaningSearch().catch(() => { sm.textContent = 'search unavailable'; }), 600);
  });
})();
