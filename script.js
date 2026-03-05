/**
 * QuantScope AI — Dashboard Frontend
 * script.js — Main Logic: API calls, charts, navigation, UI
 * Updated: February 24, 2026
 */

'use strict';

// ══════════════════════════════════════════════════════════════════
// APP STATE
// ══════════════════════════════════════════════════════════════════
const App = {
  ticker: 'PLTR',
  charts: {},
  histState: { period: '1y', interval: '1d', data: null, page: 1, pageSize: 20, sortCol: 0, sortAsc: false },
  forecastState: { period: '6m', model: 'auto', loading: false },
  currentUser: null,
};

const API = {
  status:  '/api/status/',
  ticker:  '/api/ticker-info/',
  quote:   '/api/quote/',
  history: '/api/history/',
  metrics: '/api/metrics/',
  features:'/api/features/',
  options: '/api/options/',
  forecast:'/api/forecast/',
  predict: '/api/predict/',
  ml:      '/api/ml-models/',
  dl:      '/api/deep-learning/',
  me:      '/api/auth/me',
  logout:  '/api/auth/logout',
};

// Shared Chart.js defaults — dark theme
const CD = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: { labels: { color: '#8b949e', font: { size: 11 }, boxWidth: 12 } },
    tooltip: {
      backgroundColor: '#161b22', borderColor: '#2d3748', borderWidth: 1,
      titleColor: '#e6edf3', bodyColor: '#8b949e',
    },
  },
  scales: {
    x: { ticks: { color: '#8b949e', maxTicksLimit: 8, font: { size: 10 } }, grid: { color: 'rgba(45,55,72,.5)' } },
    y: { ticks: { color: '#8b949e', font: { size: 10 } },                  grid: { color: 'rgba(45,55,72,.5)' } },
  },
};

// ══════════════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════════════
const fmt    = (v, d=2) => (v == null || isNaN(v)) ? '--' : Number(v).toFixed(d);
const fmtPct = (v)      => (v == null || isNaN(v)) ? '--' : (Number(v) >= 0 ? '+' : '') + Number(v).toFixed(2) + '%';
const fmtK   = (v) => {
  if (v == null || isNaN(v)) return '--';
  const a = Math.abs(v);
  if (a >= 1e12) return (v/1e12).toFixed(2) + 'T';
  if (a >= 1e9)  return (v/1e9).toFixed(2)  + 'B';
  if (a >= 1e6)  return (v/1e6).toFixed(2)  + 'M';
  if (a >= 1e3)  return (v/1e3).toFixed(1)  + 'K';
  return v.toLocaleString();
};

const el      = (id)       => document.getElementById(id);
const setText = (id, val)  => { const e = el(id); if (e) e.textContent = (val ?? '--'); };
const setHtml = (id, val)  => { const e = el(id); if (e) e.innerHTML  = (val ?? ''); };

function colorVal(id, val, inverse = false) {
  const e = el(id); if (!e) return;
  const n = parseFloat(val); e.classList.remove('text-green','text-red');
  if (isNaN(n)) return;
  if (inverse ? n < 0 : n > 0) e.classList.add('text-green');
  else if (n !== 0)              e.classList.add('text-red');
}

function badge(text, type = 'gray') { return `<span class="badge badge-${type}">${text}</span>`; }

function destroyChart(key) { if (App.charts[key]) { App.charts[key].destroy(); App.charts[key] = null; } }

async function apiFetch(url) {
  const res = await fetch(url);
  if (res.status === 401) { window.location.href = '/login'; throw new Error('Unauthorized'); }
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || 'API error');
  return data;
}

// ══════════════════════════════════════════════════════════════════
// NAVIGATION
// ══════════════════════════════════════════════════════════════════
function initNav() {
  document.querySelectorAll('.nav-item[data-target]').forEach(item => {
    item.addEventListener('click', () => {
      document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
      document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
      item.classList.add('active');
      const sec = el(item.dataset.target);
      if (sec) sec.classList.add('active');
      loadSectionData(item.dataset.target);
    });
  });

  // User dropdown
  const profile  = el('userProfileBtn');
  const dropdown = el('userDropdown');
  if (profile && dropdown) {
    profile.addEventListener('click', e => { e.stopPropagation(); dropdown.classList.toggle('open'); });
    document.addEventListener('click', () => dropdown.classList.remove('open'));
  }

  // Sign-out
  const signOut = el('btnSignOut');
  if (signOut) {
    signOut.addEventListener('click', async () => {
      try { await fetch(API.logout, { method: 'POST' }); } catch (_) {}
      window.location.href = '/login';
    });
  }
}

function loadSectionData(id) {
  const t = App.ticker;
  const map = {
    'sec-dashboard':   () => loadDashboard(t),
    'sec-live-quote':  () => loadQuote(t),
    'sec-history':     () => loadHistory(t, false),
    'sec-tech-ind':    () => loadFeatures(t),
    'sec-perf-metrics':() => loadMetrics(t),
    'sec-ai-predict':  () => loadAI(t),
    'sec-options':     () => loadOptions(t),
    'sec-system':      () => loadStatus(),
  };
  if (map[id]) map[id]();
}

// ══════════════════════════════════════════════════════════════════
// USER INFO
// ══════════════════════════════════════════════════════════════════
async function loadUser() {
  try {
    const data = await apiFetch(API.me);
    if (data.success) {
      const u = data.user; App.currentUser = u;
      setText('userName', u.firstName);
      setText('userAvatar', (u.firstName[0] + u.lastName[0]).toUpperCase());
    }
  } catch (_) { setText('userName', 'Guest'); setText('userAvatar', 'GU'); }
}

// ══════════════════════════════════════════════════════════════════
// SYSTEM STATUS
// ══════════════════════════════════════════════════════════════════
async function loadStatus() {
  try {
    const data = await apiFetch(API.status);
    const dot = el('systemStatusDot');
    if (dot) dot.classList.add(data.status === 'online' ? 'online' : 'offline');

    setText('sys-version', data.version || '--');
    const auth = data.authentication || {};
    setHtml('sys-auth-enabled',  auth.enabled   ? badge('Enabled', 'green') : badge('Disabled', 'gray'));
    setHtml('sys-session-active', auth.logged_in ? badge('Active',  'green') : badge('No Session','gray'));

    const fe = data.frontend || {};
    setText('sys-index', `index.html: ${fe.index_html ? '✅' : '❌ Missing'}`);
    setText('sys-css',   `style.css:  ${fe.style_css  ? '✅' : '❌ Missing'}`);
    setText('sys-js',    `script.js:  ${fe.script_js  ? '✅' : '❌ Missing'}`);

    const mods = data.modules || {};
    const list = el('module-status-list');
    if (list) list.innerHTML = Object.entries(mods).map(([k,v]) =>
      `<div class="module-row"><span>${k}</span>${v ? badge('✅ Online','green') : badge('❌ Unavailable','red')}</div>`
    ).join('');
  } catch (e) { console.error('Status error:', e); }
}

// ══════════════════════════════════════════════════════════════════
// DASHBOARD
// ══════════════════════════════════════════════════════════════════
async function loadDashboard(ticker) {
  setText('dash-title', `Dashboard — ${ticker}`);
  await Promise.allSettled([
    loadQuoteSummary(ticker),
    loadHistory(ticker, true),
    loadFeaturesSummary(ticker),
    loadMetricsSummary(ticker),
  ]);
}

async function loadQuoteSummary(ticker) {
  try {
    const { data: d } = await apiFetch(`${API.quote}?ticker=${ticker}`);
    setText('kpi-price', `${d.currency || ''} ${fmt(d.currentPrice)}`);

    const changeEl = el('kpi-change');
    if (changeEl) {
      changeEl.textContent = `${d.change >= 0 ? '+' : ''}${fmt(d.change)} (${fmtPct(d.changePercent)})`;
      changeEl.className   = 'val-mono ' + (d.change >= 0 ? 'text-green' : 'text-red');
    }
    const iconEl = el('kpi-price-icon');
    if (iconEl) iconEl.className = d.change >= 0 ? 'fa-solid fa-arrow-trend-up' : 'fa-solid fa-arrow-trend-down';

    setText('kpi-sentiment', fmt(d.sentiment, 1));
    const s = d.sentiment;
    setHtml('kpi-sentiment-badge', s > 6 ? badge('Bullish','green') : s < 4 ? badge('Bearish','red') : badge('Neutral','gray'));
    setText('kpi-predicted', `AI: ${d.currency || ''} ${fmt(d.predictedPrice)}`);
    setText('kpi-volume', fmtK(d.volume));
    setText('kpi-hilo', `H: ${fmt(d.dayHigh)}  L: ${fmt(d.dayLow)}`);

    const hi = d.high52Week, lo = d.low52Week, pct = hi !== lo ? ((d.currentPrice - lo) / (hi - lo)) * 100 : 50;
    const dot = el('range-dot');
    if (dot) dot.style.left = Math.max(0, Math.min(100, pct)) + '%';
    setText('range-low', fmt(lo)); setText('range-high', fmt(hi));
  } catch (e) { console.error('Quote summary:', e); }
}

async function loadMetricsSummary(ticker) {
  try {
    const { data } = await apiFetch(`${API.metrics}?ticker=${ticker}`);
    const s = data.statistics || {};
    setText('qs-sharpe',   fmt(s.sharpe, 3));
    setText('qs-sortino',  fmt(s.sortino, 3));
    const dd = el('qs-drawdown');
    if (dd) { dd.textContent = fmt((s.max_drawdown || 0) * 100, 2) + '%'; dd.className = 'val-mono text-red'; }
    setText('qs-winrate', fmt((s.win_rate || 0) * 100, 1) + '%');
    setText('qs-vol',  fmt((s.volatility || 0) * 100, 2) + '%');
    setText('qs-cagr', fmtPct((s.cagr || 0) * 100));
  } catch (_) {}
}

async function loadFeaturesSummary(ticker) {
  try {
    const { data } = await apiFetch(`${API.features}?ticker=${ticker}`);
    const l = data.latest;
    setText('m-rsi', fmt(l.rsi_14, 1));   setText('m-macd', fmt(l.macd, 4));
    setText('m-adx', fmt(l.adx, 1));      setText('m-atr',  fmt(l.atr_14, 2));
    setText('m-obv', fmtK(l.obv));        setText('m-mfi',  fmt(l.mfi, 1));
    renderSignals(data.signals, 'signals-container');
  } catch (_) {}
}

// ══════════════════════════════════════════════════════════════════
// LIVE QUOTE
// ══════════════════════════════════════════════════════════════════
async function loadQuote(ticker) {
  try {
    const [qr, ir] = await Promise.allSettled([
      apiFetch(`${API.quote}?ticker=${ticker}`),
      apiFetch(`${API.ticker}?ticker=${ticker}`),
    ]);

    if (qr.status === 'fulfilled') {
      const d = qr.value.data;
      const sign = d.change >= 0 ? '+' : '';
      setText('lq-price', `${d.currency} ${fmt(d.currentPrice)}`);
      const lqc = el('lq-change');
      if (lqc) { lqc.textContent = `${sign}${fmt(d.change)} (${fmtPct(d.changePercent)})`; lqc.className = 'val-mono ' + (d.change >= 0 ? 'text-green' : 'text-red'); }
      setHtml('lq-currency-badge', badge(d.currency, 'blue'));
      setText('lq-prev', `${d.currency} ${fmt(d.previousClose)}`);
      setText('lq-high', `${d.currency} ${fmt(d.dayHigh)}`);
      setText('lq-low',  `${d.currency} ${fmt(d.dayLow)}`);
      setText('lq-predicted', `${d.currency} ${fmt(d.predictedPrice)}`);
      const pct = d.high52Week !== d.low52Week ? ((d.currentPrice - d.low52Week) / (d.high52Week - d.low52Week)) * 100 : 50;
      const rdot = el('lq-range-dot'); if (rdot) rdot.style.left = Math.max(0, Math.min(100, pct)) + '%';
      setText('lq-52low', fmt(d.low52Week)); setText('lq-52high', fmt(d.high52Week));
      const vb = el('lq-vol-bar'); if (vb) vb.style.width = Math.min(100, (d.volume / 1e7) * 100) + '%';
    }

    if (ir.status === 'fulfilled') {
      const info = ir.value.data;
      setText('lq-company-name', info.name || ticker);
      setHtml('lq-badges', [
        info.sector   ? badge(info.sector,   'blue') : '',
        info.industry ? badge(info.industry, 'gray') : '',
        info.country  ? badge(info.country,  'cyan') : '',
      ].join(' '));
      setText('lq-mktcap', info.market_cap ? fmtK(info.market_cap) : '--');
      setText('lq-pe',   info.pe_ratio ? fmt(info.pe_ratio, 1) : '--');
      setText('lq-beta', info.beta     ? fmt(info.beta, 2) : '--');
      setText('lq-target', info.target_price ? `$${fmt(info.target_price)}` : '--');
      if (info.recommendation) {
        const t = info.recommendation.toLowerCase();
        setHtml('lq-rec-badge', badge(info.recommendation, t.includes('buy') ? 'green' : t.includes('sell') ? 'red' : 'gray'));
      }
      const ws = el('lq-website'); if (ws && info.website) ws.href = info.website;
      renderNews(info.news || []);
    }
  } catch (e) { console.error('Quote error:', e); }
}

function renderNews(items) {
  const c = el('news-container'); if (!c) return;
  c.innerHTML = items.length
    ? items.map(n => `<div class="news-item"><div class="news-title">${n.title||'--'}</div><div class="news-meta">${n.publisher||''} · ${n.published||''}</div></div>`).join('')
    : '<div class="text-muted">No news available</div>';
}

// ══════════════════════════════════════════════════════════════════
// PRICE HISTORY
// ══════════════════════════════════════════════════════════════════
async function loadHistory(ticker, dashOnly = false) {
  try {
    const { data: d } = await apiFetch(`${API.history}?ticker=${ticker}&period=${App.histState.period}&interval=${App.histState.interval}`);
    App.histState.data = d;
    renderOverviewChart(d);
    if (!dashOnly) { renderHistChart(d); renderHistTablePage(); }
  } catch (e) { console.error('History error:', e); }
}

function makeGradient(ctx, color = '#58a6ff') {
  const g = ctx.createLinearGradient(0, 0, 0, 400);
  g.addColorStop(0, color.replace(')', ',.18)').replace('rgb','rgba'));
  g.addColorStop(1, color.replace(')', ',0)').replace('rgb','rgba'));
  return g;
}

function renderOverviewChart(d) {
  destroyChart('overview');
  const canvas = el('overviewChart'); if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createLinearGradient(0,0,0,400);
  grad.addColorStop(0, 'rgba(88,166,255,.18)'); grad.addColorStop(1, 'rgba(88,166,255,0)');
  App.charts.overview = new Chart(canvas, {
    type: 'line',
    data: { labels: d.dates, datasets: [{ label: 'Close', data: d.close, borderColor: '#58a6ff', backgroundColor: grad, borderWidth: 1.5, pointRadius: 0, fill: true, tension: .3 }] },
    options: { ...CD },
  });
}

function renderHistChart(d) {
  destroyChart('hist');
  const canvas = el('histChart'); if (!canvas) return;
  App.charts.hist = new Chart(canvas, {
    type: 'line',
    data: { labels: d.dates, datasets: [{ label: 'Close', data: d.close, borderColor: '#58a6ff', borderWidth: 1.5, pointRadius: 0, tension: .3 }] },
    options: { ...CD },
  });

  destroyChart('histVol');
  const vCanvas = el('histVolumeChart');
  if (vCanvas) App.charts.histVol = new Chart(vCanvas, {
    type: 'bar',
    data: { labels: d.dates, datasets: [{ label: 'Volume', data: d.volume, backgroundColor: 'rgba(88,166,255,.3)', borderWidth: 0 }] },
    options: { ...CD, plugins: { ...CD.plugins, legend: { display: false } } },
  });

  const last = d.close.length - 1;
  if (last >= 0) {
    setText('hist-open',  fmt(d.open[last]));  setText('hist-high', fmt(d.high[last]));
    setText('hist-low',   fmt(d.low[last]));   setText('hist-close', fmt(d.close[last]));
    setText('hist-vol', fmtK(d.volume[last]));
  }
}

function renderHistTablePage() {
  const state = App.histState; const d = state.data; if (!d) return;
  const rows = d.dates.map((dt, i) => {
    const chg = i > 0 ? ((d.close[i] - d.close[i-1]) / d.close[i-1]) * 100 : 0;
    return { date: dt, open: d.open[i], high: d.high[i], low: d.low[i], close: d.close[i], volume: d.volume[i], change: chg };
  }).reverse();
  const total = rows.length;
  const start = (state.page - 1) * state.pageSize;
  const slice = rows.slice(start, start + state.pageSize);
  const tbody = el('hist-table-body'); if (!tbody) return;
  tbody.innerHTML = slice.map(r => `
    <tr>
      <td>${r.date}</td><td>${fmt(r.open)}</td><td>${fmt(r.high)}</td>
      <td>${fmt(r.low)}</td><td><strong>${fmt(r.close)}</strong></td>
      <td>${fmtK(r.volume)}</td>
      <td class="${r.change >= 0 ? 'text-green' : 'text-red'}">${fmtPct(r.change)}</td>
    </tr>`).join('');
  setText('hist-table-info', `Showing ${start+1}–${Math.min(start+state.pageSize,total)} of ${total} rows`);
}

window.sortTable = col => {
  const s = App.histState;
  s.sortAsc = s.sortCol === col ? !s.sortAsc : true;
  s.sortCol = col; renderHistTablePage();
};

function initHistControls() {
  [['hist-period-group','period'],['hist-interval-group','interval'],['overview-period-group','period']].forEach(([gId, key]) => {
    const g = el(gId); if (!g) return;
    g.querySelectorAll('.pill-btn').forEach(btn => btn.addEventListener('click', () => {
      g.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active')); btn.classList.add('active');
      if (gId === 'overview-period-group') {
        apiFetch(`${API.history}?ticker=${App.ticker}&period=${btn.dataset.period}`).then(r => renderOverviewChart(r.data)).catch(()=>{});
      } else { App.histState[key] = btn.dataset[key]; loadHistory(App.ticker, false); }
    }));
  });
  const pp = el('hist-prev-page'); if (pp) pp.addEventListener('click', () => { if (App.histState.page > 1) { App.histState.page--; renderHistTablePage(); } });
  const np = el('hist-next-page'); if (np) np.addEventListener('click', () => { App.histState.page++; renderHistTablePage(); });
}

// ══════════════════════════════════════════════════════════════════
// TECHNICAL INDICATORS
// ══════════════════════════════════════════════════════════════════
async function loadFeatures(ticker) {
  try {
    const { data } = await apiFetch(`${API.features}?ticker=${ticker}`);
    const { latest: l, signals, history: h } = data;

    // Gauges
    const rsi = l.rsi_14;
    const rsiEl = el('gauge-rsi-circle');
    if (rsiEl) { rsiEl.textContent = fmt(rsi,1); rsiEl.style.borderTopColor = rsi>70?'#f85149':rsi<30?'#3fb950':'#58a6ff'; }
    setHtml('gauge-rsi-label', rsi>70?badge('Overbought','red'):rsi<30?badge('Oversold','green'):badge('Neutral','gray'));

    const macdEl = el('gauge-macd-circle');
    if (macdEl) { macdEl.textContent = fmt(l.macd,4); macdEl.style.borderTopColor = l.macd>0?'#3fb950':'#f85149'; }

    const mfi = l.mfi; const mfiEl = el('gauge-mfi-circle');
    if (mfiEl) mfiEl.textContent = fmt(mfi,1);
    setHtml('gauge-mfi-label', mfi>80?badge('Overbought','red'):mfi<20?badge('Oversold','green'):badge('Neutral','gray'));

    const adx = l.adx; const adxEl = el('gauge-adx-circle');
    if (adxEl) { adxEl.textContent = fmt(adx,1); adxEl.style.borderTopColor = adx>25?'#58a6ff':'#8b949e'; }
    setHtml('gauge-adx-label', adx>25?badge('Trending','blue'):badge('Ranging','gray'));

    const atrEl = el('gauge-atr-circle'); if (atrEl) atrEl.textContent = fmt(l.atr_14,2);

    renderSignals(signals, 'tech-signals-container');
    renderTechCharts(h);
  } catch (e) { console.error('Features error:', e); }
}

function renderTechCharts(h) {
  const dates = h.dates||[], close = h.close||[];
  destroyChart('techPrice');
  const pc = el('techPriceChart');
  if (pc) App.charts.techPrice = new Chart(pc, { type:'line', data:{ labels:dates, datasets:[
    {label:'Close',  data:close,    borderColor:'#58a6ff',              borderWidth:2, pointRadius:0, tension:.3},
    {label:'BB Up',  data:h.bb_upper, borderColor:'rgba(188,140,255,.5)', borderWidth:1, pointRadius:0, borderDash:[4,4], tension:.3},
    {label:'BB Mid', data:h.bb_middle,borderColor:'rgba(188,140,255,.8)', borderWidth:1, pointRadius:0, tension:.3},
    {label:'BB Low', data:h.bb_lower, borderColor:'rgba(188,140,255,.5)', borderWidth:1, pointRadius:0, borderDash:[4,4], tension:.3},
    {label:'SMA20',  data:h.sma_20,  borderColor:'rgba(57,208,216,.7)',   borderWidth:1.5,pointRadius:0,tension:.3},
    {label:'SMA50',  data:h.sma_50,  borderColor:'rgba(217,153,34,.7)',   borderWidth:1.5,pointRadius:0,tension:.3},
  ]}, options:{...CD}});

  destroyChart('techRsi');
  const rc = el('techRsiChart');
  if (rc) App.charts.techRsi = new Chart(rc, { type:'line', data:{ labels:dates, datasets:[
    {label:'RSI 14',       data:h.rsi_14,                     borderColor:'#58a6ff',              borderWidth:1.5,pointRadius:0,tension:.3},
    {label:'Overbought 70',data:Array(dates.length).fill(70), borderColor:'rgba(248,81,73,.4)',   borderWidth:1,  pointRadius:0,borderDash:[4,4]},
    {label:'Oversold 30',  data:Array(dates.length).fill(30), borderColor:'rgba(63,185,80,.4)',   borderWidth:1,  pointRadius:0,borderDash:[4,4]},
  ]}, options:{...CD, scales:{...CD.scales, y:{...CD.scales.y, min:0, max:100}}}});

  destroyChart('techMacd');
  const mc = el('techMacdChart');
  if (mc) App.charts.techMacd = new Chart(mc, { type:'line', data:{ labels:dates, datasets:[
    {label:'MACD',   data:h.macd,        borderColor:'#58a6ff', borderWidth:1.5, pointRadius:0, tension:.3},
    {label:'Signal', data:h.macd_signal, borderColor:'#f85149', borderWidth:1.5, pointRadius:0, tension:.3},
  ]}, options:{...CD}});
}

function renderSignals(signals, containerId) {
  const c = el(containerId); if (!c) return;
  if (!signals?.length) { c.innerHTML = '<div class="text-muted">No signals available</div>'; return; }
  c.innerHTML = signals.map(s => {
    const dir = (s.direction||s.signal||'neutral').toLowerCase();
    const cls = dir.includes('buy') ? 'signal-buy' : dir.includes('sell') ? 'signal-sell' : 'signal-neutral';
    const icon = dir.includes('buy') ? '📈' : dir.includes('sell') ? '📉' : '⚖️';
    return `<div class="signal-item"><span class="signal-icon">${icon}</span><span class="signal-text">${s.description||s.name||s.indicator||''}</span><span class="signal-badge ${cls}">${s.direction||s.signal||'Neutral'}</span></div>`;
  }).join('');
}

// ══════════════════════════════════════════════════════════════════
// PERFORMANCE METRICS
// ══════════════════════════════════════════════════════════════════
async function loadMetrics(ticker) {
  try {
    const { data } = await apiFetch(`${API.metrics}?ticker=${ticker}`);
    const { statistics: s, tradeStatistics: t, distribution: dist } = data;

    setText('pm-total-return', fmtPct((s.total_return||0)*100));   colorVal('pm-total-return', s.total_return);
    setText('pm-ann-return',   fmtPct((s.annualized_return||0)*100)); colorVal('pm-ann-return', s.annualized_return);
    setText('pm-cagr',         fmtPct((s.cagr||0)*100));             colorVal('pm-cagr', s.cagr);
    setText('pm-vol',      fmt((s.volatility||0)*100,2)+'%');
    setText('pm-down-vol', fmt((s.downside_volatility||0)*100,2)+'%');
    setText('pm-max-dd',   fmt((s.max_drawdown||0)*100,2)+'%');
    setText('pm-avg-dd',   fmt((s.avg_drawdown||0)*100,2)+'%');
    setText('pm-sharpe',  fmt(s.sharpe,3));  setText('pm-sortino', fmt(s.sortino,3));
    setText('pm-calmar',  fmt(s.calmar,3));  setText('pm-ir',      fmt(s.information_ratio,3));
    setText('pm-alpha', fmtPct((s.alpha||0)*100)); setText('pm-beta', fmt(s.beta,3)); setText('pm-te', fmt(s.tracking_error,4));
    setHtml('pm-bench-badge', badge(s.benchmark||'SPY','gray'));

    if (dist) {
      setText('pm-var',  fmtPct((dist.var_95||0)*100));
      setText('pm-cvar', fmtPct((dist.cvar_95||0)*100));
      renderDistChart(dist);
    }
    if (t) {
      const wr = t.win_rate||0;
      setText('pm-wr-center', fmt(wr*100,1)+'%');
      setText('pm-pf', fmt(t.profit_factor,2)); setText('pm-obs', t.total_trades||'--'); setText('pm-years', fmt(t.years,1)+'y');
      renderWinRateDonut(wr);
    }
  } catch (e) { console.error('Metrics error:', e); }
}

function renderDistChart(dist) {
  destroyChart('dist');
  const ctx = el('distChart'); if (!ctx || !dist.bins) return;
  App.charts.dist = new Chart(ctx, { type:'bar', data:{ labels: dist.bins.map(b=>fmt(b*100,1)+'%'), datasets:[{
    label:'Frequency', data: dist.counts || dist.frequencies,
    backgroundColor: dist.bins.map(b => b<0?'rgba(248,81,73,.5)':'rgba(63,185,80,.5)'), borderWidth:0,
  }]}, options:{...CD, plugins:{...CD.plugins, legend:{display:false}}}});
}

function renderWinRateDonut(wr) {
  destroyChart('winRate');
  const ctx = el('winRateDonut'); if (!ctx) return;
  App.charts.winRate = new Chart(ctx, { type:'doughnut', data:{ datasets:[{
    data:[wr*100,(1-wr)*100], backgroundColor:['rgba(63,185,80,.8)','rgba(248,81,73,.3)'], borderWidth:0, hoverOffset:4,
  }]}, options:{ responsive:true, maintainAspectRatio:false, cutout:'72%', plugins:{ legend:{display:false}, tooltip:{...CD.plugins.tooltip} }}});
}

// ══════════════════════════════════════════════════════════════════
// FORECAST
// ══════════════════════════════════════════════════════════════════
function initForecast() {
  ['fc-period-group','fc-model-group'].forEach(gId => {
    const g = el(gId); if (!g) return;
    const key = gId === 'fc-period-group' ? 'period' : 'model';
    g.querySelectorAll('.pill-btn').forEach(btn => btn.addEventListener('click', () => {
      g.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active')); btn.classList.add('active');
      App.forecastState[key] = btn.dataset[key];
    }));
  });
  const gen = el('btnGenerateForecast');
  if (gen) gen.addEventListener('click', () => { if (!App.forecastState.loading) loadForecast(App.ticker); });
}

async function loadForecast(ticker) {
  App.forecastState.loading = true;
  const loading = el('forecast-loading'), content = el('forecast-content');
  if (loading) loading.classList.add('show');
  if (content) content.style.display = 'none';

  try {
    const { period, model } = App.forecastState;
    try {
      const { data: d } = await apiFetch(`${API.predict}?ticker=${ticker}&period=${period}&model=${model}`);
      setText('fc-model-name', d.model.model||'--'); setText('fc-rmse', d.model.rmse?fmt(d.model.rmse):'--'); setText('fc-period-label', period.toUpperCase());
      renderForecastChart(d); renderForecastTable(d);
    } catch (_) {
      // fallback to simple ARIMA endpoint
      const { data: d } = await apiFetch(`${API.forecast}?ticker=${ticker}`);
      setText('fc-model-name','ARIMA'); setText('fc-rmse','--'); setText('fc-period-label','10 Days');
      renderForecastChartSimple(d);
    }
  } catch (e) { console.error('Forecast error:', e); }
  finally {
    App.forecastState.loading = false;
    if (loading) loading.classList.remove('show');
    if (content) content.style.display = 'block';
  }
}

function renderForecastChart(d) {
  destroyChart('forecast');
  const ctx = el('forecastChart'); if (!ctx) return;
  const datasets = [{ label:'Forecast', data:d.forecast.prices, borderColor:'#3fb950', borderWidth:2, pointRadius:2, tension:.3 }];
  if (d.forecast.confidence_lower) {
    datasets.push({label:'Upper CI',data:d.forecast.confidence_upper,borderColor:'rgba(63,185,80,.3)',borderWidth:1,pointRadius:0,borderDash:[4,4]});
    datasets.push({label:'Lower CI',data:d.forecast.confidence_lower,borderColor:'rgba(63,185,80,.3)',borderWidth:1,pointRadius:0,borderDash:[4,4]});
  }
  App.charts.forecast = new Chart(ctx, { type:'line', data:{labels:d.forecast.dates, datasets}, options:{...CD}});
  renderForecastTable(d);
}

function renderForecastChartSimple(d) {
  destroyChart('forecast');
  const ctx = el('forecastChart'); if (!ctx) return;
  const hd=d.historical?.dates||[], hp=d.historical?.prices||[], fd=d.arima?.dates||[], fp=d.arima?.prices||[];
  App.charts.forecast = new Chart(ctx, { type:'line', data:{ labels:[...hd,...fd], datasets:[
    {label:'Historical', data:[...hp,...Array(fd.length).fill(null)], borderColor:'#58a6ff',borderWidth:2,pointRadius:0,tension:.3},
    {label:'ARIMA',      data:[...Array(hd.length).fill(null),...fp], borderColor:'#3fb950',borderWidth:2,pointRadius:3,tension:.3},
  ]}, options:{...CD}});
}

function renderForecastTable(d) {
  const tbody = el('forecast-table-body'); if (!tbody) return;
  const dates = d.forecast?.dates||[], prices = d.forecast?.prices||[], base = d.current?.price||prices[0];
  tbody.innerHTML = dates.map((dt,i) => {
    const p=prices[i], chg=p-base, pct=(chg/base)*100;
    return `<tr><td>${dt}</td><td><strong>${fmt(p)}</strong></td><td class="${chg>=0?'text-green':'text-red'}">${fmt(chg,2)}</td><td class="${pct>=0?'text-green':'text-red'}">${fmtPct(pct)}</td></tr>`;
  }).join('');
}

// ══════════════════════════════════════════════════════════════════
// AI PREDICTIONS
// ══════════════════════════════════════════════════════════════════
async function loadAI(ticker) {
  try {
    const { data: d } = await apiFetch(`${API.ml}?ticker=${ticker}`);
    const p = d.predictions, ma = d.market_analysis;
    setText('aip-svr',  p.svr           ? `$${fmt(p.svr)}`           : '--');
    setText('aip-rf',   p.random_forest ? `$${fmt(p.random_forest)}` : '--');
    setText('aip-xgb',  p.xgboost       ? `$${fmt(p.xgboost)}`       : '--');
    setText('aip-lgbm', p.lightgbm      ? `$${fmt(p.lightgbm)}`      : '--');
    setText('aip-beta',  fmt(ma.beta,4));
    setText('aip-alpha', fmt(ma.alpha,6));
    setText('aip-r2',    fmt(ma.r_squared,4));
  } catch (e) { console.error('AI error:', e); }
}

// ══════════════════════════════════════════════════════════════════
// OPTIONS
// ══════════════════════════════════════════════════════════════════
async function loadOptions(ticker) {
  try {
    const { data: d } = await apiFetch(`${API.options}?ticker=${ticker}`);
    const inp=d.inputs, bs=d.blackscholes, mc=d.montecarlo, binom=d.binomial;

    setText('op-spot',   `$${fmt(inp.spot_price)}`);
    setText('op-strike', `$${fmt(inp.strike_price)}`);
    setText('op-vol',    `${fmt(inp.volatility)}%`);
    setText('op-T',      `${fmt(inp.time_to_expiry_years,4)}y`);
    setText('op-r',      `${fmt(inp.risk_free_rate)}%`);
    setText('op-q',      `${fmt(inp.dividend_yield)}%`);

    // CALL
    setText('op-call-bs',    `$${fmt(bs.call.price)}`);
    setText('op-call-mc',    mc.call?.price != null ? `$${fmt(mc.call.price)}` : `$${fmt(mc.call_price)}`);
    setText('op-call-binom', `$${fmt(binom.european_call)}`);
    const cCI = mc.call?.confidence_interval_95 || mc.confidence_interval_95 || [];
    setText('op-call-ci-lo', cCI[0]!=null?`$${fmt(cCI[0])}`:'--');
    setText('op-call-ci-hi', cCI[1]!=null?`$${fmt(cCI[1])}`:'--');

    // PUT
    setText('op-put-bs',    bs.put?.price!=null?`$${fmt(bs.put.price)}`:'--');
    setText('op-put-mc',    mc.put?.price!=null?`$${fmt(mc.put.price)}`:'--');
    setText('op-put-binom', binom.european_put!=null?`$${fmt(binom.european_put)}`:'--');
    const pCI = mc.put?.confidence_interval_95||[];
    setText('op-put-ci-lo', pCI[0]!=null?`$${fmt(pCI[0])}`:'--');
    setText('op-put-ci-hi', pCI[1]!=null?`$${fmt(pCI[1])}`:'--');

    // Greeks
    setText('greek-call-delta', `Call: ${fmt(bs.call.delta,4)}`);
    setText('greek-put-delta',  bs.put ? `Put: ${fmt(bs.put.delta,4)}`  : '--');
    setText('greek-gamma', fmt(bs.call.gamma,6));
    setText('greek-vega',  fmt(bs.call.vega,4));
    setText('greek-call-theta', `Call: ${fmt(bs.call.theta,4)}`);
    setText('greek-put-theta',  bs.put ? `Put: ${fmt(bs.put.theta,4)}`  : '--');
    setText('greek-call-rho',   `Call: ${fmt(bs.call.rho,4)}`);
    setText('greek-put-rho',    bs.put ? `Put: ${fmt(bs.put.rho,4)}`    : '--');
    setText('greek-iv', inp.volatility ? `${fmt(inp.volatility)}%` : '--');
  } catch (e) { console.error('Options error:', e); }
}

// ══════════════════════════════════════════════════════════════════
// ANALYZE BUTTON
// ══════════════════════════════════════════════════════════════════
function initAnalyze() {
  const btn = el('btnAnalyze'), input = el('tickerSearch');
  if (!btn || !input) return;

  function analyze() {
    const t = input.value.toUpperCase().trim(); if (!t) return;
    App.ticker = t;
    const active = document.querySelector('.content-section.active');
    loadSectionData(active ? active.id : 'sec-dashboard');
  }
  btn.addEventListener('click', analyze);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') analyze(); });

  const rq = el('btnRefreshQuote');  if (rq) rq.addEventListener('click', () => loadQuote(App.ticker));
  const rs = el('btnRefreshStatus'); if (rs) rs.addEventListener('click', loadStatus);
}

// ══════════════════════════════════════════════════════════════════
// FEATURE CHIPS
// ══════════════════════════════════════════════════════════════════
function initFeatureChips() {
  document.querySelectorAll('.feature-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.feature-chip').forEach(c => c.classList.remove('active'));
      document.querySelectorAll('.feature-detail').forEach(d => d.classList.remove('show'));
      chip.classList.add('active');
      const feat = el(`feat-${chip.dataset.features}`); if (feat) feat.classList.add('show');
    });
  });
}

// ══════════════════════════════════════════════════════════════════
// BOOT
// ══════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  initNav();
  initAnalyze();
  initHistControls();
  initForecast();
  initFeatureChips();
  loadUser();
  loadStatus();
  loadDashboard(App.ticker);
});

// ══════════════════════════════════════════════════════════════════
// SAAS PREMIUM ENFORCEMENTS & THEME (Appended)
// ══════════════════════════════════════════════════════════════════

// 1. Theme Toggler
function initTheme() {
    const themeBtn = document.getElementById('themeToggleBtn');
    const themeIcon = document.getElementById('themeIcon');
    if (!themeBtn) return;

    // Check local storage or system preference
    const savedTheme = localStorage.getItem('qs-theme');
    if (savedTheme === 'light') {
        document.body.classList.add('light-theme');
        themeIcon.classList.replace('fa-moon', 'fa-sun');
    }

    themeBtn.addEventListener('click', () => {
        document.body.classList.toggle('light-theme');
        const isLight = document.body.classList.contains('light-theme');

        if (isLight) {
            themeIcon.classList.replace('fa-moon', 'fa-sun');
            localStorage.setItem('qs-theme', 'light');
        } else {
            themeIcon.classList.replace('fa-sun', 'fa-moon');
            localStorage.setItem('qs-theme', 'dark');
        }

        // Dispatch event to re-render charts if they depend on CSS variables
        window.dispatchEvent(new Event('resize'));
    });
}

// 2. Global Red/Green Number Enforcement
function enforceColorPolarity(root = document) {
    const POS_CLASS = "is-positive";
    const NEG_CLASS = "is-negative";

    const candidates = root.querySelectorAll([
      '[id*="change"]', '[id*="return"]', '[id*="drawdown"]', '[id*="pnl"]',
      '.val-mono', '.kpi-sub', '.badge', 'td', 'span'
    ].join(','));

    candidates.forEach(el => {
        const text = el.textContent.trim();
        // Look for numbers that represent deltas
        const cleaned = text.replace(/[,\s%()]/g, "").replace(/−/g, "-");
        const match = cleaned.match(/^[-+]?\d+(\.\d+)?$/);

        if (!match) return;
        const n = Number(match[0]);
        if (Number.isNaN(n)) return;

        // Verify it's meant to be a delta (has %, +, -, brackets)
        const looksLikeDelta = /[+\-−%()]/.test(text);
        if (!looksLikeDelta) return;

        el.classList.remove(POS_CLASS, NEG_CLASS);
        if (n > 0 || text.includes('+')) {
            el.classList.add(POS_CLASS);
        } else if (n < 0 || text.includes('-') || text.includes('−') || /^\(.*\)$/.test(text)) {
            el.classList.add(NEG_CLASS);
        }
    });
}

// 3. Setup MutationObserver to continuously enforce colors dynamically
function hookLiveUpdates() {
    const obs = new MutationObserver(muts => {
        let shouldRun = false;
        for (const m of muts) {
            if (m.type === 'characterData' || (m.type === 'childList' && (m.addedNodes.length || m.removedNodes.length))) {
                shouldRun = true;
                break;
            }
        }
        if (shouldRun) {
            // Debounce to prevent blocking main thread
            requestAnimationFrame(() => enforceColorPolarity());
        }
    });
    obs.observe(document.body, { subtree: true, childList: true, characterData: true });
}

// Boot enhancements
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    enforceColorPolarity();
    hookLiveUpdates();
});