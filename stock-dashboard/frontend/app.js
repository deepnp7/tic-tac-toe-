const listEl = document.getElementById('company-list');
const titleEl = document.getElementById('title');
const statHigh = document.getElementById('stat-high');
const statLow = document.getElementById('stat-low');
const statVol = document.getElementById('stat-vol');
const statPred = document.getElementById('stat-pred');

let chart;

async function fetchJSON(url){
  const res = await fetch(url);
  if(!res.ok){ throw new Error(`HTTP ${res.status}`); }
  return res.json();
}

async function loadCompanies(){
  const companies = await fetchJSON('/api/companies');
  listEl.innerHTML = '';
  companies.forEach(c => {
    const li = document.createElement('li');
    li.innerHTML = `<span>${c.symbol}</span><small style="color:#9ca3af">${c.name}</small>`;
    li.addEventListener('click', () => selectCompany(c.symbol, c.name, li));
    listEl.appendChild(li);
  });
  // Autoselect first
  if(companies.length){ selectCompany(companies[0].symbol, companies[0].name, listEl.firstChild); }
}

function setActiveItem(el){
  Array.from(listEl.children).forEach(i => i.classList.remove('active'));
  if(el) el.classList.add('active');
}

function makeChart(labels, closes){
  const ctx = document.getElementById('priceChart').getContext('2d');
  if(chart){ chart.destroy(); }
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Close',
        data: closes,
        borderColor: '#22d3ee',
        borderWidth: 2,
        fill: false,
        pointRadius: 0,
        tension: 0.2
      }]
    },
    options: {
      responsive: true,
      scales: {
        x: { ticks: { color: '#9ca3af' } },
        y: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.06)' } }
      },
      plugins: { legend: { labels: { color: '#e5e7eb' } } }
    }
  });
}

async function selectCompany(symbol, name, el){
  setActiveItem(el);
  titleEl.textContent = `${symbol} • ${name}`;
  try{
    const ohlc = await fetchJSON(`/api/ohlc?ticker=${encodeURIComponent(symbol)}`);
    const labels = ohlc.bars.map(b => new Date(b.timestamp).toLocaleDateString());
    const closes = ohlc.bars.map(b => b.close);
    makeChart(labels, closes);

    const summary = await fetchJSON(`/api/summary?ticker=${encodeURIComponent(symbol)}`);
    statHigh.textContent = summary.high_52w?.toLocaleString() ?? '-';
    statLow.textContent = summary.low_52w?.toLocaleString() ?? '-';
    statVol.textContent = summary.average_volume?.toLocaleString() ?? '-';

    const pred = await fetchJSON(`/api/predict?ticker=${encodeURIComponent(symbol)}`);
    statPred.textContent = pred.next_day_close_prediction?.toLocaleString() ?? '-';
  }catch(err){
    console.error(err);
    alert('Failed to load data. Try again later.');
  }
}

loadCompanies();