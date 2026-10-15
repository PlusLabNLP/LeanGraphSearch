'use strict';

// Accessible tabs: arrow keys change panels; Home and End jump to either edge.
const tabs = [...document.querySelectorAll('[role="tab"]')];
function activateTab(tab) {
  for (const item of tabs) {
    const active = item === tab;
    item.setAttribute('aria-selected', String(active));
    item.tabIndex = active ? 0 : -1;
    document.getElementById(item.getAttribute('aria-controls')).hidden = !active;
  }
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateTab(tab));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    if (next === undefined) return;
    event.preventDefault();
    activateTab(tabs[next]);
    tabs[next].focus();
  });
});

// Paper Tables 3–5. Each row is [solved@8, solved@16, solved@32, relative cost].
// Order: no retrieval, LS standard, LGS standard, LS reasoning, LGS reasoning.
const results = {
  fate: {
    name: 'FATE-H', table: 3,
    off: [[35,42,47,1],[49,61,70,1.11],[53,69,71,1.09],[52,68,73,1.41],[56,70,75,1.37]],
    on: [[33,45,52,1],[53,63,71,1.01],[54,68,73,1.01],[54,66,73,1.31],[58,75,81,1.22]]
  },
  mpr: {
    name: 'MathlibMPR-Prop', table: 4,
    off: [[15,18,23,1],[31,35,39,.82],[32,38,41,.78],[33,37,41,1.01],[33,38,39,1.06]],
    on: [[15,19,23,1],[33,35,39,.76],[35,37,40,.78],[33,36,39,.99],[31,37,40,.96]]
  },
  basic: {
    name: 'Lean-IMO Basic', table: 5,
    off: [[12,16,16,1],[15,15,18,1.49],[16,17,21,1.40],[14,16,21,2.04],[14,16,19,2.09]],
    on: [[13,15,17,1],[13,16,16,1.61],[15,18,20,1.64],[16,18,18,1.94],[15,17,18,2.25]]
  },
  advanced: {
    name: 'Lean-IMO Advanced', table: 5,
    off: [[7,7,8,1],[4,8,9,1.82],[5,9,10,1.88],[7,9,10,2.56],[7,8,10,2.53]],
    on: [[5,6,7,1],[6,7,8,1.64],[7,9,9,1.61],[7,10,12,2.31],[7,9,12,2.20]]
  }
};
const methods = [
  ['No retrieval',''], ['LeanSearch v2','Standard'], ['LeanGraphSearch','Standard'],
  ['LeanSearch v2','Reasoning'], ['LeanGraphSearch','Reasoning']
];
const benchmark = document.getElementById('benchmark');
const memory = document.getElementById('memory');
function renderProving() {
  const setting = results[benchmark.value];
  const caption = `${setting.name} · Failure memory ${memory.value === 'on' ? 'enabled' : 'disabled'} · Paper Table ${setting.table}`;
  document.getElementById('proving-caption').textContent = caption;
  const body = document.getElementById('proving-body');
  body.replaceChildren();
  setting[memory.value].forEach((values, index) => {
    const row = document.createElement('tr');
    if (index === 2 || index === 4) row.className = 'ours';
    const label = document.createElement('th');
    label.scope = 'row';
    label.textContent = methods[index][0];
    if (methods[index][1]) {
      const mode = document.createElement('small');
      mode.textContent = methods[index][1];
      label.append(mode);
    }
    row.append(label);
    values.forEach((value, column) => {
      const cell = document.createElement('td');
      cell.textContent = column === 3 ? `${value.toFixed(2)}×` : value;
      row.append(cell);
    });
    body.append(row);
  });
  document.getElementById('proving-status').textContent = `${caption}. Results updated.`;
}
benchmark.addEventListener('change', renderProving);
memory.addEventListener('change', renderProving);

// Keep all result sections available if scripts are disabled.
