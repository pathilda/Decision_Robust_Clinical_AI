// Execute the actual exported interaction script against a minimal DOM/Plotly adapter.
// This verifies combined visibility state without requiring a browser or network.
const assert = require('node:assert/strict');
const vm = require('node:vm');
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => input += chunk);
process.stdin.on('end', () => {
  const {script, data} = JSON.parse(input);
  const handlers = {};
  const gd = {data, on: (name, fn) => handlers[name] = fn};
  const menu = {value: 'All', addEventListener: (name, fn) => handlers[name] = fn};
  const count = {textContent: ''};
  const originalValues = JSON.stringify(data.map(t => [t.x, t.y, t.customdata]));
  const Plotly = {restyle: (target, change) => {
    assert.equal(target, gd);
    data.forEach((trace, i) => Object.keys(change).forEach(key => trace[key] = change[key][i]));
  }};
  vm.runInNewContext(script, {Plotly, document: {
    querySelector: () => gd,
    getElementById: id => id === 'starting-parameter' ? menu : count,
  }});
  const profiles = ['All', 'balanced', 'unbalanced', 'highly_unbalanced'];
  const hiddenModel = data[0].meta.model;
  const check = hidden => {
    for (const profile of profiles) {
      menu.value = profile;
      handlers.change();
      let n = 0;
      for (const trace of data) {
        const expected = hidden && trace.meta.model === hiddenModel ? 'legendonly' :
          profile === 'All' || trace.meta.profile === profile;
        assert.equal(trace.visible, expected);
        if (expected === true) n += trace.meta.n;
      }
      assert.equal(count.textContent, 'Visible observations: ' + n);
      assert.equal(JSON.stringify(data.map(t => [t.x, t.y, t.customdata])), originalValues);
      const legendModels = data.filter(t => t.showlegend).map(t => t.meta.model);
      assert.equal(new Set(legendModels).size, legendModels.length);
    }
  };
  check(false);
  assert.equal(handlers.plotly_legendclick({curveNumber: 0}), false);
  check(true); // Hidden model stays hidden across every dropdown change and facet.
  assert.equal(handlers.plotly_legendclick({curveNumber: 0}), false);
  check(false);
  assert.equal(handlers.plotly_legenddoubleclick(), false);
  process.stdout.write('visibility, legend state, values and counts verified');
});
