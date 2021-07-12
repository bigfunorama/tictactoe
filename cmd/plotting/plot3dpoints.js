d3.csv('http://localhost:8080/fetchdata', function(err, rows){
  function unpack(rows, key) {
      return rows.map(function(row){return row[key]; });}
  var trace1 = {
    x:unpack(rows, 'x1'), y: unpack(rows, 'y1'), z: unpack(rows, 'z1'),
    mode: 'markers',
      name: 'human',
    marker: {
  size: 3,
  line: {
      color: 'rgba(217, 217, 240, 0.14)',
      width: 0.5},
      opacity: 0.8},
        type: 'scatter3d'
  };
  var trace2 = {
    x:unpack(rows, 'x2'), y: unpack(rows, 'y2'), z: unpack(rows, 'z2'),
    mode: 'markers',
    name: 'bot',
      marker: {
      size: 3,
      line: {
          color: 'rgba(240, 217, 217, 0.14)',
          width: 0.5
          },
      opacity: 0.8
      },
        type: 'scatter3d'
  };

 var data = [trace1, trace2];
  var layout = {
      margin: {
        l: 0,
        r: 0,
        b: 0,
        t: 0
      },
      width: 1000,
      height: 1000
  };
  Plotly.newPlot('dataPlot', data, layout);
});

var errs = {
  x: [1, 2, 3, 4],
  y: [10, 15, 13, 17],
  mode: 'markers',
  type: 'scatter'
};
var data = [errs];

Plotly.newPlot('errorPlot', data);