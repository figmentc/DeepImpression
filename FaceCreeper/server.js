"use strict";

var express = require('express');
var bodyParser = require('body-parser');
var PythonShell = require('python-shell');
var multer = require('multer');

var upload = multer({dest: 'upload/'});
var type = upload.single('img');

var app = express();
app.use(express.static(__dirname + '/'));
app.use(bodyParser.json());


app.post('/process', type, function(req, res){
  console.log(req.file);
  console.log(req.body.dimensions)

  var encoded_filename = req.file.filename;
  var path = 'upload/' + encoded_filename;

  var dimensions = req.body.dimensions;

  var crop_options = {
    args: [path, dimensions]
  };

  PythonShell.run('pre-process.py', crop_options, function(err, cropped_img){
    console.log(cropped_img)
    if (err) throw err;
    console.log(cropped_img);
    var nn_options = {
      args: [cropped_img]
    }
    PythonShell.run('last_resort.py', nn_options, function(err, results){
      if (err) throw err;
      console.log("RESULTS: " + results);
      res.sendStatus(results)
      return res.send(JSON.stringify(results));
    });
  });
});
app.listen(3000);
console.log("Listening on port 3000");
