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
  var encoded_filename = req.file.filename;
  var path = 'upload/' + encoded_filename;
  var options = {
    args: [path]
  };
  PythonShell.run('nn_real_data.py', options, function(err, results){
    if (err) throw err;
    console.log(results);
    return res.send(JSON.stringify(results));
  });
});

app.listen(3000);
console.log("Listening on port 3000");
