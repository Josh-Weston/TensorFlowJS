const fs = require('fs');
const csv = require('csv'); //global module
// process.chdir("C:\\users\\jweston\\Desktop");

// var csv_file = fs.readFileSync("U:\\users\\jweston\\Desktop\\sales_item.csv", "utf8");
var csv_file = fs.readFileSync("U:\\Desktop\\sales_item.csv", "utf8");
csv.parse(csv_file, {from: 2, to: 300}, function(error, data) {
    if (error) console.log(error);
    var toJSON = JSON.stringify(data);
    fs.writeFileSync("data\\sales_item.json", toJSON, {encoding: 'utf8'}) //default to utf8 anyway;
    console.log('file written successfully');
});

/* csv.parse(csv_file, {from:2, to:145}, function(error, data) {
    var toJSON = JSON.stringify(data);
    fs.writeFileSync("data\\international-airline-passengers.json", toJSON, {encoding: 'utf8'}) //default to utf8 anyway;
    console.log('file written successfully');
}); */

