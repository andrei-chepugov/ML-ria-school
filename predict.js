console.log("Hello Tensorflow");

const tf = require("@tensorflow/tfjs");
const tfnode = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");

async function main() {
    const model = await tf.loadLayersModel("file:///cnn-model/model.json");
    let file = "/var/www/ria_school_2021/data/seg_test/seg_test/buildings/20057.jpg";
    let buffer = fs.readFileSync(file);
    let tfimage = tfnode.node.decodeImage(buffer, chanels=3);
    tfimage = tf.image.resizeBilinear(tfimage, [28, 28]);
    tfimage = tfimage.cast("float32").div(255);

    const pred = model.predict(tf.stack([tfimage]));
    pred.print();
}

main();

