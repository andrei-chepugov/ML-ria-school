console.log("Hello Tensorflow");

const tf = require("@tensorflow/tfjs");
const fs = require("fs");
const path = require("path");
const tfnode = require("@tensorflow/tfjs-node");

function loadDataset(baseDir = "./data/seg_train/seg_train") {
    const folders = fs.readdirSync(baseDir);
    const allFiles = [];
    for (const i in folders) {
        const folder = folders[i];
        const files = fs.readdirSync(path.join(baseDir, folder));
        allFiles.push(...files.map(f => {
            return [
                path.join(baseDir, folder, f),
                i
            ]
        }));
    }

    let images = [];
    let labels = [];
    for (const [file, i] of allFiles) {
        let buffer = fs.readFileSync(file);
        let tfimage = tfnode.node.decodeImage(buffer, chanels = 3);
        tfimage = tf.image.resizeBilinear(tfimage, [28, 28]);
        tfimage = tfimage.cast("float32").div(255);
        images.push(tfimage);
        labels.push(i);
    }
    return [
        tf.stack(images),
        tf.oneHot(tf.tensor1d(labels, 'int32'), 6)
    ]
}


function onEpochEnd(data, logs) {
    console.log("loss", logs.loss);
}

async function main() {
    const [xs, ys] = loadDataset();

    const model = tfnode.sequential();
    model.add(tfnode.layers.conv2d({
        inputShape: [28, 28, 3],
        filters: 32,
        kernelSize: 3,
        activation: "relu"
    }));
    model.add(tfnode.layers.maxPool2d({
        inputShape: [28, 28, 3],
        poolSize: [2, 2],
        strides: [1, 1]
    }));

    model.add(tfnode.layers.conv2d({
        inputShape: [14, 14, 6],
        filters: 32,
        kernelSize: 3,
        activation: "relu"
    }));
    model.add(tfnode.layers.maxPool2d({
        inputShape: [14, 14, 6],
        poolSize: [2, 2],
        strides: [1, 1]
    }));

    model.add(tfnode.layers.conv2d({
        inputShape: [10, 10, 16],
        filters: 32,
        kernelSize: 3,
        activation: "relu"
    }));
    model.add(tfnode.layers.maxPool2d({
        inputShape: [10, 10, 16],
        poolSize: [2, 2],
        strides: [1, 1]
    }));
    model.add(tfnode.layers.conv2d({
        inputShape: [5, 5, 16],
        filters: 32,
        kernelSize: 3,
        activation: "relu"
    }));
    model.add(tfnode.layers.maxPool2d({
        inputShape: [5, 5, 16],
        poolSize: [2, 2],
        strides: [1, 1]
    }));

    model.add(tfnode.layers.flatten());

    model.add(tfnode.layers.dense({
        units: 512,
        activation: "relu"
    }));
    model.add(tfnode.layers.dense({
        units: 6,
        activation: "softmax"
    }));
    model.compile({
        optimizer: tfnode.train.sgd(0.01),
        loss: 'categoricalCrossentropy',
        metrics: ["accuracy"]
    });

    const h = await model.fit(xs, ys, {
        epochs: 100,
        batchSize: 50
    });
    await model.save("file://./cnn-model");
}

main();