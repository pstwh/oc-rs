# oc-rs

OCR inference in rust. 

This project compiles to a single 30 mbs binary, but it's still necessary to have onnxruntime and opencv installed on the system.

Using a db_mobilenet_v2 [(DBNet: Real-time Scene Text Detection with Differentiable Binarization)](https://arxiv.org/pdf/1911.08947.pdf) as detector and a crnn_vgg16_bn [(CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition)](https://arxiv.org/pdf/1507.05717.pdf) as recognizer. In principle this code can run any model that uses the approach of **detector → recognizer**. In this case the project is using the recognizer trained by [docTR](https://github.com/mindee/doctr/). (The detector is an older version). If you want, you can train your own model.

```bash
oc-rs 0.1.0
pauloasjx

USAGE:
    oc-rs <FILE_PATH> [ARGS]

ARGS:
    <FILE_PATH>
    <FORMAT>       [default: json] [possible values: json, text]
    <resize>

OPTIONS:
    -h, --help       Print help information
    -V, --version    Print version information

```

Example
```bash
$ oc-rs "samples/sample.png" json | jq
```
<br/>
<center>
<img style="height:300px;" src="samples/sample.gif"/>
</center>
<br/>
