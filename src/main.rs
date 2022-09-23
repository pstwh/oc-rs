use std::env;

use crate::detector::Detector;
use crate::recognizer::Recognizer;
use crate::recognizer::WordBox;

mod detector;
mod recognizer;
mod utils;


fn main() {
    let detector = Detector::new(
        include_bytes!("../models/db_mobilenet_v2.ort"),
        vec![512, 512, 3],
        0.785,
        0.275,
        77.0,
        255.0,
        2,
    );

    let recognizer = Recognizer::new(
        include_bytes!("../models/crnn_vgg16_bn.ort"), 
        vec![32, 128, 3], 
        0.694, 
        0.298, 
        "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°àâéèêëîïôùûçÀÂÉÈËÎÏÔÙÛÇ£€¥¢฿".to_string(), 
        123,
        124
    );


    let args: Vec<String> = env::args().collect();
    let file_path = &args[1];

    let image = image::open(file_path).unwrap().to_rgb8();
    let (contours, scale) = detector.get_contours(image.clone());
    if args.len() < 3 {
        let words: Vec<WordBox> = recognizer.get_words(image, contours, scale);
        print!("{}", serde_json::to_string(&words).unwrap());
    } else {
        let words: Vec<String> = recognizer.get_words(image, contours, scale).into_iter().map(|x| x.word).collect();
        print!("{}", words.join(" "));
    }
}

