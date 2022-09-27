use clap::Parser;

use crate::cli::{OcrsArgs, OcrsFormat};
use crate::detector::Detector;
use crate::recognizer::{Recognizer, WordBox};

use crate::utils::resize_with_pad;
use opencv::core::Size;

mod cli;
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

    let vocab = String::from("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°àâéèêëîïôùûçÀÂÉÈËÎÏÔÙÛÇ£€¥¢฿");

    let recognizer = Recognizer::new(
        include_bytes!("../models/crnn_vgg16_bn.ort"),
        vec![32, 128, 3],
        0.694,
        0.298,
        vocab,
        123,
        124,
    );

    let args = OcrsArgs::parse();

    let mut image = opencv::imgcodecs::imread(&args.file_path, 1).expect("Invalid image file!");
    if args.resize {
        image = resize_with_pad(
            &image,
            Size {
                width: 1000,
                height: 1000,
            },
        );
    }

    let (contours, scale) = detector.get_contours(image.clone());
    let word_bboxes: Vec<WordBox> = recognizer.get_words(image, contours, scale);
    match args.format {
        OcrsFormat::Json => {
            print!(
                "{}",
                serde_json::to_string(&word_bboxes).expect("Invalid result!")
            );
        }
        OcrsFormat::Text => {
            let words: Vec<String> = word_bboxes.into_iter().map(|x| x.word).collect();
            print!("{}", words.join(" "));
        }
    }
}
