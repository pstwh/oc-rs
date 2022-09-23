use image::ImageBuffer;
use image::Rgb;
use onnxruntime::{
    environment::Environment,
    ndarray::{Array3, Array4},
    tensor::OrtOwnedTensor,
    GraphOptimizationLevel,
};
use opencv::core::Point2f;
use opencv::types::VectorOfMat;
use opencv::{core::Rect, prelude::*};
use opencv::{imgproc, prelude::Mat};
use serde::{Deserialize, Serialize};

use crate::utils::{array2_to_mat2, array_to_rgb, clamp, max_index, metric};
#[derive(Debug, Serialize, Deserialize)]
pub struct Bbox {
    xmin: i32,
    ymin: i32,
    xmax: i32,
    ymax: i32,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct WordBox {
    pub word: String,
    bbox: Bbox,
    confidence: f32,
}

pub struct Recognizer<'a> {
    model_bytes: &'a [u8],
    shape: Vec<usize>,
    mean: f32,
    std: f32,
    vocab: String,
    blank: usize,
    chunk_size: usize,
}

impl<'a> Recognizer<'a> {
    pub fn new(
        model_bytes: &'a [u8],
        shape: Vec<usize>,
        mean: f32,
        std: f32,
        vocab: String,
        blank: usize,
        chunk_size: usize,
    ) -> Self {
        Self {
            model_bytes,
            shape,
            mean,
            std,
            vocab,
            blank,
            chunk_size,
        }
    }

    pub fn get_words(
        &self,
        image: ImageBuffer<Rgb<u8>, Vec<u8>>,
        contours: VectorOfMat,
        scale: Point2f,
    ) -> Vec<WordBox> {
        let shape = image.dimensions();
        let width0 = shape.0;
        let height0 = shape.1;
        let image0: Array3<f32> =
            Array3::from_shape_fn((height0 as usize, width0 as usize, 3), |(y, x, c)| {
                image[(x as _, y as _)][c] as f32
            })
            .into();
        let cv_image0 = array2_to_mat2(image0);
        let mut cv_image0_u8 = Mat::default();
        cv_image0
            .convert_to(&mut cv_image0_u8, u8::typ(), 1.0, 0.0)
            .unwrap();

        let env = Environment::builder().with_name("env").build().unwrap();

        let mut session = env
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .unwrap()
            .with_model_from_memory(self.model_bytes)
            .unwrap();

        let vocab: Vec<char> = self.vocab.chars().collect();
        let height = self.shape.get(0).expect("Invalid height, check shape");
        let width = self.shape.get(1).expect("Invalid with, check shape");

        let mut word_boxes: Vec<WordBox> = vec![];
        contours.iter().enumerate().for_each(|(idx, contour)| {
            let rect = imgproc::bounding_rect(&contour).unwrap();
            if rect.width > 2 && rect.height > 2 {
                let x = (rect.x as f32 * scale.x) as i32;
                let y = (rect.y as f32 * scale.y) as i32;
                let w = (rect.width as f32 * scale.x) as i32;
                let h = (rect.height as f32 * scale.y) as i32;
                let offset = ((w as f32 * h as f32 * 1.8) / (2.0 * (w as f32 + h as f32))) as i32;

                let nx = clamp(x - offset, width0.to_owned() as i32);
                let nw = clamp(nx + w + 2 * offset, width0.to_owned() as i32);
                let ny = clamp(y - offset, height0.to_owned() as i32);
                let nh = clamp(ny + h + 2 * offset, height0.to_owned() as i32);

                let nrect = Rect::new(nx, ny, nw - nx, nh - ny);
                let roi = opencv::core::Mat::roi(&cv_image0_u8, nrect).unwrap();
                let mut roi_resized = Mat::default();
                imgproc::resize(
                    &roi,
                    &mut roi_resized,
                    opencv::core::Size::new(width.to_owned() as i32, height.to_owned() as i32),
                    0.0,
                    0.0,
                    imgproc::INTER_LINEAR,
                )
                .unwrap();

                let roi_bytes = roi_resized.data_bytes().unwrap();
                let roi_image = array_to_rgb(&roi_bytes);
                // roi_image.save(format!("{}.jpg", idx)).unwrap();

                let mean = 255.0 * self.mean;
                let std = 255.0 * self.std;

                let roi_image2: Array4<f32> =
                    Array4::from_shape_fn((1, *height, *width, 3), |(_, y, x, c)| {
                        (roi_image[(x as _, y as _)][c] as f32 - mean) / std
                    })
                    .into();

                let input_tensor_values = vec![roi_image2];
                let output: Vec<OrtOwnedTensor<f32, _>> = session.run(input_tensor_values).unwrap();
                let prediction = output.first().unwrap().to_owned().as_standard_layout();
                let raw: Vec<(char, f32)> = prediction
                    .as_slice()
                    .unwrap()
                    .chunks(self.chunk_size)
                    .fold(
                        (Vec::new(), false),
                        |(mut acc, added), chunk| -> (Vec<(char, f32)>, bool) {
                            let idx = max_index(chunk);
                            let result = if idx == self.blank {
                                (acc, false)
                            } else if idx != self.blank && !added {
                                acc.push((vocab.get(idx).unwrap().clone(), metric(chunk)));
                                (acc, true)
                            } else {
                                (acc, false)
                            };
                            result
                        },
                    )
                    .0
                    .into_iter()
                    .collect();

                let word_box: WordBox = WordBox {
                    word: raw.clone().into_iter().map(|(char, _)| char).collect(),
                    bbox: Bbox {
                        xmin: nx,
                        ymin: ny,
                        xmax: nw,
                        ymax: nh,
                    },
                    confidence: (raw.into_iter().map(|(_, conf)| conf).product::<f32>() * 100.0)
                        .round()
                        / 100.0,
                };
                word_boxes.push(word_box);
            }
        });
        word_boxes
    }
}
