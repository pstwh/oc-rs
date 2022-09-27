use std::ops::Deref;

use onnxruntime::{
    environment::Environment,
    ndarray::{Array3, Array4},
    tensor::OrtOwnedTensor,
    GraphOptimizationLevel,
};
use opencv::{
    core::{Point2f, Rect, Rect_, Vec3b},
    imgproc,
    prelude::{Mat, *},
    types::VectorOfMat,
};
use serde::{Deserialize, Serialize};

use crate::utils::{clamp, closest, max_index, metric, ndarray2mat};

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

    pub fn get_words(&self, image: Mat, contours: VectorOfMat, scale: Point2f) -> Vec<WordBox> {
        let width0 = image.cols();
        let height0 = image.rows();

        let vec = Mat::data_typed::<Vec3b>(&image).unwrap();

        let image0: Array3<f32> =
            Array3::from_shape_fn((height0 as usize, width0 as usize, 3), |(y, x, c)| {
                Vec3b::deref(&vec[x + y * width0 as usize])[c] as f32
            })
            .into();
        let cv_image0 = ndarray2mat(image0, 21);
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
        let bboxes: Vec<Rect_<i32>> = contours
            .iter()
            .map(|contour| {
                let rect = imgproc::bounding_rect(&contour).unwrap();
                if rect.width > 2 && rect.height > 2 {
                    let x0 = (rect.x as f32 * scale.x) as i32;
                    let y0 = (rect.y as f32 * scale.y) as i32;
                    let x = (rect.width as f32 * scale.x) as i32;
                    let y = (rect.height as f32 * scale.y) as i32;
                    let offset =
                        ((x as f32 * y as f32 * 1.8) / (2.0 * (x as f32 + y as f32))) as i32;

                    let nx0 = clamp(x0 - offset, width0.to_owned() as i32);
                    let ny0 = clamp(y0 - offset, height0.to_owned() as i32);
                    let nx = clamp(nx0 + x + 2 * offset, width0.to_owned() as i32);
                    let ny = clamp(ny0 + y + 2 * offset, height0.to_owned() as i32);

                    return Some(Rect::new(nx0, ny0, nx - nx0, ny - ny0));
                }
                None
            })
            .flatten()
            .collect();

        let mut lines: Vec<i32> = bboxes
            .iter()
            .map(|bbox| bbox.x + bbox.width / 2)
            .fold((Vec::new(), 0), |(mut nbboxes, mut greater_y), last_y| {
                if (greater_y - last_y).abs() > 8 {
                    greater_y = last_y;
                    nbboxes.push(last_y)
                }
                (nbboxes, greater_y)
            })
            .0;

        lines.sort();

        let mut bboxes_test: Vec<(i32, Rect_<i32>)> = bboxes
            .iter()
            .map(|bbox| (closest(&lines, &bbox.y), bbox.clone()))
            .collect();

        bboxes_test.sort_by(|a, b| a.0.cmp(&b.0));

        let mut bboxes_n: (Vec<Rect_<i32>>, Vec<Rect_<i32>>, i32) = bboxes_test.iter().fold(
            (Vec::new(), Vec::new(), -1),
            |(mut result, mut temp, current), (tag, bbox)| {
                if current == -1 {
                    temp.push(bbox.clone());
                    (result, temp, tag.to_owned())
                } else if current == tag.to_owned() {
                    temp.push(bbox.clone());
                    (result, temp, current)
                } else if current != tag.to_owned() && temp.len() != 0 {
                    temp.sort_by(|a, b| a.x.cmp(&b.x));
                    result = [result, temp].concat();
                    temp = Vec::new();
                    temp.push(bbox.clone());
                    (result, temp, tag.to_owned())
                } else {
                    (result, temp, tag.to_owned())
                }
            },
        );

        bboxes_n.1.sort_by(|a, b| a.x.cmp(&b.x));

        let bboxes_result = [bboxes_n.0, bboxes_n.1].concat();

        bboxes_result.iter().for_each(|bbox| {
            let roi = opencv::core::Mat::roi(&cv_image0_u8, bbox.to_owned()).unwrap();
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

            let mean = 255.0 * self.mean;
            let std = 255.0 * self.std;

            let vec = Mat::data_typed::<Vec3b>(&roi_resized).unwrap();

            let roi_array = Array4::from_shape_fn((1, *height, *width, 3), |(_, y, x, c)| {
                (Vec3b::deref(&vec[x + y * width])[c] as f32 - mean) / std
            })
            .into();

            let input_tensor_values = vec![roi_array];
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
                    xmin: bbox.x,
                    ymin: bbox.y,
                    xmax: bbox.x + bbox.width,
                    ymax: bbox.y + bbox.height,
                },
                confidence: (raw.into_iter().map(|(_, conf)| conf).product::<f32>() * 100.0)
                    .round()
                    / 100.0,
            };
            word_boxes.push(word_box);
        });
        word_boxes
    }
}
