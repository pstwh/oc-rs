use std::ops::Mul;

use image::ImageBuffer;
use image::Rgb;
use onnxruntime::{
    environment::Environment,
    ndarray::{Array4, Dim, IxDynImpl},
    tensor::OrtOwnedTensor,
    GraphOptimizationLevel,
};
use opencv::core::Point2f;
use opencv::prelude::{DataType, MatTraitConst};
use opencv::types::VectorOfMat;
use opencv::{imgproc, prelude::Mat};

use crate::utils::array2_to_mat;

pub struct Detector<'a> {
    model_bytes: &'a [u8],
    shape: Vec<u32>,
    mean: f32,
    std: f32,
    thresh_min: f32,
    thresh_max: f32,
    kernel_size: u8,
}

impl<'a> Detector<'a> {
    pub fn new(
        model_bytes: &'a [u8],
        shape: Vec<u32>,
        mean: f32,
        std: f32,
        thresh_min: f32,
        thresh_max: f32,
        kernel_size: u8,
    ) -> Self {
        Self {
            model_bytes,
            shape,
            mean,
            std,
            thresh_min,
            thresh_max,
            kernel_size,
        }
    }

    fn prepare_input(&self, image: ImageBuffer<Rgb<u8>, Vec<u8>>) -> (Array4<f32>, Point2f) {
        let shape = image.dimensions();
        let width0 = shape.0;
        let height0 = shape.1;
        let width = self.shape.get(0).expect("Invalid width, check shape");
        let height = self.shape.get(1).expect("Invalid height, check shape");
        let resized = image::imageops::resize(
            &image,
            *width,
            *height,
            ::image::imageops::FilterType::Triangle,
        );

        let scale = Point2f {
            x: width0.to_owned() as f32 / width.to_owned() as f32,
            y: height0.to_owned() as f32 / height.to_owned() as f32,
        };

        let mean = 255.0 * self.mean;
        let std = 255.0 * self.std;

        let array =
            Array4::from_shape_fn((1, *width as usize, *height as usize, 3), |(_, y, x, c)| {
                (resized[(x as _, y as _)][c] as f32 - mean) / std
            })
            .into();

        (array, scale)
    }

    pub fn get_mask(&self, image: ImageBuffer<Rgb<u8>, Vec<u8>>) -> (Mat, Point2f) {
        let width = self.shape.get(0).expect("Invalid width, check shape");
        let height = self.shape.get(1).expect("Invalid height, check shape");
        let (array, scale) = self.prepare_input(image);
        let input_tensor_values = vec![array];
        let env = Environment::builder().with_name("env").build().unwrap();

        let mut session = env
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .unwrap()
            .with_model_from_memory(self.model_bytes)
            .unwrap();

        let output: Vec<OrtOwnedTensor<f32, Dim<IxDynImpl>>> =
            session.run(input_tensor_values).unwrap();
        let prediction = output.first().unwrap().to_owned().as_standard_layout();
        let raw = prediction
            .into_shape((*width as usize, *height as usize, 1))
            .unwrap()
            .mul(255.0)
            .mapv(|elem| elem as f32);

        let cv_raw = array2_to_mat(raw);
        let mut cv_threshold = Mat::default();
        imgproc::threshold(
            &cv_raw,
            &mut cv_threshold,
            self.thresh_min.into(),
            self.thresh_max.into(),
            imgproc::THRESH_BINARY,
        )
        .unwrap();

        let kernel = Mat::ones(self.kernel_size.into(), self.kernel_size.into(), 0).unwrap();
        let mut cv_morphology_ex = Mat::default();
        imgproc::morphology_ex(
            &cv_threshold,
            &mut cv_morphology_ex,
            imgproc::MORPH_OPEN,
            &kernel,
            opencv::core::Point::new(-1, -1),
            1,
            opencv::core::BORDER_CONSTANT,
            opencv::core::Scalar::default(),
        )
        .unwrap();

        let mut cv_converted = Mat::default();
        cv_morphology_ex
            .convert_to(&mut cv_converted, u8::typ(), 1.0, 0.0)
            .unwrap();

        (cv_converted, scale)
    }

    pub fn get_contours(&self, image: ImageBuffer<Rgb<u8>, Vec<u8>>) -> (VectorOfMat, Point2f) {
        let (mask, scale) = self.get_mask(image);
        let mut contours = VectorOfMat::new();
        imgproc::find_contours(
            &mask,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            opencv::core::Point::new(0, 0),
        )
        .unwrap();

        (contours, scale)
    }
}
