use image::RgbImage;
use onnxruntime::ndarray::Array3;
use opencv::core::copy_make_border;
use opencv::core::Mat_AUTO_STEP;
use opencv::core::Scalar;
use opencv::prelude::MatTraitConst;
use opencv::{
    core::Point2f,
    imgproc,
    prelude::{DataType, Mat},
};
use std::cmp::{max, min};
use std::os::raw::{c_int, c_void};

pub fn array2_to_mat(data: Array3<f32>) -> Mat {
    // transforms an Array2 into a opencv Mat data type.
    let rows = data.shape()[0];
    let cols = data.shape()[1];
    let data_ptr: *mut c_void = data.as_ptr() as *mut c_int as *mut c_void;
    let m = unsafe {
        Mat::new_rows_cols_with_data(
            rows as i32,
            cols as i32,
            f32::typ(),
            data_ptr,
            Mat_AUTO_STEP,
        )
        .unwrap()
    };
    m
}

pub fn array2_to_mat2(data: Array3<f32>) -> Mat {
    // transforms an Array2 into a opencv Mat data type.
    let rows = data.shape()[0];
    let cols = data.shape()[1];
    let data_ptr: *mut c_void = data.as_ptr() as *mut c_int as *mut c_void;
    let m = unsafe {
        Mat::new_rows_cols_with_data(rows as i32, cols as i32, 21, data_ptr, Mat_AUTO_STEP).unwrap()
    };
    m
}

pub fn max_index(array: &[f32]) -> usize {
    let mut i = 0;

    for (j, &value) in array.iter().enumerate() {
        if value > array[i] {
            i = j;
        }
    }

    i
}

pub fn array_to_rgb(arr: &[u8]) -> RgbImage {
    let raw = arr.to_vec();

    RgbImage::from_raw(128 as u32, 32 as u32, raw)
        .expect("container should have the right size for the image dimensions")
}

pub fn clamp(n: i32, size: i32) -> i32 {
    max(0, min(n, size))
}

pub fn metric(outputs: &[f32]) -> f32 {
    let exp: Vec<f32> = outputs
        .into_iter()
        .map(|x| f32::powf(2.71828, *x))
        .collect();
    let sum: f32 = exp.iter().sum();
    let softmax: Vec<f32> = exp.iter().map(|x| x / sum).collect();

    return softmax.iter().cloned().fold(0. / 0., f32::max);
}

pub fn resize_with_pad(image: Mat, shape: Point2f) -> Mat {
    let height0 = image.rows();
    let width0 = image.cols();
    let height = shape.y as i32;
    let width = shape.x as i32;

    let ratio = max(height, width) as f32 / max(height0, width0) as f32;
    let nheight = (height0 as f32 * ratio) as i32;
    let nwidth = (width0 as f32 * ratio) as i32;

    let mut resized = Mat::default();
    imgproc::resize(
        &image,
        &mut resized,
        opencv::core::Size::new(nwidth.to_owned() as i32, nheight.to_owned() as i32),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )
    .unwrap();

    let dwidth = width - nwidth;
    let dheight = height - nheight;

    let top = dheight / 2;
    let bottom = dheight - (dheight / 2);
    let left = dwidth / 2;
    let right = dwidth - (dwidth / 2);

    let mut filled = Mat::default();
    copy_make_border(
        &image,
        &mut filled,
        top,
        bottom,
        left,
        right,
        0,
        Scalar::default(),
    );

    filled
}
