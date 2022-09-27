use std::{
    cmp::{max, min},
    ops::Deref,
    os::raw::{c_int, c_void},
};

use onnxruntime::ndarray::{Array3, Array4};
use opencv::{
    core::{copy_make_border, Mat_AUTO_STEP, Scalar, Vec3b},
    imgproc,
    prelude::{Mat, MatTraitConst, MatTraitConstManual},
};

pub fn ndarray2mat(data: Array3<f32>, typ: i32) -> Mat {
    let rows = data.shape()[0];
    let cols = data.shape()[1];
    let data_ptr: *mut c_void = data.as_ptr() as *mut c_int as *mut c_void;
    let m = unsafe {
        Mat::new_rows_cols_with_data(rows as i32, cols as i32, typ, data_ptr, Mat_AUTO_STEP)
            .unwrap()
    };
    m
}

pub fn mat2input(mat: &Mat, height: usize, width: usize, mean: f32, std: f32) -> Array4<f32> {
    let vec = Mat::data_typed::<Vec3b>(mat).unwrap();

    let array = Array4::from_shape_fn((1, width, height, 3), |(_, y, x, c)| {
        (Vec3b::deref(&vec[x + y * width as usize])[c] as f32 - mean) / std
    })
    .into();

    return array;
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

pub fn closest(list: &Vec<i32>, value: &i32) -> i32 {
    let mut last_value = 0;
    let mut last_difference = 10000;
    for n in 0..list.len() {
        let v = list.get(n).unwrap();
        let difference: i32 = (v.clone() - value.clone()).abs();
        if difference > last_difference {
            break;
        }
        last_value = v.clone();
        last_difference = difference;
    }

    return last_value;
}

pub fn resize_with_pad(image: &Mat, shape: opencv::core::Size) -> Mat {
    let height0 = image.rows();
    let width0 = image.cols();
    let height = shape.height as i32;
    let width = shape.width as i32;

    let ratio = (height as f32 / height0 as f32).min(width as f32 / width0 as f32);
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
        &resized,
        &mut filled,
        top,
        bottom,
        left,
        right,
        0,
        Scalar::default(),
    )
    .unwrap();

    filled
}
