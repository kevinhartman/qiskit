// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::{env, path::PathBuf};
use vcpkg;

fn main() {
    let build_info = vcpkg::Config::new()
        .emit_includes(true)
        .find_package("symengine").unwrap();

    let include_path = match build_info.include_paths.as_slice() {
        [path] => {
            path.as_os_str().to_str().unwrap()
        },
        _ => panic!("Include path not found!")
    };

    println!("INC PATH: {}", &include_path);

    let bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{}/", include_path))
        .header("wrapper.h")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings");
}