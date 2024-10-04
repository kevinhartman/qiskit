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

use conan::*;
use std::{env, path::Path, path::PathBuf};

fn main() {
    let command = InstallCommandBuilder::new()
        .build_policy(BuildPolicy::Missing)
        .with_options(&["symengine:integer_class=boostmp", "symengine:shared=False"])
        .recipe_path(Path::new("conanfile.txt"))
        .build();

    let build_info = command.generate().expect("Failed to run conan install");
    println!("using conan build info");
    build_info.cargo_emit();

    // for test
    println!("cargo:rustc-link-lib=static=symengine");
    println!("cargo:rustc-link-lib=dylib=teuchos");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=c");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    let mut include_paths = Vec::<&str>::new();
    for dependency in build_info.dependencies() {
        for include_path in dependency.get_include_dirs() {
            if include_path.contains("symengine") {
                include_paths.push(include_path);
            }
        }
    }

    let bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{}/", include_paths[0]))
        .header("wrapper.h")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings");
}