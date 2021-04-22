use std::env;
use std::path::PathBuf;

fn main() {
    let lib_path = PathBuf::from(env::current_dir().unwrap().join("."));

    println!("cargo:rustc-link-search={}", lib_path.display());
    println!("cargo:rustc-link-lib=tengine-lite");
    println!("cargo:rustc-link-lib=ncnn");
}
