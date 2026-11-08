use std::{
    env, fs,
    path::{Path, PathBuf},
};

const PTX: &str = "src/domain/gpu/kernels.ptx";
const SOURCES: &str = "src/domain/gpu/kernels.d";

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/domain/gpu");
    if env::var_os("CARGO_FEATURE_GPU").is_none() {
        return;
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo gives the directory of outputs"))
        .join("kernels.ptx");
    match fs::read(PTX) {
        Ok(ptx) => {
            fs::write(&out, ptx).expect("the PTX can be copied");
            stale();
        }
        Err(_) => fs::write(&out, b"").expect("the placeholder can be written"),
    }
}

fn stale() {
    if !Path::new(".git").exists() {
        return;
    }
    let Ok(listing) = fs::read_to_string(SOURCES) else {
        return;
    };
    let modified = |path: &Path| fs::metadata(path).and_then(|metadata| metadata.modified());
    let Ok(built) = modified(Path::new(PTX)) else {
        return;
    };
    let sources = listing
        .split_once(": ")
        .map_or("", |(_, sources)| sources)
        .replace("\\ ", "\u{0}");
    let mut newer = false;
    for source in sources.split_whitespace() {
        let path = Path::new(&source.replace('\u{0}', " ")).to_path_buf();
        println!("cargo::rerun-if-changed={}", path.display());
        newer |= modified(&path).is_ok_and(|modified| modified > built);
    }
    if newer {
        println!(
            "cargo::warning=src/domain/gpu/kernels.ptx is older than the sources it was built from: run kernels/build.sh"
        );
    }
}
