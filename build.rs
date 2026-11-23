use std::{
    collections::BTreeSet,
    env::var_os,
    fs::{metadata, read_dir, read_to_string},
    path::Path,
};

const APPLICATIONS: &str = "kernels";
const PTX: &str = "src/gpu/kernels";

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed={APPLICATIONS}");
    println!("cargo::rerun-if-changed={PTX}");
    if var_os("CARGO_FEATURE_GPU").is_none() {
        return;
    }
    let applications = applications();
    println!(
        "cargo::rustc-check-cfg=cfg(conspire_ptx, values({}))",
        applications
            .iter()
            .map(|application| format!("\"{application}\""))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for application in &applications {
        let ptx = Path::new(PTX).join(format!("{application}.ptx"));
        if ptx.is_file() {
            stale(application, &ptx);
            println!("cargo::rustc-cfg=conspire_ptx=\"{application}\"");
        }
    }
}

fn applications() -> BTreeSet<String> {
    let mut applications = BTreeSet::new();
    for entry in read_dir(APPLICATIONS).into_iter().flatten().flatten() {
        if entry.path().join("lib.rs").is_file() {
            applications.extend(entry.file_name().to_str().map(String::from));
        }
    }
    for entry in read_dir(PTX).into_iter().flatten().flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|extension| extension == "ptx") {
            applications.extend(
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .map(String::from),
            );
        }
    }
    applications
}

fn stale(application: &str, ptx: &Path) {
    if !Path::new(".git").exists() {
        return;
    }
    let listing = Path::new(PTX).join(format!("{application}.d"));
    let Ok(listing) = read_to_string(listing) else {
        return;
    };
    let modified = |path: &Path| metadata(path).and_then(|metadata| metadata.modified());
    let Ok(built) = modified(ptx) else {
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
            "cargo::warning={} is older than the sources it was built from: run kernels/build.sh {application}",
            ptx.display()
        );
    }
}
