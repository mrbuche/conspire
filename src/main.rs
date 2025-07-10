use std::{
    fs::{create_dir_all, write},
    io::Error,
    path::Path,
};

use conspire::constitutive::solid::{
    elastic::doc::almansi_hamel,
    hyperelastic::{
        doc::arruda_boyce, doc::fung, doc::gent, doc::mooney_rivlin, doc::neo_hookean,
        doc::saint_venant_kirchhoff,
    },
};

fn main() -> Result<(), Error> {
    let models = [
        almansi_hamel(),
        arruda_boyce(),
        fung(),
        gent(),
        mooney_rivlin(),
        neo_hookean(),
        saint_venant_kirchhoff(),
    ];
    models.iter().try_for_each(|model| {
        let path = model[0][0];
        create_dir_all(Path::new(format!("target/doc/{path}").as_str()))?;
        write(
            Path::new(format!("target/doc/{path}/doc.md").as_str()),
            model[0][1],
        )?;
        model.iter().skip(1).try_for_each(|[method, doc]| {
            if doc.is_empty() {
                write(
                    Path::new(format!("target/doc/{path}/{method}.md").as_str()),
                    "@private",
                )
            } else {
                write(
                    Path::new(format!("target/doc/{path}/{method}.md").as_str()),
                    doc,
                )
            }
        })
    })
}
