use std::{
    fs::{create_dir_all, write},
    io::Error,
    path::Path,
};

use conspire::constitutive::solid::{
    elastic::doc::{DOC as ELASTIC, almansi_hamel},
    hyperelastic::doc::{
        DOC as HYPERELASTIC, arruda_boyce, fung, gent, mooney_rivlin, neo_hookean,
        saint_venant_kirchhoff, yeoh,
    },
};

fn main() -> Result<(), Error> {
    let models = [
        vec![["constitutive/solid/elastic", ELASTIC]],
        almansi_hamel(),
        vec![["constitutive/solid/hyperelastic", HYPERELASTIC]],
        arruda_boyce(),
        fung(),
        gent(),
        mooney_rivlin(),
        neo_hookean(),
        saint_venant_kirchhoff(),
        yeoh(),
    ];
    let mut path = "";
    models.iter().try_for_each(|model| {
        path = model[0][0];
        create_dir_all(Path::new(format!("target/doc/{path}").as_str()))?;
        write(
            Path::new(format!("target/doc/{path}/doc.md").as_str()),
            model[0][1].replace("super::NeoHookean", "neo_hookean.html"),
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
