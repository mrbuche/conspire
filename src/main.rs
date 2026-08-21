use std::{
    fs::{create_dir_all, write},
    io::Error,
    path::Path,
};

use conspire::{
    constitutive::{
        solid::{
            elastic::doc::{
                DOC as ELASTIC, almansi_hamel, hencky as hencky_elastic,
                saint_venant_kirchhoff as saint_venant_kirchhoff_elastic,
            },
            hyperelastic::doc::{
                DOC as HYPERELASTIC, arruda_boyce, eight_chain, fung, gent, hencky,
                mooney_rivlin, neo_hookean, saint_venant_kirchhoff, yeoh,
            },
        },
        thermal::conduction::doc::{DOC as THERMAL_CONDUCTION, fourier},
    },
    math::integrate::doc::{
        EXPLICIT, IMPLICIT, backward_euler, bogacki_shampine, dormand_prince, verner_8, verner_9,
    },
};

fn main() -> Result<(), Error> {
    math()?;
    constitutive()?;
    thermal()
}

fn math() -> Result<(), Error> {
    let methods = [
        vec![["math/integrate/explicit", EXPLICIT]],
        bogacki_shampine(),
        dormand_prince(),
        verner_8(),
        verner_9(),
        vec![["math/integrate/implicit", IMPLICIT]],
        backward_euler(),
    ];
    let mut path = "";
    methods.iter().try_for_each(|method| {
        path = method[0][0];
        create_dir_all(Path::new(format!("target/doc/{path}").as_str()))?;
        write(
            Path::new(format!("target/doc/{path}/doc.md").as_str()),
            method[0][1],
        )
    })
}

fn constitutive() -> Result<(), Error> {
    write_models(&[
        vec![["constitutive/solid/elastic", ELASTIC]],
        almansi_hamel(),
        hencky_elastic(),
        saint_venant_kirchhoff_elastic(),
        vec![["constitutive/solid/hyperelastic", HYPERELASTIC]],
        arruda_boyce(),
        eight_chain(),
        fung(),
        gent(),
        hencky(),
        mooney_rivlin(),
        neo_hookean(),
        saint_venant_kirchhoff(),
        yeoh(),
    ])
}

fn thermal() -> Result<(), Error> {
    write_models(&[
        vec![["constitutive/thermal/conduction", THERMAL_CONDUCTION]],
        fourier(),
    ])
}

fn write_models(models: &[Vec<[&str; 2]>]) -> Result<(), Error> {
    let mut path = "";
    models.iter().try_for_each(|model| {
        path = model[0][0];
        create_dir_all(Path::new(format!("target/doc/{path}").as_str()))?;
        write(
            Path::new(format!("target/doc/{path}/doc.md").as_str()),
            model[0][1]
                .replace("super::NeoHookean", "neo_hookean.html")
                .replace("super::ArrudaBoyce", "arruda_boyce.html"),
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
