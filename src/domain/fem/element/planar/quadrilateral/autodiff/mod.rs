use crate::fem::element::solid::hyperelastic::autodiff::autodiff_element;

autodiff_element!(2, 4, 4, 1);

#[cfg(test)]
mod test;
