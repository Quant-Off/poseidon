use rust_resampler::dataset_reader;
use rust_resampler::nids_dtype;

use rust_resampler::df_test;

fn main() {
    println!("Hello, world!");

    println!("================================================");
    let result = df_test::smote_test("NF-UNSW-NB15-v3");
    if result.is_err() {
        println!("Error: {:?}", result.unwrap_err());
    } else {
        println!("Success");
    }
    println!("================================================");

    // println!("================================================");
    // println!("Dtypes: \n{:#?}", dtypes_test());
    // println!("================================================");
    
    // println!("================================================");
    // let dataset = dataset_reader::read_dataset("NF-UNSW-NB15-v3");
    // println!("{:?}", dataset);
    // println!("================================================");
}

fn dtypes_test() {
    let dtypes = nids_dtype::dtypes();
    println!("{:?}", dtypes);
}