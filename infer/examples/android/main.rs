use {
    min_trtr::*,
    std::{error::Error, path::Path},
    tokio::runtime::Builder,
};

#[mobile_entry_point::mobile_entry_point]
fn main() {
    let rt = Builder::new_multi_thread().enable_all().build().unwrap();
    rt.block_on(run()).unwrap();
}

async fn run() -> Result<(), Box<dyn Error>> {
    let assets_dir = Path::new("/data/local/tmp");
    let stream = translate_stream(
        assets_dir.join("translation_encoder.onnx"),
        assets_dir.join("translation_decoder.onnx"),
        "thank you for help",
    )
    .await?;

    pin_mut!(stream);
    while let Some(item) = stream.next().await {
        println!("{:?}", item?);
    }

    Ok(())
}
