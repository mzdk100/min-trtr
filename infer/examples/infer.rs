use min_trtr::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let stream = translate_stream(
        "../checkpoint/translation_encoder.onnx",
        "../checkpoint/translation_decoder.onnx",
        "Hello world",
    )
    .await?;

    pin_mut!(stream);
    while let Some(item) = stream.next().await {
        print!("{:?}", item?);
    }

    Ok(())
}
