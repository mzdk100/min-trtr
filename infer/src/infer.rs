pub use futures_util::{StreamExt, pin_mut};
use {
    super::{TranslationError, get_word, get_word_id, get_word_ids},
    async_stream::stream,
    fancy_regex::{Error as RegexError, Regex},
    futures_util::Stream,
    ndarray::{Array, Axis, Slice},
    ort::{inputs, session::RunOptions, session::Session, value::TensorRef},
    std::{
        path::Path,
        sync::LazyLock,
        time::{Duration, SystemTime},
    },
};

fn split_word<S>(sentence: S) -> Result<Vec<String>, TranslationError>
where
    S: AsRef<str>,
{
    static RE: LazyLock<Result<Regex, RegexError>> = LazyLock::new(|| {
        Regex::new(r#"\b\w(?:[\w'-]*\w)?\b|\d{3}(?=\d{3})|\d{1,3}(?!\d)|[^\w\s]"#)
    });

    Ok(RE
        .as_ref()
        .map_err(|e| e.to_owned())?
        .captures_iter(sentence.as_ref())
        .filter_map(|i| i.ok())
        .map(|i| i[0].to_owned())
        .collect())
}

/// 使用给定的编码器和解码器模型对输入的英文句子进行翻译，并返回一个异步流，其中包含翻译后的中文词语和每个词语的翻译时间。
///
/// # 参数
///
/// * `encoder` - 编码器模型的路径。
/// * `decoder` - 解码器模型的路径。
/// * `en_text` - 需要翻译的英文句子。
///
/// # 返回值
///
/// 返回一个异步流，其中每个元素都是一个包含翻译后的中文词语和翻译时间的元组。
///
/// # 错误处理
///
/// 如果在加载模型或运行推理时发生错误，将返回一个`TranslationError`。
///
/// # 示例
///
///```rust
/// use min_trtr::*;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let stream = translate_stream(
///         "../checkpoint/translation_encoder.onnx",
///         "../checkpoint/translation_decoder.onnx",
///         "thank you for help",
///     )
///     .await?;
///
///     pin_mut!(stream);
///     while let Some(item) = stream.next().await {
///         println!("{:?}", item?);
///     }
///
///     Ok(())
/// }
/// ```
pub async fn translate_stream<P, S>(
    encoder: P,
    decoder: P,
    en_text: S,
) -> Result<impl Stream<Item = Result<(&'static str, Duration), TranslationError>>, TranslationError>
where
    P: AsRef<Path>,
    S: AsRef<str>,
{
    const MAX_LEN: usize = 20;

    let mut encoder = Session::builder()?.commit_from_file(encoder)?;
    let mut decoder = Session::builder()?.commit_from_file(decoder)?;
    let pad = get_word_id(false, "<pad>")?;
    let sos = get_word_id(false, "<sos>")?;
    let eos = get_word_id(false, "<eos>")?;

    let src = get_word_ids(true, &split_word(en_text.as_ref().to_lowercase())?)?;
    let src = Array::from_shape_vec((1, src.len()), src)?;
    let src_pad_mask = Array::from_elem((1, src.len(), src.len()), false);
    let options = RunOptions::new()?;

    Ok(stream! {
        let src_tensor = TensorRef::from_array_view(&src)?;
        let src_pad_mask_tensor = TensorRef::from_array_view(&src_pad_mask)?;
        let output = encoder.run_async(
            inputs![
                "src" => src_tensor,
                "src_pad_mask" => src_pad_mask_tensor,
            ],
            &options,
        )?
        .await?;
        let memory = output["memory"].try_extract_array::<f32>()?;

        let mut ys = Vec::with_capacity(MAX_LEN);
        ys.push(sos);
        for i in 0..MAX_LEN {
            let tgt = Array::from_shape_vec((1, ys.len()), ys.clone())?;
            let tgt_pad_mask = Array::from_elem((1, tgt.len(), tgt.len()), false);
            let enc_pad_mask = Array::from_elem((1, tgt.len(), src.len()), false);
            let tgt_tensor = TensorRef::from_array_view(&tgt)?;
            let memory_tensor = TensorRef::from_array_view(memory.clone())?;
            let t = SystemTime::now();
            let tgt_pad_mask_tensor = TensorRef::from_array_view(&tgt_pad_mask)?;
            let enc_pad_mask_tensor = TensorRef::from_array_view(&enc_pad_mask)?;
            let output = decoder.run_async(
                inputs![
                    "tgt" => tgt_tensor,
                    "memory" => memory_tensor,
                    "tgt_pad_mask" => tgt_pad_mask_tensor,
                    "enc_pad_mask" => enc_pad_mask_tensor,
                ],
                &options,
            )?
            .await?;
            let output = output["output"].try_extract_array::<f32>()?;
            let prob = output
                .slice_axis(Axis(1), Slice::new(i as _, None, 1))
                .as_slice()
                .map(|i| {
                    i.iter()
                        .enumerate()
                        .max_by(|i, j| i.1.total_cmp(&j.1))
                        .map(|i| i.0 as _)
                        .unwrap_or(pad)
                })
                .unwrap_or_default();

            if eos == prob {
                break;
            }
            ys.push(prob);
            yield Ok((get_word(false, prob)?, t.elapsed()?));
        }
    })
}
