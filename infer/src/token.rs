use {
    super::TranslationError,
    std::{collections::HashMap, sync::LazyLock},
};

static SRC_VOCAB: LazyLock<Result<(HashMap<&str, i64>, HashMap<i64, &str>), TranslationError>> =
    LazyLock::new(|| get_vocab_map(include_str!("../../data/vocab_source.txt")));
static TGT_VOCAB: LazyLock<Result<(HashMap<&str, i64>, HashMap<i64, &str>), TranslationError>> =
    LazyLock::new(|| get_vocab_map(include_str!("../../data/vocab_target.txt")));

fn get_vocab_map(
    vocab: &'static str,
) -> Result<(HashMap<&'static str, i64>, HashMap<i64, &'static str>), TranslationError> {
    let lines = vocab.split('\n').collect::<Vec<_>>();
    let items = lines
        .iter()
        .filter_map(|i| {
            let mut j = i.trim_end().split('\t');
            if let (Some(w), Some(id)) = (j.next(), j.next()) {
                Some((w, id))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let mut word2id = HashMap::with_capacity(items.len());
    let mut id2word = HashMap::with_capacity(items.len());
    for (w, id) in items {
        let id = id.parse()?;
        word2id.insert(w, id);
        id2word.insert(id, w);
    }

    Ok((word2id, id2word))
}

pub fn get_word_id<S>(is_src_vocab: bool, word: S) -> Result<i64, TranslationError>
where
    S: AsRef<str>,
{
    let vocab = if is_src_vocab {
        SRC_VOCAB.as_ref()
    } else {
        TGT_VOCAB.as_ref()
    }
    .map_err(|e| e.to_owned())?;
    vocab.0.get(word.as_ref()).map_or(
        Err(TranslationError::WordTokenNotFound(
            word.as_ref().to_owned(),
        )),
        |i| Ok(*i),
    )
}

pub fn get_word_ids<S>(is_src_vocab: bool, words: &[S]) -> Result<Vec<i64>, TranslationError>
where
    S: AsRef<str> + Copy,
{
    let mut ids = Vec::with_capacity(words.len());
    for i in 0..words.len() {
        ids.push(get_word_id(is_src_vocab, words[i])?);
    }

    Ok(ids)
}

pub fn get_word(is_src_vocab: bool, id: i64) -> Result<&'static str, TranslationError> {
    let vocab = if is_src_vocab {
        SRC_VOCAB.as_ref()
    } else {
        TGT_VOCAB.as_ref()
    }
    .map_err(|e| e.to_owned())?;
    vocab
        .1
        .get(&id)
        .map_or(Err(TranslationError::WordIdTokenNotFound(id)), |i| Ok(*i))
}

pub fn get_words(is_src_vocab: bool, ids: &[i64]) -> Result<Vec<&'static str>, TranslationError> {
    let mut words = Vec::with_capacity(ids.len());
    for id in ids {
        words.push(get_word(is_src_vocab, *id)?);
    }

    Ok(words)
}
