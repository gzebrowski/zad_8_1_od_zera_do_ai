import re
import unicodedata


def slugify(value, words_sep='-'):
    value = str(value).replace('ł', 'l').replace('Ł', 'l')
    value = value
    value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    value = re.sub(r"[^\w\s%s]" % words_sep, "", value.lower())
    return re.sub(r"[%s\s]+" % words_sep, words_sep, value).strip("-_")
