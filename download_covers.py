#!/usr/bin/env python3
"""
Download album cover art from Apple's iTunes / Apple Music catalog at the
highest resolution the CDN will serve (the original master, typically
1400x1400 up to 4000x4000 or more).

Usage:
    python download_covers.py albums.txt
    python download_covers.py albums.txt --out covers --country us,gb --png

Input file format (one album per line):
    Artist - Album
    Album                                   (artist unknown / compilations - less reliable)
    https://music.apple.com/us/album/x/123  (exact album by Apple Music / iTunes URL)
    id:123456789                            (exact album by iTunes collection ID)
    # lines starting with # are comments, blank lines are ignored

Only the Python standard library is used.
"""

import argparse
import csv
import difflib
import json
import re
import struct
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SEARCH_URL = "https://itunes.apple.com/search"
LOOKUP_URL = "https://itunes.apple.com/lookup"
USER_AGENT = "Mozilla/5.0 (album-cover-downloader)"

# Separators accepted between artist and album (hyphen, en/em dash, pipe, slash).
SEPARATOR_RE = re.compile(r"\s+(?:-|–|—|\||/)\s+")
URL_ID_RE = re.compile(r"(?:music\.apple\.com|itunes\.apple\.com)/.*?/album/(?:[^/]+/)?(?:id)?(\d+)", re.I)
RAW_ID_RE = re.compile(r"^id:\s*(\d+)$", re.I)

# Size specifiers tried in order. "<W>x<H>-999" = original resolution capped at
# W x H, maximum JPEG quality. Apple never upscales, so 10000 yields the master
# size. Later entries are fallbacks if the CDN rejects a spec (HTTP 400).
SIZE_LADDER = ["{px}x{px}-999", "{px}x{px}bb-999", "5000x5000-999", "3000x3000bb-100",
               "3000x3000bb", "1400x1400bb", "600x600bb"]

# Words in a *result* title that indicate a different product than the studio
# album, unless the user's own text also contains them.
DIFFERENT_PRODUCT = ["live", "karaoke", "tribute", "instrumental", "cover", "covers", "performs",
                     "in the style of", "made famous", "commentary", "remixes", "acoustic",
                     "unplugged", "soundtrack", "sessions", "demos", "8 bit", "8-bit", "lullaby",
                     "orchestral", "piano version", "string quartet"]
# Parenthetical words that merely mark an edition of the same album.
EDITION_WORDS = re.compile(r"\b(remaster(ed)?|deluxe|edition|expanded|anniversary|bonus|version|"
                           r"mix|remix|stereo|mono|explicit|clean|special|collector'?s|super|"
                           r"digital|reissue|redux|\d{4})\b", re.I)


# --------------------------------------------------------------------------- helpers
def log(msg: str) -> None:
    print(msg, flush=True)


def normalize(s: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("&", " and ")
    s = re.sub(r"['’‘`]", "", s)          # can't == cant
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_edition_tags(title: str) -> str:
    """Remove ' - Single', ' - EP' and parentheticals made only of edition words."""
    t = re.sub(r"\s+-\s+(single|ep)\s*$", "", title, flags=re.I)

    def keep(m):
        inner = m.group(1)
        return "" if not EDITION_WORDS.sub("", inner).strip(" &,'-/") else m.group(0)

    t = re.sub(r"\s*[\(\[]([^\)\]]*)[\)\]]", keep, t)
    return t.strip(" -:")


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def artist_similarity(query: str, candidate: str) -> float:
    q, c = normalize(query), normalize(candidate)
    if q == c:
        return 1.0
    # "The Beatles" vs "Beatles", or "Artist & Friend" containing the artist
    if q in c.split(" and ") or c in q.split(" and "):
        return 0.9
    return ratio(q, c)


def album_similarity(query: str, candidate: str) -> float:
    q, c = normalize(query), normalize(candidate)
    if q == c:
        score = 1.0
    elif normalize(strip_edition_tags(candidate)) == normalize(strip_edition_tags(query)):
        score = 0.98            # same album, different edition tag
    else:
        base = ratio(q, c)
        if c.startswith(q + " ") or c.startswith(q + " ("):
            base = max(base, 0.85)
        score = base
    for word in DIFFERENT_PRODUCT:
        if word in c and word not in q:
            score -= 0.3
            break
    if re.search(r"\s-\s(single|ep)$", candidate.lower()) and not re.search(r"\b(single|ep)\b", q):
        score -= 0.25
    return score


def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:150] or "untitled"


def image_dimensions(data: bytes):
    """Return (width, height) for JPEG or PNG bytes, or None."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", data[16:24])
    if data[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                return None
            marker = data[i + 1]
            length = struct.unpack(">H", data[i + 2:i + 4])[0]
            if marker in (0xC0, 0xC1, 0xC2):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h
            i += 2 + length
    return None


def http_get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    """GET with exponential backoff on rate limiting / transient errors."""
    delay = 2.0
    for attempt in range(retries):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 429, 500, 502, 503) and attempt < retries - 1:
                log(f"    HTTP {e.code}, retrying in {delay:.0f}s ...")
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries - 1:
                log(f"    network error ({e}), retrying in {delay:.0f}s ...")
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------- iTunes API
class ITunes:
    def __init__(self, delay: float):
        self.delay = delay
        self.artist_cache = {}

    def _api(self, url: str, **params) -> list:
        data = http_get(f"{url}?{urllib.parse.urlencode(params)}")
        time.sleep(self.delay)
        return json.loads(data.decode("utf-8")).get("results", [])

    def search_albums(self, term: str, country: str, limit: int = 50) -> list:
        return self._api(SEARCH_URL, term=term, entity="album", media="music",
                         country=country, limit=limit)

    def lookup_album(self, collection_id: str, country: str):
        res = self._api(LOOKUP_URL, id=collection_id, entity="album", country=country)
        return next((r for r in res if r.get("wrapperType") == "collection"), None)

    def artist_discography(self, artist: str, country: str) -> list:
        """All albums of the best-matching artist (cached per artist/store)."""
        key = (normalize(artist), country)
        if key in self.artist_cache:
            return self.artist_cache[key]
        albums = []
        artists = self._api(SEARCH_URL, term=artist, entity="musicArtist", media="music",
                            country=country, limit=10)
        ranked = sorted(((artist_similarity(artist, a["artistName"]) - 0.01 * i, a)
                         for i, a in enumerate(artists)), key=lambda t: t[0], reverse=True)
        if ranked and ranked[0][0] >= 0.8:
            res = self._api(LOOKUP_URL, id=ranked[0][1]["artistId"], entity="album",
                            country=country, limit=200)
            albums = [r for r in res if r.get("wrapperType") == "collection"]
        self.artist_cache[key] = albums
        return albums


# --------------------------------------------------------------------------- matching
def parse_line(line: str):
    """Return (kind, artist, album): kind is 'id' (album = collection id) or 'name'."""
    m = URL_ID_RE.search(line) or RAW_ID_RE.match(line)
    if m:
        return "id", None, m.group(1)
    parts = SEPARATOR_RE.split(line, maxsplit=1)
    if len(parts) == 2:
        return "name", parts[0].strip(), parts[1].strip()
    return "name", None, line.strip()


def read_album_list(path: Path):
    entries = []
    with open(path, encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            entries.append((lineno, line, *parse_line(line)))
    return entries


def find_album(api: ITunes, artist, album, country: str):
    """Return (best_result, score) or (None, 0). Score is 0-1 confidence."""
    candidates = {}   # collectionId -> (score, result)

    def consider(result, artist_score, rank):
        cid = result.get("collectionId")
        if not cid or not result.get("artworkUrl100"):
            return
        a = album_similarity(album, result.get("collectionName", ""))
        score = (0.4 * artist_score + 0.6 * a) if artist else a
        score -= 0.01 * rank                              # prefer Apple's own ranking on ties
        if result.get("collectionType") == "Compilation" and "various" not in normalize(artist or ""):
            score -= 0.02
        if cid not in candidates or candidates[cid][0] < score:
            candidates[cid] = (score, result)

    if artist:
        # 1) Verified discography of the artist - most reliable source.
        for r in api.artist_discography(artist, country):
            consider(r, 1.0, 0)
        # 2) Free-text search catches albums credited to "Artist & X", etc.
        for i, r in enumerate(api.search_albums(f"{artist} {album}", country)):
            consider(r, artist_similarity(artist, r.get("artistName", "")), i)
    else:
        for i, r in enumerate(api.search_albums(album, country)):
            consider(r, 0.0, i)
        if not candidates:
            for i, r in enumerate(api.search_albums(strip_edition_tags(album), country)):
                consider(r, 0.0, i)

    if not candidates:
        return None, 0.0
    score, result = max(candidates.values(), key=lambda t: t[0])
    return result, max(0.0, min(1.0, score))


def artwork_candidates(artwork_url: str, max_px: int, png: bool) -> list:
    base = artwork_url.rsplit("/", 1)[0]
    ext = "png" if png else "jpg"
    return [f"{base}/{spec.format(px=max_px)}.{ext}" for spec in SIZE_LADDER]


def download_artwork(artwork_url: str, max_px: int, png: bool):
    last_err = None
    for url in artwork_candidates(artwork_url, max_px, png):
        try:
            return http_get(url, retries=2), url
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 400:      # CDN rejected this size spec, try the next one
                continue
            raise
    raise RuntimeError(f"all size variants failed ({last_err})")


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("list_file", type=Path, help="text file with one album per line")
    ap.add_argument("--out", type=Path, default=Path("covers"), help="output folder (default: covers)")
    ap.add_argument("--country", default="us",
                    help="iTunes storefront(s) to try in order, comma-separated 2-letter codes (default: us)")
    ap.add_argument("--png", action="store_true", help="save lossless PNG instead of max-quality JPEG")
    ap.add_argument("--max-px", type=int, default=10000,
                    help="cap on requested side length; Apple never upscales (default: 10000 = original)")
    ap.add_argument("--min-score", type=float, default=0.7,
                    help="skip matches scoring below this 0-1 confidence (default: 0.7)")
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between API calls (default: 0.5)")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = ap.parse_args()

    if not args.list_file.is_file():
        log(f"error: {args.list_file} not found")
        return 1
    entries = read_album_list(args.list_file)
    if not entries:
        log("error: no albums found in list file")
        return 1

    countries = [c.strip().lower() for c in args.country.split(",") if c.strip()]
    api = ITunes(args.delay)
    args.out.mkdir(parents=True, exist_ok=True)
    ext = "png" if args.png else "jpg"
    report_path = args.out / "report.csv"
    rows = []
    ok = skipped = failed = 0

    log(f"{len(entries)} album(s) -> {args.out.resolve()}  [stores: {', '.join(countries)}]")
    for idx, (lineno, line, kind, artist, album) in enumerate(entries, 1):
        prefix = f"[{idx}/{len(entries)}] {line}"
        target = None if kind == "id" else args.out / f"{safe_filename(line)}.{ext}"
        if target and target.exists() and not args.force:
            log(f"{prefix}: exists, skipping")
            skipped += 1
            rows.append([lineno, line, "skipped (exists)", "", "", "", "", target.name])
            continue

        try:
            result, score = None, 0.0
            for country in countries:
                if kind == "id":
                    result = api.lookup_album(album, country)
                    score = 1.0 if result else 0.0
                else:
                    result, score = find_album(api, artist, album, country)
                if result and score >= args.min_score:
                    break

            if result is None:
                log(f"{prefix}: NOT FOUND")
                failed += 1
                rows.append([lineno, line, "not found", "", "", "", "", ""])
                continue

            matched = (f"{result.get('artistName')} - {result.get('collectionName')} "
                       f"({(result.get('releaseDate') or '')[:4]})")
            if score < args.min_score:
                log(f"{prefix}: LOW CONFIDENCE ({score:.2f}) best guess '{matched}' - skipped")
                failed += 1
                rows.append([lineno, line, f"low confidence {score:.2f}", result.get("artistName"),
                             result.get("collectionName"), f"{score:.2f}", "", ""])
                continue

            if target is None:
                target = args.out / f"{safe_filename(f'{result.get('artistName')} - {result.get('collectionName')}')}.{ext}"
                if target.exists() and not args.force:
                    log(f"{prefix}: exists, skipping ({target.name})")
                    skipped += 1
                    rows.append([lineno, line, "skipped (exists)", "", "", "", "", target.name])
                    continue

            data, _ = download_artwork(result["artworkUrl100"], args.max_px, args.png)
            target.write_bytes(data)
            dims = image_dimensions(data)
            dims_s = f"{dims[0]}x{dims[1]}" if dims else "?"
            flag = "" if score >= 0.9 else f"  <-- verify (score {score:.2f})"
            log(f"{prefix}: OK  {matched}  {dims_s}  {len(data) / 1e6:.1f} MB{flag}")
            ok += 1
            rows.append([lineno, line, "ok", result.get("artistName"), result.get("collectionName"),
                         f"{score:.2f}", dims_s, target.name])
        except Exception as e:  # keep going on any single failure
            log(f"{prefix}: ERROR {e}")
            failed += 1
            rows.append([lineno, line, f"error: {e}", "", "", "", "", ""])

    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["line", "input", "status", "matched_artist", "matched_album", "score", "dimensions", "file"])
        w.writerows(rows)

    log(f"\nDone: {ok} downloaded, {skipped} skipped, {failed} failed.  Report: {report_path}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
